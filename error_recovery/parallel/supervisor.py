"""Run one isolated collection process per worker and restart it on failure."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .io_util import append_jsonl, write_json_atomic
from .resources import (
    assign_cores,
    gpu_memory_by_pid,
    gpu_used_mb_by_index,
    read_load_average,
    read_mem_available_mb,
    read_rss_mb,
    thread_env,
)
from .sharding import ATTEMPTS_PER_EPISODE, ResumePlan, Shard, load_events, plan_shards, resume_plan


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLECT_SCRIPT = REPO_ROOT / "error_recovery" / "collect_universal.py"


@dataclass
class SupervisorConfig:
    task: str
    episodes: int
    gpus: List[str]
    workers_per_gpu: int = 2
    seed_start: int = 0
    save_path: str = ""
    max_episodes_per_process: int = 50
    max_restarts: int = 5
    episode_timeout_s: float = 1200.0
    max_rss_gb: float = 16.0
    max_vram_gb: float = 16.0
    min_free_gb: float = 32.0
    reserve_cores: int = 8
    core_offset: int = 0
    core_count: int = 0
    nice: int = 10
    poll_s: float = 2.0
    term_grace_s: float = 5.0
    status_interval_s: float = 10.0
    passthrough: List[str] = field(default_factory=list)
    python: str = ""
    worker_prefix: List[str] = field(default_factory=list)
    enable_affinity: bool = True
    enable_nice: bool = True
    rss_fn: Optional[Callable[[int], Optional[float]]] = None
    mem_available_fn: Optional[Callable[[], Optional[float]]] = None
    vram_fn: Optional[Callable[[], Dict[int, float]]] = None
    collect_script: str = ""


@dataclass
class WorkerSlot:
    shard: Shard
    cores: List[int]
    active_seed_end: int
    proc: Optional[subprocess.Popen] = None
    log_handle: Any = None
    launches: int = 0
    crash_restarts: int = 0
    state: str = "pending"
    killed_by_supervisor: bool = False
    kill_reason: str = ""
    last_was_crash: bool = False
    started_at: float = 0.0
    saved_at_launch: int = 0


class Supervisor:
    def __init__(self, config: SupervisorConfig):
        self.config = config
        if not config.save_path:
            config.save_path = str(REPO_ROOT / "error_recovery" / "data" / config.task)
        if not config.python:
            config.python = sys.executable
        if not config.collect_script:
            config.collect_script = str(DEFAULT_COLLECT_SCRIPT)
        self.started_at = time.time()
        self._orphan_notice: set = set()
        self.last_status_at = 0.0
        self.slots: List[WorkerSlot] = []
        self.overflow = config.seed_start

    def run(self) -> int:
        self._install_shutdown_hooks()
        shards = plan_shards(
            self.config.episodes,
            self.config.gpus,
            self.config.workers_per_gpu,
            self.config.seed_start,
        )
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.save_path, "events"), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_path, "logs"), exist_ok=True)
        core_sets = assign_cores(
            len(shards),
            reserve=self.config.reserve_cores,
            offset=self.config.core_offset,
            length=self.config.core_count or None,
        )
        self.overflow = max((shard.seed_end for shard in shards), default=self.config.seed_start)
        self.slots = [
            WorkerSlot(shard=shard, cores=core_sets[index] if index < len(core_sets) else [], active_seed_end=shard.seed_end)
            for index, shard in enumerate(shards)
        ]
        if not self.slots:
            self._write_status()
            self._merge_stats()
            return 0

        while any(slot.state not in ("done", "failed") for slot in self.slots):
            for slot in self.slots:
                if slot.state == "running":
                    self._apply_limits(slot)
                    if self._reap(slot):
                        self._decide_after_exit(slot)
                if slot.state == "pending":
                    self._launch(slot)
            self._apply_memory_floor()
            self._apply_device_vram_cap()
            self._write_status()
            if any(slot.state not in ("done", "failed") for slot in self.slots):
                time.sleep(self.config.poll_s)

        self._write_status(force=True)
        self._merge_stats()
        saved = self._saved_total()
        failed = any(slot.state == "failed" for slot in self.slots)
        return 0 if not failed and saved >= self.config.episodes else 1

    def _events_path(self, slot: WorkerSlot) -> str:
        return os.path.join(self.config.save_path, "events", f"{slot.shard.worker_id}.jsonl")

    def _resume(self, slot: WorkerSlot) -> ResumePlan:
        return resume_plan(
            load_events(self._events_path(slot)),
            slot.shard.episode_start,
            slot.shard.episode_end,
            slot.shard.seed_start,
            slot.active_seed_end,
        )

    def _saved_count(self, slot: WorkerSlot) -> int:
        plan = self._resume(slot)
        return plan.next_episode - slot.shard.episode_start

    def _saved_total(self) -> int:
        return sum(self._saved_count(slot) for slot in self.slots)

    def _worker_still_running(self, event: Dict[str, Any]) -> bool:
        """True when the process that wrote this start event is still collecting."""
        pid = event.get("pid")
        if pid is None:
            return False
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                command = handle.read().replace(b"\x00", b" ").decode(errors="ignore")
        except OSError:
            return False
        return "collect_universal.py" in command and self.config.save_path in command

    def _launch(self, slot: WorkerSlot) -> None:
        plan = self._resume(slot)
        if plan.open_attempt and self._worker_still_running(plan.open_attempt):
            if slot.shard.worker_id not in self._orphan_notice:
                self._orphan_notice.add(slot.shard.worker_id)
                print(
                    f"[supervisor] wait {slot.shard.worker_id} pid={plan.open_attempt.get('pid')} still running",
                    flush=True,
                )
            return
        if plan.open_attempt:
            self._close_open_attempt(slot, "crash", "interrupted")
            plan = self._resume(slot)
        if plan.remaining_episodes <= 0:
            slot.state = "done"
            return
        seed_start = plan.next_seed
        seed_end = slot.active_seed_end
        if seed_start >= seed_end:
            extra = max(1, plan.remaining_episodes) * ATTEMPTS_PER_EPISODE
            seed_start = self.overflow
            seed_end = self.overflow + extra
            self.overflow = seed_end
            slot.active_seed_end = seed_end
        command = self._build_command(slot, plan.remaining_episodes, seed_start, seed_end, plan.next_episode)
        env = os.environ.copy()
        env.update(thread_env(max(1, len(slot.cores))))
        env["CUDA_VISIBLE_DEVICES"] = slot.shard.gpu
        env["FRBENCH_CUROBO_CORRUPTION_EXIT"] = "1"
        env["FRBENCH_CUROBO_STRICT"] = "1"
        log_path = os.path.join(self.config.save_path, "logs", f"{slot.shard.worker_id}.out")
        log_handle = open(log_path, "ab", buffering=0)
        slot.saved_at_launch = self._saved_count(slot)
        slot.killed_by_supervisor = False
        slot.kill_reason = ""
        slot.started_at = time.time()
        slot.proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=self._preexec(slot),
        )
        slot.log_handle = log_handle
        slot.launches += 1
        slot.state = "running"
        self._ionice(slot.proc.pid)
        print(
            f"[supervisor] start {slot.shard.worker_id} gpu={slot.shard.gpu} pid={slot.proc.pid} "
            f"episodes={plan.next_episode}:{slot.shard.episode_end} seeds={seed_start}:{seed_end} "
            f"cores={len(slot.cores)} launch={slot.launches}",
            flush=True,
        )

    def _preexec(self, slot: WorkerSlot):
        cores = set(slot.cores)
        nice_level = self.config.nice
        enable_affinity = self.config.enable_affinity
        enable_nice = self.config.enable_nice

        def _inner() -> None:
            if enable_nice and nice_level:
                try:
                    os.nice(nice_level)
                except OSError:
                    pass
            if enable_affinity and cores:
                try:
                    os.sched_setaffinity(0, cores)
                except OSError:
                    pass

        return _inner

    @staticmethod
    def _ionice(pid: int) -> None:
        try:
            subprocess.run(
                ["ionice", "-c", "2", "-n", "7", "-p", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            pass

    def _build_command(
        self,
        slot: WorkerSlot,
        episodes: int,
        seed_start: int,
        seed_end: int,
        episode_offset: int,
    ) -> List[str]:
        if self.config.worker_prefix:
            command = list(self.config.worker_prefix)
        else:
            command = [self.config.python, self.config.collect_script]
        command.extend(self.config.passthrough)
        command.extend(
            [
                "--task",
                self.config.task,
                "--episodes",
                str(episodes),
                "--save_path",
                self.config.save_path,
                "--seed_start",
                str(seed_start),
                "--seed_end",
                str(seed_end),
                "--episode_offset",
                str(episode_offset),
                "--worker_id",
                slot.shard.worker_id,
                "--events_path",
                self._events_path(slot),
                "--fail_fast",
                "--max_episodes_per_process",
                str(self.config.max_episodes_per_process),
                "--gpu",
                slot.shard.gpu,
            ]
        )
        return command

    def _rss(self, pid: int) -> Optional[float]:
        if self.config.rss_fn is not None:
            return self.config.rss_fn(pid)
        return read_rss_mb(pid)

    def _mem_available(self) -> Optional[float]:
        if self.config.mem_available_fn is not None:
            return self.config.mem_available_fn()
        return read_mem_available_mb()

    def _vram_map(self) -> Dict[int, float]:
        if self.config.vram_fn is not None:
            return self.config.vram_fn()
        return gpu_memory_by_pid()

    def _apply_limits(self, slot: WorkerSlot) -> None:
        proc = slot.proc
        if proc is None or proc.poll() is not None:
            return
        rss = self._rss(proc.pid)
        if rss is not None and rss > self.config.max_rss_gb * 1024.0:
            self._terminate(slot, "rss")
            return
        vram = self._vram_map().get(proc.pid)
        if vram is not None and vram > self.config.max_vram_gb * 1024.0:
            self._terminate(slot, "vram")
            return
        if self._attempt_timed_out(slot):
            self._terminate(slot, "timeout")

    def _install_shutdown_hooks(self) -> None:
        import atexit

        atexit.register(self._stop_workers)
        signal.signal(signal.SIGTERM, self._on_shutdown_signal)
        signal.signal(signal.SIGINT, self._on_shutdown_signal)

    def _on_shutdown_signal(self, signum, _frame) -> None:
        self._stop_workers()
        raise SystemExit(128 + int(signum))

    def _stop_workers(self) -> None:
        for slot in self.slots:
            proc = slot.proc
            if proc is None or proc.poll() is not None:
                continue
            self._signal(proc.pid, signal.SIGTERM)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if all(slot.proc is None or slot.proc.poll() is not None for slot in self.slots):
                return
            time.sleep(0.05)
        for slot in self.slots:
            proc = slot.proc
            if proc is not None and proc.poll() is None:
                self._signal(proc.pid, signal.SIGKILL)

    def _apply_device_vram_cap(self) -> None:
        """Kill a worker when its GPU, not an individual PID, exceeds the cap.

        Some drivers report host PIDs that this process cannot see, so the
        per-PID reading stays empty. The device total is still meaningful.
        """
        if self.config.vram_fn is not None:
            return
        running: Dict[str, List[WorkerSlot]] = {}
        for slot in self.slots:
            if slot.proc is not None and slot.proc.poll() is None:
                running.setdefault(slot.shard.gpu, []).append(slot)
        if not running:
            return
        usage = gpu_used_mb_by_index()
        for gpu, slots in running.items():
            used = usage.get(str(gpu))
            if used is None:
                continue
            cap = self.config.max_vram_gb * 1024.0 * len(slots)
            if used <= cap:
                continue
            victim = max(slots, key=lambda slot: self._rss(slot.proc.pid) or 0.0)
            self._terminate(victim, "vram")
            return

    def _apply_memory_floor(self) -> None:
        available = self._mem_available()
        if available is None or available >= self.config.min_free_gb * 1024.0:
            return
        running = [slot for slot in self.slots if slot.proc is not None and slot.proc.poll() is None]
        if not running:
            return
        victim = max(running, key=lambda slot: self._rss(slot.proc.pid) or 0.0)
        self._terminate(victim, "mem")

    def _attempt_timed_out(self, slot: WorkerSlot) -> bool:
        events = load_events(self._events_path(slot))
        now = time.time()
        if not events:
            return now - slot.started_at > self.config.episode_timeout_s
        last = events[-1]
        try:
            stamp = float(last.get("ts"))
        except (TypeError, ValueError):
            stamp = slot.started_at
        return now - stamp > self.config.episode_timeout_s

    def _terminate(self, slot: WorkerSlot, reason: str) -> None:
        proc = slot.proc
        if proc is None or proc.poll() is not None:
            return
        slot.killed_by_supervisor = True
        slot.kill_reason = reason
        print(f"[supervisor] stop {slot.shard.worker_id} pid={proc.pid} reason={reason}", flush=True)
        self._signal(proc.pid, signal.SIGTERM)
        deadline = time.time() + self.config.term_grace_s
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        self._signal(proc.pid, signal.SIGKILL)

    @staticmethod
    def _signal(pid: int, sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                return

    def _reap(self, slot: WorkerSlot) -> bool:
        proc = slot.proc
        if proc is None:
            return False
        return_code = proc.poll()
        if return_code is None:
            return False
        if slot.log_handle is not None:
            try:
                slot.log_handle.close()
            except OSError:
                pass
            slot.log_handle = None
        outcome = "crash"
        if slot.killed_by_supervisor and slot.kill_reason == "timeout":
            outcome = "timeout"
        if return_code != 0 or slot.killed_by_supervisor:
            self._close_open_attempt(slot, outcome, slot.kill_reason or f"exit {return_code}")
        saved_now = self._saved_count(slot)
        made_progress = saved_now > slot.saved_at_launch
        clean_exit = return_code == 0 and not slot.killed_by_supervisor
        if clean_exit and made_progress:
            slot.last_was_crash = False
        elif clean_exit and not made_progress:
            slot.crash_restarts += 1
            slot.last_was_crash = True
            self._close_open_attempt(slot, "crash", "no progress")
        else:
            slot.crash_restarts += 1
            slot.last_was_crash = True
        print(
            f"[supervisor] exit {slot.shard.worker_id} code={return_code} "
            f"saved={saved_now}/{slot.shard.episodes} restarts={slot.crash_restarts}",
            flush=True,
        )
        slot.proc = None
        return True

    def _close_open_attempt(self, slot: WorkerSlot, outcome: str, reason: str) -> None:
        plan = self._resume(slot)
        if not plan.open_attempt:
            return
        event = {
            "ts": time.time(),
            "type": "attempt",
            "worker_id": slot.shard.worker_id,
            "gpu": slot.shard.gpu,
            "pid": plan.open_attempt.get("pid"),
            "episode_id": plan.open_attempt.get("episode_id"),
            "seed": plan.open_attempt.get("seed"),
            "outcome": outcome,
            "reason": reason,
        }
        append_jsonl(self._events_path(slot), event)

    def _decide_after_exit(self, slot: WorkerSlot) -> None:
        plan = self._resume(slot)
        if plan.remaining_episodes <= 0:
            slot.state = "done"
            return
        if slot.last_was_crash and slot.crash_restarts > self.config.max_restarts:
            slot.state = "failed"
            print(
                f"[supervisor] give up {slot.shard.worker_id} after {slot.crash_restarts} failed runs",
                flush=True,
            )
            return
        slot.state = "pending"

    def _write_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_status_at < self.config.status_interval_s:
            return
        self.last_status_at = now
        vram = self._vram_map()
        saved = self._saved_total()
        elapsed = max(now - self.started_at, 1e-6)
        rate = saved / elapsed * 3600.0
        remaining = max(0, self.config.episodes - saved)
        eta_s = remaining / (saved / elapsed) if saved else None
        success, recovery = self._rates()
        workers = []
        for slot in self.slots:
            pid = slot.proc.pid if slot.proc is not None and slot.proc.poll() is None else None
            workers.append(
                {
                    "id": slot.shard.worker_id,
                    "gpu": slot.shard.gpu,
                    "pid": pid,
                    "state": slot.state,
                    "launches": slot.launches,
                    "restarts": slot.crash_restarts,
                    "done": self._saved_count(slot),
                    "target": slot.shard.episodes,
                    "rss_mb": self._rss(pid) if pid else None,
                    "vram_mb": vram.get(pid) if pid else None,
                    "cores": slot.cores,
                }
            )
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "task": self.config.task,
            "episodes_done": saved,
            "episodes_total": self.config.episodes,
            "success_rate": success,
            "recovery_rate": recovery,
            "episodes_per_hour": rate,
            "eta_s": eta_s,
            "load_1m": read_load_average(),
            "workers": workers,
        }
        write_json_atomic(os.path.join(self.config.save_path, "status.json"), payload)
        eta_txt = "n/a" if eta_s is None else f"{eta_s / 60.0:.1f}m"
        parts = [
            f"{item['id']} {item['state']} {item['done']}/{item['target']} r={item['restarts']}"
            for item in workers
        ]
        print(
            f"[supervisor] {saved}/{self.config.episodes} ok={success:.0%} recovery={recovery:.0%} "
            f"{rate:.1f} ep/h eta={eta_txt} | " + " ".join(parts),
            flush=True,
        )

    def _rates(self):
        saved = 0
        succeeded = 0
        injected = 0
        recovered = 0
        for slot in self.slots:
            for event in load_events(self._events_path(slot)):
                if event.get("type") != "attempt" or event.get("outcome") != "saved":
                    continue
                saved += 1
                if event.get("success"):
                    succeeded += 1
                error_types = event.get("error_types") or []
                if error_types:
                    injected += 1
                    if event.get("recovery") == "success":
                        recovered += 1
        success_rate = succeeded / saved if saved else 0.0
        recovery_rate = recovered / injected if injected else 0.0
        return success_rate, recovery_rate

    def _merge_stats(self) -> None:
        episodes: Dict[int, Dict[str, Any]] = {}
        task_name = self.config.task
        for slot in self.slots:
            stats_path = os.path.join(
                self.config.save_path, f"collection_stats_{slot.shard.worker_id}.json"
            )
            if not os.path.exists(stats_path):
                continue
            try:
                import json
                with open(stats_path, "r", encoding="utf-8") as handle:
                    stats = json.load(handle)
            except (OSError, ValueError):
                continue
            task_name = stats.get("task_name") or task_name
            for episode in stats.get("episodes") or []:
                if not isinstance(episode, dict) or episode.get("episode_id") is None:
                    continue
                episodes[int(episode["episode_id"])] = episode
        ordered = [episodes[key] for key in sorted(episodes)]
        successful = sum(1 for episode in ordered if episode.get("success"))
        injected = sum(1 for episode in ordered if episode.get("error_injected"))
        recovered = sum(1 for episode in ordered if episode.get("final_phase") == "recovered")
        total = len(ordered)
        payload = {
            "task_name": task_name,
            "save_path": self.config.save_path,
            "total_episodes": total,
            "successful_episodes": successful,
            "error_injected_episodes": injected,
            "error_recovery_episodes": recovered,
            "success_rate": successful / total if total else 0.0,
            "error_recovery_rate": recovered / injected if injected else 0.0,
            "episodes": ordered,
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        write_json_atomic(os.path.join(self.config.save_path, "collection_stats.json"), payload)
