#!/usr/bin/env python3
"""Supervisor unit tests. Run as a script so the simulator package is not imported."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Keep the real error_recovery/__init__.py from pulling in SAPIEN/CUDA.
if "error_recovery" not in sys.modules:
    package = types.ModuleType("error_recovery")
    package.__path__ = [os.path.join(ROOT, "error_recovery")]
    package.__package__ = "error_recovery"
    sys.modules["error_recovery"] = package

from error_recovery.parallel.io_util import append_jsonl, write_text_atomic
from error_recovery.parallel.resources import assign_cores
from error_recovery.parallel.sharding import load_events, plan_shards, resume_plan
from error_recovery.parallel.supervisor import Supervisor, SupervisorConfig, WorkerSlot

FAKE_WORKER = os.path.join(ROOT, "error_recovery", "tests", "fake_worker.py")


def _config(tmp: str, **overrides) -> SupervisorConfig:
    values = dict(
        task="move_can_pot",
        episodes=2,
        gpus=["0"],
        workers_per_gpu=1,
        seed_start=0,
        save_path=tmp,
        max_episodes_per_process=50,
        max_restarts=5,
        episode_timeout_s=30.0,
        max_rss_gb=64.0,
        max_vram_gb=40.0,
        min_free_gb=0.0,
        reserve_cores=0,
        nice=0,
        poll_s=0.05,
        term_grace_s=0.3,
        status_interval_s=0.2,
        passthrough=[],
        python=sys.executable,
        worker_prefix=[sys.executable, FAKE_WORKER, "--mode", "ok"],
        enable_affinity=False,
        enable_nice=False,
        rss_fn=lambda _pid: 32.0,
        mem_available_fn=lambda: 1_000_000.0,
        vram_fn=lambda: {},
    )
    values.update(overrides)
    return SupervisorConfig(**values)


class ShardingTest(unittest.TestCase):
    def test_ranges_do_not_overlap(self):
        shards = plan_shards(8, ["0", "1"], workers_per_gpu=2, seed_start=5)
        self.assertEqual(len(shards), 4)
        episode_ids = []
        seeds = []
        for shard in shards:
            episode_ids.extend(range(shard.episode_start, shard.episode_end))
            seeds.extend(range(shard.seed_start, shard.seed_end))
            self.assertEqual(shard.seed_end - shard.seed_start, shard.episodes * 5)
        self.assertEqual(episode_ids, list(range(8)))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(shards[0].gpu, "0")
        self.assertEqual(shards[2].gpu, "1")

    def test_resume_skips_unfinished_seed(self):
        events = [
            {"type": "start", "episode_id": 0, "seed": 0},
            {"type": "attempt", "episode_id": 0, "seed": 0, "outcome": "saved"},
            {"type": "start", "episode_id": 1, "seed": 1},
            {"type": "attempt", "episode_id": 1, "seed": 1, "outcome": "unstable"},
            {"type": "start", "episode_id": 1, "seed": 2},
        ]
        plan = resume_plan(events, 0, 2, 0, 10)
        self.assertEqual(plan.next_episode, 1)
        self.assertEqual(plan.remaining_episodes, 1)
        self.assertEqual(plan.next_seed, 3)
        self.assertEqual(plan.open_attempt["seed"], 2)

    def test_resume_does_not_reuse_another_workers_seed(self):
        events = []
        for seed in list(range(5)) + list(range(10, 15)):
            events.append({"type": "start", "episode_id": 0, "seed": seed})
            events.append({"type": "attempt", "episode_id": 0, "seed": seed, "outcome": "unstable"})
        plan = resume_plan(events, 0, 1, 0, 15)
        self.assertEqual(plan.next_seed, 15)
        self.assertEqual(plan.remaining_episodes, 1)

    def test_load_events_skips_broken_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"type": "start", "seed": 1}\nnot-json\n')
            events = load_events(path)
        self.assertEqual(len(events), 1)


class AtomicWriteTest(unittest.TestCase):
    def test_replace_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "episode0.hdf5")
            write_text_atomic(dest, "ok")
            self.assertEqual(open(dest, encoding="utf-8").read(), "ok")
            self.assertFalse(os.path.exists(dest + ".tmp"))
            append_jsonl(os.path.join(tmp, "events", "w0.jsonl"), {"type": "attempt", "outcome": "saved"})
            self.assertTrue(os.path.exists(os.path.join(tmp, "events", "w0.jsonl")))


class CoreAssignmentTest(unittest.TestCase):
    def test_cores_are_disjoint(self):
        chunks = assign_cores(4, reserve=8, cpu_count=32)
        flat = [cpu for chunk in chunks for cpu in chunk]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertTrue(all(chunks))
        self.assertTrue(all(cpu < 24 for cpu in flat))


class SupervisorTest(unittest.TestCase):
    def _run(self, tmp: str, **overrides) -> Supervisor:
        supervisor = Supervisor(_config(tmp, **overrides))
        code = supervisor.run()
        supervisor.returncode = code
        return supervisor

    def test_completes_and_merges_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._run(tmp, episodes=2, max_episodes_per_process=1)
            self.assertEqual(supervisor.returncode, 0)
            self.assertEqual(supervisor.slots[0].launches, 2)
            events = load_events(os.path.join(tmp, "events", "w0.jsonl"))
            saved = [event for event in events if event.get("outcome") == "saved"]
            self.assertEqual([event["episode_id"] for event in saved], [0, 1])
            self.assertEqual([event["seed"] for event in saved], [0, 1])
            stats_path = os.path.join(tmp, "collection_stats_w0.json")
            with open(stats_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "task_name": "move_can_pot",
                        "episodes": [
                            {"episode_id": 0, "success": True, "error_injected": True, "final_phase": "recovered"},
                            {"episode_id": 1, "success": False, "error_injected": True, "final_phase": "recovery_failed"},
                        ],
                    },
                    handle,
                )
            supervisor._merge_stats()
            merged = json.load(open(os.path.join(tmp, "collection_stats.json"), encoding="utf-8"))
            self.assertEqual(merged["total_episodes"], 2)
            self.assertEqual(merged["successful_episodes"], 1)
            self.assertEqual(merged["error_recovery_rate"], 0.5)
            self.assertTrue(os.path.exists(os.path.join(tmp, "status.json")))

    def test_crash_loop_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._run(
                tmp,
                episodes=1,
                max_restarts=2,
                worker_prefix=[sys.executable, FAKE_WORKER, "--mode", "crash"],
            )
            self.assertNotEqual(supervisor.returncode, 0)
            self.assertEqual(supervisor.slots[0].state, "failed")
            self.assertEqual(supervisor.slots[0].launches, 3)

    def test_timeout_kills_hung_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._run(
                tmp,
                episodes=1,
                max_restarts=0,
                episode_timeout_s=0.4,
                worker_prefix=[sys.executable, FAKE_WORKER, "--mode", "hang"],
            )
            self.assertNotEqual(supervisor.returncode, 0)
            self.assertEqual(supervisor.slots[0].launches, 1)
            events = load_events(os.path.join(tmp, "events", "w0.jsonl"))
            outcomes = [event.get("outcome") for event in events if event.get("type") == "attempt"]
            self.assertIn("timeout", outcomes)
            self.assertFalse(self._fake_alive())

    def test_rss_limit_kills_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._run(
                tmp,
                episodes=1,
                max_restarts=1,
                episode_timeout_s=30,
                max_rss_gb=1,
                rss_fn=lambda _pid: 10_000_000.0,
                worker_prefix=[sys.executable, FAKE_WORKER, "--mode", "hang"],
            )
            self.assertNotEqual(supervisor.returncode, 0)
            self.assertGreaterEqual(supervisor.slots[0].launches, 2)
            self.assertEqual(supervisor.slots[0].state, "failed")
            self.assertFalse(self._fake_alive())

    def test_low_memory_kills_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._run(
                tmp,
                episodes=1,
                max_restarts=0,
                episode_timeout_s=30,
                min_free_gb=1,
                mem_available_fn=lambda: 0.0,
                worker_prefix=[sys.executable, FAKE_WORKER, "--mode", "hang"],
            )
            self.assertEqual(supervisor.slots[0].state, "failed")
            self.assertEqual(supervisor.slots[0].kill_reason, "mem")
            self.assertFalse(self._fake_alive())

    def test_resume_skips_crashed_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "events"), exist_ok=True)
            path = os.path.join(tmp, "events", "w0.jsonl")
            for event in (
                {"type": "start", "episode_id": 0, "seed": 0, "ts": 1},
                {"type": "attempt", "episode_id": 0, "seed": 0, "outcome": "saved", "ts": 2},
                {"type": "start", "episode_id": 1, "seed": 1, "ts": 3},
            ):
                append_jsonl(path, event)
            supervisor = self._run(tmp, episodes=2)
            self.assertEqual(supervisor.returncode, 0)
            events = load_events(path)
            saved_seeds = [event["seed"] for event in events if event.get("outcome") == "saved"]
            self.assertEqual(saved_seeds, [0, 2])

    def test_live_worker_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "events"), exist_ok=True)
            holder = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)", "collect_universal.py", tmp]
            )
            try:
                append_jsonl(
                    os.path.join(tmp, "events", "w0.jsonl"),
                    {"type": "start", "episode_id": 0, "seed": 0, "pid": holder.pid},
                )
                supervisor = Supervisor(_config(tmp, episodes=1, worker_prefix=[sys.executable, FAKE_WORKER]))
                shard = plan_shards(1, ["0"], 1, 0)[0]
                supervisor.slots = [WorkerSlot(shard=shard, cores=[], active_seed_end=shard.seed_end)]
                supervisor._launch(supervisor.slots[0])
                self.assertEqual(supervisor.slots[0].launches, 0)
                self.assertIsNone(supervisor.slots[0].proc)
                events = load_events(os.path.join(tmp, "events", "w0.jsonl"))
                self.assertFalse(any(event.get("type") == "attempt" for event in events))
            finally:
                holder.kill()
                holder.wait(timeout=5)

    def test_cli_help(self):
        for script, flag in (
            ("error_recovery/collect_universal.py", "--seed_end"),
            ("error_recovery/collect_parallel.py", "--workers_per_gpu"),
        ):
            result = subprocess.run(
                [sys.executable, script, "--help"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(flag, result.stdout)

    @staticmethod
    def _fake_alive() -> bool:
        # pgrep -f matches itself because the pattern is in its own command line.
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or entry == str(os.getpid()):
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as handle:
                    command = handle.read().replace(b"\x00", b" ").decode(errors="ignore")
            except OSError:
                continue
            if "fake_worker.py" in command:
                return True
        return False


if __name__ == "__main__":
    unittest.main()
