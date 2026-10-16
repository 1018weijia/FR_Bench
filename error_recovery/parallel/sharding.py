"""Episode and seed assignment for parallel collection workers.

Pure functions: no simulator imports, so the supervisor and unit tests can
load this module without initializing CUDA.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


ATTEMPTS_PER_EPISODE = 5
SAVED_OUTCOME = "saved"


@dataclass(frozen=True)
class Shard:
    worker_id: str
    gpu: str
    episode_start: int
    episode_end: int
    seed_start: int
    seed_end: int

    @property
    def episodes(self) -> int:
        return self.episode_end - self.episode_start


@dataclass(frozen=True)
class ResumePlan:
    next_episode: int
    next_seed: int
    remaining_episodes: int
    open_attempt: Optional[Dict[str, Any]]


def plan_shards(
    episodes: int,
    gpus: Iterable[str],
    workers_per_gpu: int,
    seed_start: int = 0,
    attempts_per_episode: int = ATTEMPTS_PER_EPISODE,
) -> List[Shard]:
    """Split episodes into disjoint id ranges and disjoint seed blocks."""
    gpu_list = [str(gpu).strip() for gpu in gpus if str(gpu).strip() != ""]
    if episodes < 0:
        raise ValueError(f"episodes must be non-negative, got {episodes}")
    if workers_per_gpu < 1:
        raise ValueError(f"workers_per_gpu must be >= 1, got {workers_per_gpu}")
    if seed_start < 0:
        raise ValueError(f"seed_start must be non-negative, got {seed_start}")
    if attempts_per_episode < 1:
        raise ValueError(f"attempts_per_episode must be >= 1, got {attempts_per_episode}")
    if episodes == 0 or not gpu_list:
        if episodes > 0 and not gpu_list:
            raise ValueError("at least one GPU is required")
        return []

    worker_count = len(gpu_list) * workers_per_gpu
    base, remainder = divmod(episodes, worker_count)
    shards: List[Shard] = []
    episode_cursor = 0
    seed_cursor = seed_start
    worker_index = 0
    for gpu in gpu_list:
        for _ in range(workers_per_gpu):
            count = base + (1 if worker_index < remainder else 0)
            worker_index += 1
            if count <= 0:
                continue
            seed_span = count * attempts_per_episode
            shards.append(
                Shard(
                    worker_id=f"w{len(shards)}",
                    gpu=gpu,
                    episode_start=episode_cursor,
                    episode_end=episode_cursor + count,
                    seed_start=seed_cursor,
                    seed_end=seed_cursor + seed_span,
                )
            )
            episode_cursor += count
            seed_cursor += seed_span
    return shards


def load_events(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def resume_plan(
    events: List[Dict[str, Any]],
    episode_start: int,
    episode_end: int,
    seed_start: int,
    seed_end: int,
) -> ResumePlan:
    """Continue a shard from its event log.

    Saved attempts advance the episode id. Any seed that was started is
    consumed, including a start line with no matching attempt (the process
    died on that seed and must not retry it).
    """
    saved_ids = []
    consumed = set()
    open_attempt = None
    for event in events:
        kind = event.get("type")
        seed = event.get("seed")
        if kind == "start":
            open_attempt = event
            if seed is not None:
                consumed.add(int(seed))
            continue
        if kind != "attempt":
            continue
        if seed is not None:
            consumed.add(int(seed))
        open_attempt = None
        if event.get("outcome") != SAVED_OUTCOME:
            continue
        episode_id = event.get("episode_id")
        if episode_id is None:
            continue
        episode_id = int(episode_id)
        if episode_start <= episode_id < episode_end and episode_id not in saved_ids:
            saved_ids.append(episode_id)

    next_episode = episode_start + len(saved_ids)
    if next_episode > episode_end:
        next_episode = episode_end
    # Continue after the highest seed this worker already used. Scanning for a
    # hole below seed_end would pick up another worker's seed block.
    next_seed = max(consumed) + 1 if consumed else seed_start
    remaining = max(0, episode_end - next_episode)
    if next_seed >= seed_end:
        next_seed = seed_end
    return ResumePlan(
        next_episode=next_episode,
        next_seed=next_seed,
        remaining_episodes=remaining,
        open_attempt=open_attempt,
    )
