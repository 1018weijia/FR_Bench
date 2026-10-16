#!/usr/bin/env python3
"""Compact dataset summary tool for error_recovery outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


class ErrorRecoveryDataAnalyzer:
    """Load collection_stats.json and episode metadata for quick summaries."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.stats = self._load_json(self.data_path / "collection_stats.json") or {}
        self.episodes_data = self._load_episode_metadata()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_episode_metadata(self) -> list[dict[str, Any]]:
        metadata_dir = self.data_path / "metadata"
        if not metadata_dir.exists():
            return []
        episodes = []
        for path in sorted(metadata_dir.glob("episode*_metadata.json")):
            item = self._load_json(path)
            if item is not None:
                episodes.append(item)
        return episodes

    @property
    def summaries(self) -> list[dict[str, Any]]:
        if self.episodes_data:
            return [item.get("episode_summary", {}) for item in self.episodes_data]
        return list(self.stats.get("episodes", []))

    def generate_report(self) -> str:
        summaries = self.summaries
        total = int(self.stats.get("total_episodes", len(summaries)))
        success = sum(1 for item in summaries if item.get("success"))
        injected = sum(1 for item in summaries if item.get("error_injected"))
        recovery_attempted = sum(1 for item in summaries if item.get("error_recovery_attempted"))
        error_types = Counter(error_type for item in summaries for error_type in item.get("error_types", []))
        phases = Counter(str(item.get("final_phase", "unknown")) for item in summaries)
        failed = self.get_failed_episodes()
        recovery_failed = self.get_recovery_failed_episodes()

        lines = [
            f"data_path: {self.data_path}",
            f"task: {self.stats.get('task_name', 'unknown')}",
            f"episodes: {total}",
            f"success: {success}/{max(total, 1)} ({success / max(total, 1) * 100:.1f}%)",
            f"injected: {injected}",
            f"recovery_attempted: {recovery_attempted}",
        ]
        if error_types:
            lines.append("error_types:")
            lines.extend(f"  {name}: {count}" for name, count in error_types.most_common())
        if phases:
            lines.append("final_phases:")
            lines.extend(f"  {name}: {count}" for name, count in phases.most_common())
        if failed:
            lines.append(f"failed_episodes: {failed}")
        if recovery_failed:
            lines.append(f"recovery_failed_episodes: {recovery_failed}")
        return "\n".join(lines)

    def export_to_csv(self, save_path: str) -> None:
        rows = []
        for item in self.summaries:
            rows.append({
                "episode_id": item.get("episode_id"),
                "success": item.get("success"),
                "error_injected": item.get("error_injected"),
                "error_recovery_attempted": item.get("error_recovery_attempted"),
                "final_phase": item.get("final_phase"),
                "total_frames": item.get("total_frames"),
                "trim_detected_frame": item.get("trim_detected_frame"),
                "trimmed_total_frames": item.get("trimmed_total_frames"),
                "error_types": ",".join(item.get("error_types", [])),
                "attempt_count": item.get("attempt_count"),
            })
        if not rows:
            print("no episodes to export")
            return
        with open(save_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv: {save_path}")

    def get_failed_episodes(self) -> list[int]:
        return [item.get("episode_id") for item in self.summaries if not item.get("success", False)]

    def get_recovery_failed_episodes(self) -> list[int]:
        return [item.get("episode_id") for item in self.summaries if item.get("final_phase") == "recovery_failed"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an error_recovery dataset directory")
    parser.add_argument("data_path", help="Dataset directory, e.g. error_recovery/data/pick_diverse_bottles")
    parser.add_argument("--export", help="Optional CSV output path")
    args = parser.parse_args()

    analyzer = ErrorRecoveryDataAnalyzer(args.data_path)
    print(analyzer.generate_report())
    if args.export:
        analyzer.export_to_csv(args.export)


if __name__ == "__main__":
    main()
