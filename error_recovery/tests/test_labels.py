#!/usr/bin/env python3
"""Label tests. Run as a script so the simulator package is not imported."""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "error_recovery" not in sys.modules:
    package = types.ModuleType("error_recovery")
    package.__path__ = [os.path.join(ROOT, "error_recovery")]
    package.__package__ = "error_recovery"
    sys.modules["error_recovery"] = package

from error_recovery.core.data_collector import ConfigManager, DataCollector
from error_recovery.core.execution.flow import check_error_injection
from error_recovery.core.lifecycle.attempt_state import close_open_attempt, is_error_phase, start_error_attempt
from error_recovery.core.recovery.executor import handle_detected_failure
from error_recovery.core.recovery.judge import _backup_grasp_actor_check, finalize_recovery_outcome


class _Injector:
    name = "grasp_slip"

    def plan_recovery(self, context, recovery_attempt=1):
        return [{"action_type": "perceive_actor", "phase": "recovery_perceive"}]

    def get_metadata(self):
        return {}


class _Wrapper:
    def __init__(self):
        self.current_phase = "normal"
        self.error_metadata = []
        self.frame_count = 10
        self.current_recovery_injector_name = None
        self.error_recovery_log_level = "quiet"
        self.error_injectors = []
        self.error_attempt_counter = 0
        self.current_error_attempt_id = None
        self.wrapped_task = None
        self.last_recovery_error = None


def _failed_payload():
    # current_phase stays "recovered" on purpose: that was the old overwrite.
    # The summary must follow end_status, not this stale phase.
    return {
        "current_phase": "recovered",
        "error_stage_precise": "normal",
        "is_error_frame_precise": False,
        "frame_count": 180,
        "active_injectors": [],
        "action_history": [
            {"frame_count": 10, "method_name": "move"},
            {"frame_count": 160, "method_name": "move"},
        ],
        "error_metadata": [{
            "injector": "grasp_orientation_mismatch",
            "end_status": "recovery_failed",
            "inject_decision_frame": 40,
            "error_onset_frame": 44,
            "error_detected_frame": 64,
            "recovery_start_frame": 64,
            "recovery_end_frame": 154,
            "error_end_frame": 154,
        }],
    }


class LabelTests(unittest.TestCase):
    def test_error_phase_includes_recovery_subphases(self):
        self.assertTrue(is_error_phase("recovery"))
        self.assertTrue(is_error_phase("recovery_perceive"))
        self.assertTrue(is_error_phase("recovery_reposition"))
        self.assertTrue(is_error_phase("recovery_failed"))
        self.assertTrue(is_error_phase("error_detected"))
        self.assertFalse(is_error_phase("normal"))
        self.assertFalse(is_error_phase("recovered"))

    def test_failed_recovery_is_not_relabeled_recovered(self):
        wrapper = _Wrapper()
        injector = _Injector()
        start_error_attempt(wrapper, injector)
        handle_detected_failure(wrapper, injector, {}, logging.getLogger("label-test"))
        self.assertEqual(wrapper.current_phase, "normal")
        self.assertEqual(wrapper.error_metadata[-1]["end_status"], "recovery_failed")
        self.assertEqual(wrapper.error_metadata[-1]["phase"], "recovery_failed")

    def test_open_attempt_closes_at_episode_end(self):
        wrapper = _Wrapper()
        start_error_attempt(wrapper, _Injector())
        close_open_attempt(wrapper)
        self.assertEqual(wrapper.current_phase, "normal")
        self.assertEqual(wrapper.error_metadata[-1]["end_status"], "no_failure")
        self.assertEqual(wrapper.error_metadata[-1]["phase"], "no_failure")

        detected = _Wrapper()
        start_error_attempt(detected, _Injector())
        detected.error_metadata[-1]["error_detected_frame"] = 5
        close_open_attempt(detected)
        self.assertEqual(detected.current_phase, "normal")
        self.assertEqual(detected.error_metadata[-1]["end_status"], "recovery_failed")

    def test_zero_detected_frame_is_not_a_trim_point(self):
        raw = {
            "current_phase": "normal",
            "frame_count": 10,
            "error_metadata": [{"error_detected_frame": 0}],
        }
        info = DataCollector.compute_failure_trim_info(raw, success=False)
        self.assertIsNone(info["trim_detected_frame"])

    def test_successful_recovery_returns_live_phase_to_normal(self):
        wrapper = _Wrapper()
        start_error_attempt(wrapper, _Injector())
        wrapper.current_phase = "recovery"
        finalize_recovery_outcome(wrapper, [], True, logging.getLogger("label-test"))
        self.assertEqual(wrapper.current_phase, "normal")
        self.assertEqual(wrapper.error_metadata[-1]["end_status"], "recovered")
        self.assertEqual(wrapper.error_metadata[-1]["phase"], "recovered")

    def test_summary_uses_end_status_and_drops_unsaved_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = DataCollector(tmp, "move_can_pot")
            raw = _failed_payload()
            trim = DataCollector.compute_failure_trim_info(raw, success=False)
            prepared = DataCollector._prepare_metadata_error_recovery_data(raw, trim, success=False)
            summary = collector._build_episode_summary(0, prepared, trim, success=False)
            attempt = prepared["error_metadata"][0]
            self.assertEqual(summary["final_phase"], "recovery_failed")
            self.assertTrue(summary["error_recovery_attempted"])
            self.assertEqual(attempt["end_status"], "recovery_failed")
            self.assertEqual(attempt["error_onset_frame"], 44)
            self.assertEqual(attempt["recovery_start_frame"], 64)
            self.assertIsNone(attempt["recovery_end_frame"])
            self.assertIsNone(attempt["error_end_frame"])
            self.assertEqual([item["frame_count"] for item in prepared["action_history"]], [10])

            success_raw = {
                "current_phase": "normal",
                "frame_count": 200,
                "error_metadata": [{"injector": "premature_close", "end_status": "recovered"}],
            }
            success = collector._build_episode_summary(
                1, success_raw, DataCollector.compute_failure_trim_info(success_raw, success=True), success=True
            )
            self.assertEqual(success["final_phase"], "recovered")
            collector._update_stats(summary)
            collector._update_stats(success)
            collector.save_collection_stats()
            self.assertEqual(collector.collection_stats["error_recovery_episodes"], 1)
            self.assertEqual(collector.collection_stats["error_recovery_rate"], 0.5)

    def test_wrap_get_obs_writes_phase_labels(self):
        from error_recovery.core.framework_support import wrap_get_obs

        class _ObsTask:
            def get_obs(self):
                return {"observation": True}

        class _ObsWrapper:
            def __init__(self):
                self.frame_count = 0
                self.payload = {
                    "current_phase": "recovery_reposition",
                    "error_stage_precise": "recovery",
                    "is_error_frame_precise": True,
                    "frame_count": 0,
                    "active_injectors": [],
                    "error_metadata": [],
                }

            def update_frame_count(self):
                self.frame_count += 1
                self.payload["frame_count"] = self.frame_count

            def get_error_recovery_data(self):
                return self.payload

            def sanitize_error_metadata_item(self, item):
                return item

        task = _ObsTask()
        wrapper = _ObsWrapper()
        wrap_get_obs(task, wrapper)
        first = task.get_obs()["error_recovery"]
        self.assertEqual(first["phase"], "recovery_reposition")
        self.assertEqual(first["error_stage_precise"], "recovery")
        self.assertTrue(first["is_error_frame"])
        self.assertTrue(first["is_error_frame_precise"])
        self.assertEqual(first["frame_count"], 1)

        wrapper.payload["current_phase"] = "normal"
        wrapper.payload["error_stage_precise"] = "normal"
        wrapper.payload["is_error_frame_precise"] = False
        second = task.get_obs()["error_recovery"]
        self.assertEqual(second["phase"], "normal")
        self.assertEqual(second["error_stage_precise"], "normal")
        self.assertFalse(second["is_error_frame"])
        self.assertEqual(second["frame_count"], 2)

    def test_stats_are_current_before_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = DataCollector(tmp, "move_can_pot", worker_id="w0")
            collector._update_stats({"episode_id": 0, "success": True, "error_injected": True, "final_phase": "recovered"})
            collector._update_stats({"episode_id": 1, "success": False, "error_injected": True, "final_phase": "recovery_failed"})
            live = collector.get_stats()
            self.assertEqual(live["total_episodes"], 2)
            self.assertEqual(live["successful_episodes"], 1)
            self.assertEqual(live["error_recovery_episodes"], 1)
            collector.save_collection_stats()
            with open(os.path.join(tmp, "collection_stats_w0.json"), encoding="utf-8") as handle:
                saved = json.load(handle)
            for key in (
                "total_episodes",
                "successful_episodes",
                "error_injected_episodes",
                "error_recovery_episodes",
                "success_rate",
                "error_recovery_rate",
            ):
                self.assertEqual(saved[key], collector.get_stats()[key])

    def test_restarted_worker_keeps_earlier_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = DataCollector(tmp, "move_can_pot", worker_id="w0")
            first._update_stats({"episode_id": 0, "success": True, "error_injected": True, "final_phase": "recovered"})
            first._update_stats({"episode_id": 1, "success": False, "error_injected": True, "final_phase": "recovery_failed"})
            first.save_collection_stats()

            second = DataCollector(tmp, "move_can_pot", worker_id="w0")
            second._update_stats({"episode_id": 2, "success": True, "error_injected": True, "final_phase": "recovered"})
            second.save_collection_stats()
            with open(os.path.join(tmp, "collection_stats_w0.json"), encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual([episode["episode_id"] for episode in saved["episodes"]], [0, 1, 2])
            self.assertEqual(saved["total_episodes"], 3)
            self.assertEqual(saved["successful_episodes"], 2)
            self.assertEqual(saved["error_recovery_episodes"], 2)
            self.assertAlmostEqual(saved["error_recovery_rate"], 2 / 3)
            self.assertEqual(saved["start_time"], first.collection_stats["start_time"])

            third = DataCollector(tmp, "move_can_pot", worker_id="w0")
            third._update_stats({"episode_id": 0, "success": False, "error_injected": True, "final_phase": "recovery_failed"})
            third.save_collection_stats()
            with open(os.path.join(tmp, "collection_stats_w0.json"), encoding="utf-8") as handle:
                replaced = json.load(handle)
            self.assertEqual([episode["episode_id"] for episode in replaced["episodes"]], [0, 1, 2])
            self.assertFalse(replaced["episodes"][0]["success"])
            self.assertEqual(replaced["error_recovery_episodes"], 1)

    def test_serial_rerun_drops_episodes_it_did_not_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = DataCollector(tmp, "move_can_pot")
            first._update_stats({"episode_id": 0, "success": True, "error_injected": True, "final_phase": "recovered"})
            first._update_stats({"episode_id": 1, "success": False, "error_injected": True, "final_phase": "recovery_failed"})
            first.save_collection_stats()

            second = DataCollector(tmp, "move_can_pot")
            second._update_stats({"episode_id": 0, "success": True, "error_injected": True, "final_phase": "recovered"})
            second.save_collection_stats()
            with open(os.path.join(tmp, "collection_stats.json"), encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual([episode["episode_id"] for episode in saved["episodes"]], [0])
            self.assertEqual(saved["total_episodes"], 1)
            self.assertEqual(saved["error_recovery_episodes"], 1)

    def test_saved_end_frame_becomes_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = tmp + os.sep
            wrapper = _Wrapper()
            start_error_attempt(wrapper, _Injector())
            attempt = wrapper.error_metadata[-1]
            attempt["inject_decision_frame"] = 1
            attempt["error_onset_frame"] = 2
            attempt["error_detected_frame"] = 2
            attempt["recovery_start_frame"] = 3
            wrapper.frame_count = 4
            wrapper.current_phase = "recovery_reposition"
            wrapper.wrapped_task = types.SimpleNamespace(folder_path={"cache": cache}, FRAME_IDX=2)
            with open(cache + "1.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "error_recovery": {
                            "phase": "recovery_reposition",
                            "frame_count": 4,
                            "is_error_frame": True,
                            "error_metadata": {"phase": "recovery"},
                        }
                    },
                    handle,
                )
            finalize_recovery_outcome(wrapper, [], True, logging.getLogger("label-test"))
            with open(cache + "1.pkl", "rb") as handle:
                saved = pickle.load(handle)
            recovery = saved["error_recovery"]
            self.assertEqual(recovery["phase"], "normal")
            self.assertFalse(recovery["is_error_frame"])
            self.assertEqual(recovery["error_stage_precise"], "recovered")
            self.assertEqual(recovery["error_metadata"]["phase"], "recovered")
            self.assertEqual(recovery["error_metadata"]["end_status"], "recovered")

    def test_backup_grasp_does_not_mark_recovered(self):
        class _Pose:
            p = [0.0, 0.0, 0.0]

        class _Actor:
            def get_pose(self):
                return _Pose()

        class _Robot:
            def get_left_gripper_val(self):
                return 0.0

        class _Task:
            robot = _Robot()

            def get_arm_pose(self, arm_tag):
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

            def get_dynamic_distance_threshold(self, actor, base_threshold):
                return 0.2

            def get_dynamic_gripper_threshold(self, actor, base_threshold):
                return 0.3

        wrapper = _Wrapper()
        wrapper.current_phase = "recovery"
        accepted = _backup_grasp_actor_check(
            wrapper,
            [],
            0,
            _Actor(),
            "left",
            _Task(),
            False,
            logging.getLogger("label-test"),
        )
        self.assertFalse(accepted)
        self.assertEqual(wrapper.current_phase, "recovery")

    def test_stats_save_with_no_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for worker_id in (None, "w0"):
                collector = DataCollector(tmp, "move_can_pot", worker_id=worker_id)
                collector.save_collection_stats()
                name = f"collection_stats_{worker_id}.json" if worker_id else "collection_stats.json"
                with open(os.path.join(tmp, name), encoding="utf-8") as handle:
                    saved = json.load(handle)
                self.assertEqual(saved["total_episodes"], 0)
                self.assertEqual(saved["success_rate"], 0.0)

    def test_scene_selects_domain_randomization(self):
        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            self.assertIsNone(ConfigManager.scene_randomization("clean"))
            random_scene = ConfigManager.scene_randomization("random")
        finally:
            os.chdir(previous)
        self.assertTrue(random_scene["cluttered_table"])
        self.assertTrue(random_scene["random_background"])
        self.assertTrue(random_scene["random_light"])
        self.assertFalse(random_scene["random_embodiment"])
        with self.assertRaises(ValueError):
            ConfigManager.scene_randomization("noisy")

    def test_scene_is_not_passed_to_task_kwargs(self):
        from error_recovery.core.framework_support import load_and_setup_config

        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            config = load_and_setup_config("lift_pot", "/tmp/unused", False, scene="random")
        finally:
            os.chdir(previous)
        self.assertNotIn("scene", config)
        self.assertEqual(config["scene_profile"], "random")
        self.assertTrue(config["domain_randomization"]["cluttered_table"])

    def test_only_one_injection_even_during_unlisted_recovery_phase(self):
        class _Ready:
            has_injected_this_episode = False

            def should_inject(self, context):
                return True

        wrapper = _Wrapper()
        wrapper.current_phase = "recovery_perceive"
        wrapper.error_injection_count = 1
        wrapper.max_errors_per_episode = 4
        wrapper.error_injectors = [_Ready()]
        self.assertIsNone(check_error_injection(wrapper, {}))


if __name__ == "__main__":
    unittest.main()
