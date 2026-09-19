"""CPU-only world-pose window contract tests; no model or video dependencies."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/umi_pretrain/tools"))

import umi_valid_windows as windows
from umi_valid_windows import (
    DATA_VERSION, FRAME_VALIDITY, HORIZON, LOGICAL_SIGNALS, LOGICAL_VIDEOS,
    SIGNAL_COLUMNS, SIGNAL_WIDTHS, VIDEO_KEYS, dataset_indices, load_private_umi,
    select_valid_windows, validate_mapping,
)


def mapping():
    signals = {
        key: {"original_key": source, "start": 0, "end": width,
              "absolute": True, "dtype": "float32"}
        for key, source, width in zip(LOGICAL_SIGNALS, SIGNAL_COLUMNS, SIGNAL_WIDTHS)
    }
    return {
        "state": copy.deepcopy(signals), "action": copy.deepcopy(signals),
        "video": {key: {"original_key": source}
                  for key, source in zip(LOGICAL_VIDEOS, VIDEO_KEYS)},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def frames(n=40):
    data = {
        "episode_index": [3] * n, "frame_index": np.arange(n),
        "timestamp": np.arange(n, dtype=np.float32) / 30,
        "task_index": [7] * n, FRAME_VALIDITY: [True] * n,
    }
    for signal, width in zip(SIGNAL_COLUMNS, SIGNAL_WIDTHS):
        data[signal] = (
            [np.array([i / 10, .12345679, .3, 0, 0, 0, 1], dtype=np.float32) for i in range(n)]
            if width == 7 else [np.float32(.12345679 + i / 1000) for i in range(n)]
        )
    return pd.DataFrame(data)


class UMIValidWindowTests(unittest.TestCase):
    def scan(self, raw):
        return select_valid_windows(raw, {7: "pick object", 8: "place object"}, 30)

    def test_tail_is_excluded_and_values_keep_float32_precision(self):
        raw = frames()
        steps, report, values = self.scan(raw)
        self.assertEqual(DATA_VERSION, "world-pose-16D")
        self.assertEqual(steps, [(3, i) for i in range(40 - HORIZON)])
        self.assertEqual(report["window_rejections"], {"incomplete_future": 16})
        self.assertEqual(values.shape, (40, 16))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values[0, 1], raw.iloc[0][SIGNAL_COLUMNS[0]][1])
        self.assertNotEqual(values[0, 1], np.float32(np.float16(values[0, 1])))
        self.assertEqual(report["valid_starts"][0]["future_frame_indices"], list(range(1, 17)))
        json.dumps(report, allow_nan=False)

    def test_invalid_middle_removes_every_overlapping_window(self):
        raw = frames()
        raw.at[20, FRAME_VALIDITY] = False
        steps, report, values = self.scan(raw)
        self.assertEqual(steps, [(3, i) for i in [0, 1, 2, 3, 21, 22, 23]])
        self.assertEqual(report["frame_rejections"]["missing_or_false_frame_validity"], 1)
        self.assertEqual(len(values), len(raw))
        self.assertEqual(raw.iloc[20]["frame_index"], 20)

    def test_timestamp_gap_rejects_crossing_windows(self):
        raw = frames()
        raw.loc[20:, "timestamp"] += .5
        steps, report, _ = self.scan(raw)
        self.assertEqual(steps, [(3, i) for i in [0, 1, 2, 3, 20, 21, 22, 23]])
        self.assertEqual(report["window_rejections"]["timestamp_gap"], 16)

    def test_frame_gap_keeps_original_row_offsets(self):
        raw = frames()
        raw.loc[10:, "frame_index"] += 20
        raw["timestamp"] = raw["frame_index"].astype(np.float32) / 30
        steps, report, _ = self.scan(raw)
        self.assertEqual(steps[0], (3, 10))
        first = report["valid_starts"][0]
        self.assertEqual(first["row_index"], 10)
        self.assertEqual(first["episode_row_index"], 10)
        self.assertEqual(first["frame_index"], 30)
        dataset = SimpleNamespace(all_steps=[(3, i) for i in reversed(range(40))])
        self.assertEqual(dataset_indices(dataset, steps)[0], 29)
        self.assertEqual(raw.iloc[10]["frame_index"], 30)

    def test_root_source_timestamp_gap_detects_hidden_grid_discontinuity(self):
        raw = frames()
        raw["source_timestamp_ns"] = np.int64(1700000000000000000) + np.arange(len(raw), dtype=np.int64) * 33333333
        self.assertEqual(len(self.scan(raw)[0]), 24)
        raw.loc[20:, "source_timestamp_ns"] += 500000000
        steps, report, _ = self.scan(raw)
        self.assertEqual(steps, [(3, i) for i in [0, 1, 2, 3, 20, 21, 22, 23]])
        self.assertEqual(report["window_rejections"]["source_timestamp_gap"], 16)
        self.assertNotIn("timestamp_gap", report["window_rejections"])
        self.assertTrue(report["policy"]["root_source_timestamp_interval_checked"])

    def test_root_source_timestamp_must_increase(self):
        raw = frames()
        raw["source_timestamp_ns"] = np.int64(1700000000000000000) + np.arange(len(raw), dtype=np.int64) * 33333333
        raw.at[20, "source_timestamp_ns"] = raw.at[19, "source_timestamp_ns"] - 1
        steps, report, _ = self.scan(raw)
        self.assertNotIn((3, 4), steps)
        self.assertNotIn((3, 20), steps)
        self.assertIn("source_timestamp_gap", report["window_rejections"])

    def test_load_requires_explicit_pose_order_and_full_window(self):
        with TemporaryDirectory() as tmp:
            private = Path(tmp) / "datasets/sample"
            (private / "meta").mkdir(parents=True)
            info_path = private / "meta/info.json"
            info = {"total_episodes": 1, "total_frames": 17}
            with patch.object(windows, "REPO", Path(tmp) / "repo"):
                for order in (None, ["x", "y", "z", "qw", "qx", "qy", "qz"]):
                    info["numeric_conversion"] = {"pose_order": order}
                    info_path.write_text(json.dumps(info))
                    with self.assertRaisesRegex(ValueError, "pose_order"):
                        load_private_umi(private)
                info.update(total_frames=16, numeric_conversion={"pose_order": ["x", "y", "z", "qx", "qy", "qz", "qw"]})
                info_path.write_text(json.dumps(info))
                with self.assertRaisesRegex(ValueError, "17 to"):
                    load_private_umi(private)

    def test_task_change_rejects_crossing_windows(self):
        raw = frames()
        raw.loc[20:, "task_index"] = 8
        steps, report, _ = self.scan(raw)
        self.assertEqual(steps, [(3, i) for i in [0, 1, 2, 3, 20, 21, 22, 23]])
        self.assertEqual(report["window_rejections"]["task_change"], 16)

    def test_missing_or_nonboolean_frame_flag_is_never_assumed_true(self):
        for mutate in (lambda raw: raw.drop(columns=FRAME_VALIDITY),
                       lambda raw: raw.assign(**{FRAME_VALIDITY: [1] * len(raw)})):
            with self.subTest(mutate=mutate):
                steps, report, _ = self.scan(mutate(frames()))
                self.assertEqual(steps, [])
                self.assertEqual(report["frame_rejections"]["missing_or_false_frame_validity"], 40)
        raw = frames()
        raw[FRAME_VALIDITY] = [np.array([True])] * len(raw)
        self.assertEqual(len(self.scan(raw)[0]), 24)

    def test_empty_or_missing_task_text_invalidates_frames(self):
        for tasks in ({}, {7: "  "}):
            steps, report, _ = select_valid_windows(frames(), tasks, 30)
            self.assertEqual(steps, [])
            self.assertEqual(report["frame_rejections"]["missing_or_empty_task"], 40)

    def test_optional_schema_presence_false_rejects_required_signal(self):
        raw = frames()
        self.assertEqual(len(self.scan(raw)[0]), 24)
        flag = f"schema_presence.{SIGNAL_COLUMNS[0]}"
        raw[flag] = [True] * len(raw)
        raw.at[20, flag] = False
        steps, report, _ = self.scan(raw)
        self.assertNotIn((3, 20), steps)
        self.assertEqual(report["frame_rejections"][f"schema_absent:{SIGNAL_COLUMNS[0]}"], 1)

    def test_malformed_or_missing_pose_and_nonfinite_gripper(self):
        raw = frames()
        raw.at[20, SIGNAL_COLUMNS[0]] = [0] * 6
        raw.at[21, SIGNAL_COLUMNS[1]] = np.inf
        steps, report, _ = self.scan(raw)
        self.assertNotIn((3, 20), steps)
        self.assertEqual(report["frame_rejections"][f"invalid_signal:{SIGNAL_COLUMNS[0]}"], 1)
        self.assertEqual(report["frame_rejections"][f"invalid_signal:{SIGNAL_COLUMNS[1]}"], 1)

    def test_zero_quaternion_rejected_signs_and_grippers_unchanged(self):
        raw = frames()
        raw.at[20, SIGNAL_COLUMNS[0]] = [0, 0, 0, 0, 0, 0, 0]
        steps, report, _ = self.scan(raw)
        self.assertNotIn((3, 20), steps)
        self.assertEqual(report["frame_rejections"]["unavailable_quaternion:robot1"], 1)
        raw = frames()
        raw.at[1, SIGNAL_COLUMNS[0]] = [0, 0, 0, 0, 0, 0, -1]
        raw[SIGNAL_COLUMNS[1]] = np.float32(12345)
        steps, report, values = self.scan(raw)
        self.assertEqual(len(steps), 24)
        self.assertEqual(values[1, 6], -1)
        self.assertEqual(values[1, 7], 12345)
        self.assertEqual(report["quaternions"]["robot1"]["adjacent_rotation_degrees_sign_invariant"]["max"], 0)

    def test_available_source_timestamp_sentinels_reject_without_gap_cutoff(self):
        raw = frames()
        key = SIGNAL_COLUMNS[0] + ".source_timestamp_ns"
        raw[key] = np.int64(1700000000000000000)
        raw.at[20, key] = 0
        raw[SIGNAL_COLUMNS[0] + ".gap_ns"] = np.int64(999999999999)
        steps, report, _ = self.scan(raw)
        self.assertNotIn((3, 20), steps)
        self.assertEqual(report["frame_rejections"][f"unavailable_source_timestamp:{key}"], 1)
        self.assertIsNone(report["policy"]["source_gap_ns_limit"])
        self.assertEqual(len(self.scan(raw.drop(columns=key))[0]), 24)

    def test_episode_boundary_and_offsets(self):
        raw = pd.concat([frames(20), frames(20)], ignore_index=True)
        raw.loc[20:, "episode_index"] = 9
        steps, report, _ = self.scan(raw)
        self.assertEqual(steps, [(ep, row) for ep in (3, 9) for row in range(4)])
        self.assertEqual(report["valid_starts"][4]["row_index"], 20)
        self.assertEqual(report["valid_starts"][4]["episode_row_index"], 0)

    def test_mapping_rejects_relative_sources_dtype_slices_and_wrong_order(self):
        validate_mapping(mapping())
        mutations = [
            lambda m: m["state"]["robot1_pose"].update(original_key="observation.umi.robot1_finger_relative_eef_pose"),
            lambda m: m["action"]["robot1_pose"].update(dtype="float16"),
            lambda m: m["state"]["robot1_pose"].update(start=1),
            lambda m: m["action"]["robot2_pose"].update(absolute=False),
            lambda m: m["state"].update({"unexpected": m["state"]["robot1_pose"]}),
            lambda m: m.update(video=dict(reversed(list(m["video"].items())))),
            lambda m: m["annotation"]["human.action.task_description"].update(original_key="wrong"),
        ]
        for mutate in mutations:
            candidate = mapping()
            mutate(candidate)
            with self.subTest(mapping=candidate), self.assertRaises(ValueError):
                validate_mapping(candidate)

    def test_dataset_mapping_is_explicit_and_missing_or_duplicate_steps_fail(self):
        dataset = SimpleNamespace(all_steps=[(3, 2), (3, 0), (3, 1)])
        self.assertEqual(dataset_indices(dataset, [(3, 0), (3, 2)]), [1, 0])
        with self.assertRaises(ValueError):
            dataset_indices(dataset, [(3, 4)])
        with self.assertRaises(ValueError):
            dataset_indices(dataset, [(3, 0), (3, 0)])
        with self.assertRaises(ValueError):
            dataset_indices(SimpleNamespace(all_steps=[(3, 0), (3, 0)]), [(3, 0)])

    def test_private_guard_and_small_data_guard(self):
        with self.assertRaises(ValueError):
            load_private_umi(Path("/tmp/not-a-private-umi-dataset"))
        with self.assertRaises(ValueError):
            self.scan(frames(10001))
        with self.assertRaises(ValueError):
            select_valid_windows(frames(), {7: "pick"}, 30, horizon=8)


if __name__ == "__main__":
    unittest.main()
