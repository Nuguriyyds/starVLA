"""Focused v3 metadata checks; no dataset initialization or video decoding."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class LeRobotV3MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(
            prefix="metadata-test-", dir=Path(__file__).resolve().parent
        )
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "meta").mkdir()
        self.dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        self.dataset._dataset_path = self.root
        self.dataset._lerobot_version = "v3.0"
        self.dataset._chunk_size = 1000
        self.dataset._video_path_pattern = (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        self.dataset._lerobot_modality_meta = SimpleNamespace(video={
            "head": SimpleNamespace(original_key="observation.images.head_left"),
            "wrist": SimpleNamespace(original_key="observation.images.wrist_left"),
        })

    def read_tasks(self, frame):
        frame.to_parquet(self.root / "meta/tasks.parquet")
        return self.dataset._get_tasks()

    def assert_task_lookup(self, frame):
        tasks = self.read_tasks(frame)
        self.assertEqual(tasks.index.name, "task_index")
        self.assertTrue(tasks.columns.is_unique)
        self.assertEqual(tasks.loc[[42, 17], "task"].tolist(), ["close", "open"])

    def test_physical_task_column_with_sparse_ids(self):
        self.assert_task_lookup(pd.DataFrame({
            "task_index": [17, 42], "task": ["open", "close"]
        }))

    def test_text_in_named_or_unnamed_index(self):
        for name in (None, "task"):
            with self.subTest(index_name=name):
                self.assert_task_lookup(pd.DataFrame(
                    {"task_index": [17, 42]},
                    index=pd.Index(["open", "close"], name=name),
                ))

    def test_task_id_in_index(self):
        self.assert_task_lookup(pd.DataFrame(
            {"task": ["open", "close"]},
            index=pd.Index([17, 42], name="task_index"),
        ))

    def test_duplicate_task_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.read_tasks(pd.DataFrame({
                "task_index": [17, 17], "task": ["open", "close"]
            }))

    def test_missing_task_information_rejected(self):
        malformed = [pd.DataFrame({"task_index": [17, 42]}),
                     pd.DataFrame({"task": ["open", "close"]})]
        for frame in malformed:
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaises(ValueError):
                    self.read_tasks(frame)

    def set_signal(self, values, start, end):
        self.dataset.curr_traj_data = pd.DataFrame({"signal": values})
        self.dataset._trajectory_ids = np.array([5])
        self.dataset._trajectory_lengths = np.array([len(values)])
        spec = SimpleNamespace(original_key="signal", start=start, end=end, absolute=True)
        self.dataset._lerobot_modality_meta.state = {"signal": spec}
        self.dataset._lerobot_modality_meta.action = {"signal": spec}
        self.dataset._metadata = SimpleNamespace(modalities=SimpleNamespace(
            state={"signal": spec}, action={"signal": spec}
        ))
        self.dataset._delta_indices = {
            "state.signal": np.array([0]), "action.signal": np.array([1, 2, 3])
        }

    def test_scalar_signal_current_future_and_tail_padding(self):
        self.set_signal([10.0, 20.0, 30.0, 40.0], 0, 1)
        state = self.dataset.get_state_or_action(5, "state", "state.signal", 1)
        future = self.dataset.get_state_or_action(5, "action", "action.signal", 0)
        tail = self.dataset.get_state_or_action(5, "action", "action.signal", 2)
        np.testing.assert_array_equal(state, [[20.0]])
        np.testing.assert_array_equal(future, [[20.0], [30.0], [40.0]])
        np.testing.assert_array_equal(tail, [[40.0], [40.0], [40.0]])

    def test_vector_signal_column_slicing_preserved(self):
        self.set_signal([
            [100.0, 10.0, 20.0, 900.0],
            [101.0, 11.0, 21.0, 901.0],
            [102.0, 12.0, 22.0, 902.0],
            [103.0, 13.0, 23.0, 903.0],
        ], 1, 3)
        state = self.dataset.get_state_or_action(5, "state", "state.signal", 1)
        future = self.dataset.get_state_or_action(5, "action", "action.signal", 0)
        np.testing.assert_array_equal(state, [[11.0, 21.0]])
        np.testing.assert_array_equal(future, [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])

    def test_per_camera_file_indices_override_data_indices(self):
        self.dataset.trajectory_ids_to_metadata = {5: {
            "data/chunk_index": 0, "data/file_index": 0,
            "videos/file_indices": {
                "observation.images.head_left": {"chunk_index": 2, "file_index": 7},
                "observation.images.wrist_left": {"chunk_index": 3, "file_index": 11},
            },
        }}
        for key, suffix in (("head", "head_left/chunk-002/file-007.mp4"),
                            ("wrist", "wrist_left/chunk-003/file-011.mp4")):
            with self.subTest(camera=key):
                expected = self.root / ("videos/observation.images." + suffix)
                self.assertEqual(self.dataset.get_video_path(5, key), expected)

    def test_video_indices_fall_back_to_data_indices(self):
        self.dataset.trajectory_ids_to_metadata = {5: {
            "data/chunk_index": 4, "data/file_index": 9,
        }}
        expected = self.root / (
            "videos/observation.images.head_left/chunk-004/file-009.mp4"
        )
        self.assertEqual(self.dataset.get_video_path(5, "head"), expected)


if __name__ == "__main__":
    unittest.main()
