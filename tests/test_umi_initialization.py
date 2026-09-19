"""CPU regression for malformed rows outside an otherwise valid UMI window."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples/umi_pretrain/tools"))

import umi_valid_windows as windows
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot import datasets as dataset_module


class UMIInitializationTests(unittest.TestCase):
    def test_fresh_dataset_uses_valid_window_statistics_without_cache(self):
        # Keep every temporary write inside this private checkout. Video decoding
        # is stubbed; initialization, parquet reading and lowdim packing are real.
        with TemporaryDirectory(prefix="umi-init-", dir=REPO / "tests") as temporary:
            private = Path(temporary)
            root = private / "datasets/sample"
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            mapping = json.loads((REPO / "examples/umi_pretrain/train_files/modality.json").read_text())
            (root / "meta/modality.json").write_text(json.dumps(mapping))

            count = 40
            raw = pd.DataFrame({
                "episode_index": np.zeros(count, dtype=np.int64),
                "frame_index": np.arange(count, dtype=np.int64),
                "index": np.arange(count, dtype=np.int64),
                "timestamp": np.arange(count, dtype=np.float32) / np.float32(30),
                "task_index": np.zeros(count, dtype=np.int64),
                windows.FRAME_VALIDITY: [True] * count,
            })
            for signal, width in zip(windows.SIGNAL_COLUMNS, windows.SIGNAL_WIDTHS):
                if width == 7:
                    raw[signal] = [np.array([i / 10, .12345679, .3, 0, 0, 0, 1], dtype=np.float32)
                                   for i in range(count)]
                else:
                    raw[signal] = np.arange(count, dtype=np.float32) / 1000 + np.float32(.12345679)
            expected = np.concatenate([
                np.stack(raw[signal].iloc[:17].tolist()).reshape(17, width)
                for signal, width in zip(windows.SIGNAL_COLUMNS, windows.SIGNAL_WIDTHS)
            ], axis=1).astype(np.float32)
            # The first complete window is intact. Row 20 cannot be stacked with
            # the other poses, so whole-episode statistics/stacking would fail.
            raw.at[20, windows.SIGNAL_COLUMNS[0]] = np.arange(6, dtype=np.float32)
            data_path = root / "data/chunk-000/file-000.parquet"
            raw.to_parquet(data_path, index=False)
            original_parquet = data_path.read_bytes()
            pd.DataFrame({"task_index": [0], "task": ["pick object"]}).to_parquet(
                root / "meta/tasks.parquet", index=False,
            )
            info = {
                "codebase_version": "v3.0", "fps": 30, "chunks_size": 1000,
                "total_episodes": 1, "total_frames": count,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "numeric_conversion": {"pose_order": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
                "features": {
                    key: {"dtype": "video", "shape": [8, 8, 3],
                          "names": ["height", "width", "channel"],
                          "video_info": {"video.fps": 30}}
                    for key in windows.VIDEO_KEYS
                },
            }
            (root / "meta/info.json").write_text(json.dumps(info))
            episode = {"episode_index": 0, "length": count, "tasks": ["pick object"],
                       "data/chunk_index": 0, "data/file_index": 0}
            for key in windows.VIDEO_KEYS:
                episode.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                                f"videos/{key}/from_timestamp": 0.0})
                video_path = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
                video_path.parent.mkdir(parents=True)
                video_path.touch()
            pd.DataFrame([episode]).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)

            with patch.object(windows, "REPO", private / "repo"):
                audit = windows.load_private_umi(root)
            self.assertEqual(audit["valid_steps"], [(0, i) for i in [0, 1, 2, 3, 21, 22, 23]])
            self.assertEqual(audit["report"]["total_frames"], count)
            self.assertFalse((root / "meta/stats_gr00t.json").exists())
            config = ROBOT_TYPE_CONFIG_MAP["roban_umi"]
            dataset = dataset_module.LeRobotSingleDataset(
                dataset_path=root, modality_configs=config.modality_config(),
                embodiment_tag=config.embodiment_tag, transforms=config.transform(),
                video_backend="torchvision_av",
                data_cfg={"lerobot_version": "v3.0", "include_state": True,
                          "action_mode": "abs", "lowdim_dtype": "float32",
                          "strict_window_sampling": True,
                          "raw_lowdim_statistics": windows.raw_window_statistics(audit)},
            )
            index = windows.dataset_indices(dataset, audit["valid_steps"])[0]
            with patch.object(dataset_module, "get_frames_by_timestamps",
                              return_value=np.zeros((1, 8, 8, 3), dtype=np.uint8)):
                sample = dataset[index]
            self.assertEqual(sample["state"].dtype, np.dtype("float32"))
            self.assertEqual(sample["action"].dtype, np.dtype("float32"))
            np.testing.assert_array_equal(sample["state"], expected[:1])
            np.testing.assert_array_equal(sample["action"], expected[1:])
            self.assertEqual(sample["lang"], "pick object")
            self.assertFalse((root / "meta/stats_gr00t.json").exists())
            self.assertFalse((root / "meta/stats_gr00t.tmp").exists())
            self.assertEqual(data_path.read_bytes(), original_parquet)
            saved = pd.read_parquet(data_path)
            self.assertEqual(len(saved), count)
            self.assertEqual(len(saved.iloc[20][windows.SIGNAL_COLUMNS[0]]), 6)


if __name__ == "__main__":
    unittest.main()
