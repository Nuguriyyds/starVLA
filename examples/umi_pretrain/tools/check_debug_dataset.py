"""CPU-only check of real images, language, state, future targets, and padding."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from starVLA.dataloader.gr00t_lerobot import datasets as dataset_module
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP, DATASET_NAMED_MIXTURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO.parent / "datasets/roban_umi_debug")
    args = parser.parse_args()
    root = args.dataset.resolve()
    if not root.is_relative_to((REPO.parent / "datasets").resolve()):
        raise ValueError("This smoke checker only accepts a private dataset under your datasets directory")
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] != 1 or not 9 <= info["total_frames"] <= 10000:
        raise ValueError("Use a single small complete episode, not the full dataset")
    config = ROBOT_TYPE_CONFIG_MAP["roban_umi"]
    assert DATASET_NAMED_MIXTURES["roban_umi_debug"] == [("roban_umi_debug", 1.0, "roban_umi")]
    data_cfg = {"lerobot_version": "v3.0", "include_state": True, "action_mode": "abs"}
    dataset = dataset_module.LeRobotSingleDataset(
        dataset_path=root, modality_configs=config.modality_config(),
        embodiment_tag=config.embodiment_tag, transforms=config.transform(),
        video_backend="torchvision_av", data_cfg=data_cfg,
    )
    ep = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet").iloc[0]
    raw = pd.read_parquet(root / info["data_path"].format(chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"])))
    mapping = json.loads((root / "meta/modality.json").read_text())
    tasks = {int(row["task_index"]): row["task"] for row in pq.read_table(root / "meta/tasks.parquet").to_pylist()}

    def expected(group, keys, indices):
        parts = []
        for key in keys:
            spec = mapping[group][key.split(".", 1)[1]]
            values = np.stack(raw[spec["original_key"]].iloc[indices].tolist())
            if values.ndim == 1:
                values = values[:, None]
            parts.append(values[:, spec["start"]:spec["end"]])
        return np.concatenate(parts, axis=1).astype(np.float16)

    report = {"dataset": str(root), "frames": len(raw), "normalization": "none; raw-value data smoke check", "checks": []}
    for frame in [0, min(100, len(raw) - 9), len(raw) - 1]:
        assert tuple(dataset.all_steps[frame]) == (0, frame)
        with patch.object(dataset_module, "get_frames_by_timestamps", wraps=dataset_module.get_frames_by_timestamps) as decode:
            sample = dataset[frame]
        assert sample["state"].shape == (1, 16)
        assert sample["action"].shape == (8, 16)
        assert len(sample["image"]) == 4
        assert all(isinstance(im, Image.Image) and im.size == (224, 224) for im in sample["image"])
        future = np.minimum(frame + np.arange(1, 9), len(raw) - 1)
        np.testing.assert_array_equal(sample["state"], expected("state", config.state_keys, [frame]))
        np.testing.assert_array_equal(sample["action"], expected("action", config.action_keys, future))
        assert np.isfinite(sample["state"]).all() and np.isfinite(sample["action"]).all()
        assert sample["lang"] == tasks[int(raw.iloc[frame]["task_index"])]
        assert decode.call_count == 4
        video_times = {}
        for logical, call in zip(config.video_keys, decode.call_args_list):
            key = mapping["video"][logical.split(".", 1)[1]]["original_key"]
            wanted = float(raw.iloc[frame]["timestamp"]) + float(ep[f"videos/{key}/from_timestamp"])
            np.testing.assert_allclose(call.args[1], [wanted], rtol=0, atol=3e-6)
            path = root / info["video_path"].format(video_key=key, chunk_index=int(ep[f"videos/{key}/chunk_index"]), file_index=int(ep[f"videos/{key}/file_index"]))
            assert Path(call.args[0]) == path
            video_times[key] = wanted
        report["checks"].append({"frame": frame, "future_frames": future.tolist(), "state_shape": list(sample["state"].shape), "action_shape": list(sample["action"].shape), "video_timestamps": video_times, "lang": sample["lang"]})
        if frame == 0:
            preview = Image.new("RGB", (448, 492), "white")
            draw = ImageDraw.Draw(preview)
            for i, (name, im) in enumerate(zip(config.video_keys, sample["image"])):
                x, y = (i % 2) * 224, (i // 2) * 246
                draw.text((x + 4, y + 3), name, fill="black")
                preview.paste(im, (x, y + 22))
            preview.save(root / "debug_preview.jpg")
    report["passed"] = True
    (root / "debug_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
