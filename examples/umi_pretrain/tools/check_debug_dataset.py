"""Check the private world-pose-16D dataset against unchanged source rows."""
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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from starVLA.dataloader.gr00t_lerobot import datasets as dataset_module
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.lerobot_datasets import collate_fn
from umi_valid_windows import (
    DATA_VERSION, HORIZON, SIGNAL_COLUMNS, VIDEO_KEYS,
    load_private_umi, dataset_indices, raw_window_statistics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO.parent / "datasets/roban_umi_debug")
    args = parser.parse_args()
    root = args.dataset.resolve()
    if not root.is_relative_to((REPO.parent / "datasets").resolve()):
        raise ValueError("The checker only accepts private datasets")
    report_path = root / f"debug_check.{DATA_VERSION}.json"
    report = {"data_version": DATA_VERSION, "dataset": str(root), "passed": False}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    audit = load_private_umi(root, horizon=HORIZON)
    raw, info, mapping, tasks = (audit[k] for k in ("raw", "info", "mapping", "tasks"))
    config = ROBOT_TYPE_CONFIG_MAP["roban_umi"]
    if config.action_indices != list(range(1, HORIZON + 1)):
        raise ValueError("Registry must select future frames 1 through 16")
    if config.state_indices != [0] or config.observation_indices != [0]:
        raise ValueError("State and image observations must use the current frame")
    data_cfg = {
        "lerobot_version": "v3.0", "include_state": True,
        "action_mode": "abs", "lowdim_dtype": "float32",
        "strict_window_sampling": True,
        "raw_lowdim_statistics": raw_window_statistics(audit),
    }
    dataset = dataset_module.LeRobotSingleDataset(
        dataset_path=root, modality_configs=config.modality_config(),
        embodiment_tag=config.embodiment_tag, transforms=config.transform(),
        video_backend="torchvision_av", data_cfg=data_cfg,
    )
    allowed = dataset_indices(dataset, audit["valid_steps"])
    if not allowed:
        raise ValueError("No complete valid windows are available")
    chosen = sorted({0, len(allowed) // 2, len(allowed) - 1})
    report.update({
        "modality_path": str(root / "meta/modality.json"),
        "pose_sources": {
            f"{group}.{key}": mapping[group][key]["original_key"]
            for group in ("state", "action") for key in ("robot1_pose", "robot2_pose")
        },
        "normalization": "none; raw FP32 measured world poses and gripper widths",
        "data_cfg": data_cfg,
        "video_order": list(VIDEO_KEYS),
        "window_audit": audit["report"],
        "valid_dataset_indices": allowed,
        "checks": [],
    })
    episodes = audit["episodes"].set_index("episode_index", verify_integrity=True)
    # Select raw rows by episode and row offset; frame numbers are reported separately.
    for order, subset_index in enumerate(chosen):
        ds_index = allowed[subset_index]
        episode_id, base_index = map(int, dataset.all_steps[ds_index])
        rows = np.flatnonzero(raw["episode_index"].to_numpy() == episode_id)
        positions = rows[base_index:base_index + HORIZON + 1]
        if len(positions) != HORIZON + 1:
            raise AssertionError("A selected window would require padding")
        ep = episodes.loc[episode_id]
        with patch.object(dataset_module, "get_frames_by_timestamps",
                          wraps=dataset_module.get_frames_by_timestamps) as decode:
            sample = dataset[ds_index]
        expected_state = audit["values"][positions[:1]]
        expected_action = audit["values"][positions[1:]]
        field_errors = {}
        for key, expected, shape in (
            ("state", expected_state, (1, 16)),
            ("action", expected_action, (HORIZON, 16)),
        ):
            actual = sample[key]
            if actual.dtype != np.dtype("float32") or actual.shape != shape:
                raise AssertionError(f"{key}: expected FP32 {shape}, got {actual.dtype} {actual.shape}")
            np.testing.assert_array_equal(actual, expected)
            if not np.isfinite(actual).all():
                raise AssertionError(f"{key} contains non-finite values")
            field_errors[key] = {
                source: float(np.max(np.abs(actual[:, begin:end] - expected[:, begin:end])))
                for source, begin, end in zip(SIGNAL_COLUMNS, (0, 7, 8, 15), (7, 8, 15, 16))
            }
        assert len(sample["image"]) == 4
        assert all(isinstance(im, Image.Image) and im.size == (224, 224) for im in sample["image"])
        assert sample["lang"] == tasks[int(raw.iloc[positions[0]]["task_index"])]
        assert decode.call_count == 4
        video_checks = {}
        for logical, source, call in zip(config.video_keys, VIDEO_KEYS, decode.call_args_list):
            assert mapping["video"][logical.split(".", 1)[1]]["original_key"] == source
            offset = float(ep[f"videos/{source}/from_timestamp"])
            wanted = float(raw.iloc[positions[0]]["timestamp"]) + offset
            np.testing.assert_allclose(call.args[1], [wanted], rtol=0, atol=3e-6)
            path = root / info["video_path"].format(
                video_key=source, chunk_index=int(ep[f"videos/{source}/chunk_index"]),
                file_index=int(ep[f"videos/{source}/file_index"]),
            )
            assert Path(call.args[0]) == path and path.is_file()
            video_checks[source] = {
                "path": str(path), "resolved_path": str(path.resolve()),
                "from_timestamp": offset, "requested_timestamp": wanted,
            }
        report["checks"].append({
            "subset_index": subset_index, "dataset_index": ds_index,
            "episode_index": episode_id, "episode_row_offset": base_index,
            "frame_index": int(raw.iloc[positions[0]]["frame_index"]),
            "future_frame_indices": raw.iloc[positions[1:]]["frame_index"].astype(int).tolist(),
            "raw_row_positions": positions.tolist(),
            "state_shape": list(sample["state"].shape), "state_dtype": str(sample["state"].dtype),
            "action_shape": list(sample["action"].shape), "action_dtype": str(sample["action"].dtype),
            "max_absolute_error_by_source": field_errors,
            "videos": video_checks, "lang": sample["lang"],
        })
        if order == 0:
            preview = Image.new("RGB", (448, 492), "white")
            draw = ImageDraw.Draw(preview)
            for i, (name, im) in enumerate(zip(config.video_keys, sample["image"])):
                x, y = (i % 2) * 224, (i // 2) * 246
                draw.text((x + 4, y + 3), name, fill="black")
                preview.paste(im, (x, y + 22))
            preview.save(root / f"debug_preview.{DATA_VERSION}.jpg")

    batch = next(iter(DataLoader(Subset(dataset, allowed), batch_size=2,
                                num_workers=0, collate_fn=collate_fn)))
    actions = np.stack([s["action"] for s in batch])
    states = np.stack([s["state"] for s in batch])
    assert actions.shape == (len(batch), HORIZON, 16) and actions.dtype == np.float32
    assert states.shape == (len(batch), 1, 16) and states.dtype == np.float32
    report["batch"] = {"action_shape": list(actions.shape), "state_shape": list(states.shape)}
    report["passed"] = True
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "passed": True, "data_version": DATA_VERSION, "report": str(report_path),
        "valid_windows": len(allowed),
        "checked_frames": [c["frame_index"] for c in report["checks"]],
        "batch": report["batch"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
