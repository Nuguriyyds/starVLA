"""Extract curated episode 0; write only to a new private directory."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
PRIVATE = REPO.parent.resolve()
SOURCE_ROOT = Path("/mnt/workspace/public/roban_umi/restricted_data").resolve()
DEFAULT_SOURCE = SOURCE_ROOT / "derived/umi_v30_curated_v1"


def inside(path, root):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path is outside {root}: {path}")
    return path


def select(table, name, value):
    return table.filter(pc.equal(table[name], value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=PRIVATE / "datasets/roban_umi_debug")
    args = parser.parse_args()
    source = inside(args.source, SOURCE_ROOT)
    output = inside(args.output, PRIVATE)
    if output.exists():
        raise FileExistsError(f"Output already exists; reuse it or choose a new --output: {output}")
    modality = REPO / "examples/umi_pretrain/train_files/modality.json"
    mapping = json.loads(modality.read_text())
    info = json.loads((source / "meta/info.json").read_text())
    episodes = select(pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet"), "episode_index", 0)
    if len(episodes) != 1:
        raise ValueError("Expected exactly one metadata row for episode 0")
    ep = episodes.to_pylist()[0]
    data_rel = info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
    data = select(pq.read_table(inside(source / data_rel, SOURCE_ROOT)), "episode_index", 0)
    if len(data) != ep["length"] or len(data) < 9:
        raise ValueError("Incomplete episode or insufficient frames for the action horizon")
    if not np.array_equal(data["frame_index"].to_numpy(), np.arange(len(data))):
        raise ValueError("This helper expects episode 0 to start at frame 0")
    if data["timestamp"][0].as_py() != 0 or data["index"][0].as_py() != 0:
        raise ValueError("This helper expects the first episode to start at time/index 0")
    for group in ("state", "action"):
        for field in mapping[group].values():
            if field["original_key"] not in data.column_names:
                raise ValueError(f"Missing source column: {field['original_key']}")
    task_ids = sorted(set(data["task_index"].to_pylist()))
    tasks = pq.read_table(source / "meta/tasks.parquet")
    tasks = tasks.filter(pc.is_in(tasks["task_index"], value_set=data["task_index"].combine_chunks().unique()))
    if sorted(tasks["task_index"].to_pylist()) != task_ids:
        raise ValueError("Missing or duplicate task descriptions")
    videos = []
    for key in [v["original_key"] for v in mapping["video"].values()]:
        rel = info["video_path"].format(video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"], file_index=ep[f"videos/{key}/file_index"])
        src_video = inside(source / rel, SOURCE_ROOT)
        target = inside(output / rel, output)
        if not src_video.is_file():
            raise FileNotFoundError(src_video)
        videos.append((key, src_video, target, ep[f"videos/{key}/from_timestamp"]))
    # All inputs are checked before creating the private output.
    output.mkdir(parents=True)
    data_path = inside(output / data_rel, output)
    data_path.parent.mkdir(parents=True)
    pq.write_table(data, data_path)
    meta = output / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    pq.write_table(episodes, meta / "episodes/chunk-000/file-000.parquet")
    pq.write_table(tasks, meta / "tasks.parquet")
    info.update(total_episodes=1, total_frames=len(data), total_tasks=len(tasks), splits={"train": "0:1"})
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(modality, meta / "modality.json")
    for _, src_video, target, _ in videos:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(src_video)
    manifest = {
        "source": str(source), "episode_index": 0, "frames": len(data),
        "task_ids": task_ids, "video_mode": "symlink; original offsets preserved",
        "videos": [{"key": k, "source": str(s), "from_timestamp": t} for k, s, _, t in videos],
    }
    (meta / "debug_subset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": len(data), "episodes": 1, "views": len(videos)}, indent=2))


if __name__ == "__main__":
    main()
