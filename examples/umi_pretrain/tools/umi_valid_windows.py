"""Small private UMI datasets: immutable world-pose signals and full future windows.

Only numpy/pandas/pyarrow are required. No robot transforms, normalization, row
deletion, quaternion sign changes, gripper clipping, or target padding occur.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
DATA_VERSION = "world-pose-16D"
HORIZON = 16
MAX_FRAMES = 10000
FRAME_VALIDITY = "observation.umi.robot0_frame_validity"
SIGNAL_COLUMNS = [
    "observation.umi.robot1_finger_eef_pose",
    "observation.umi.robot1_sensor_magnetic_encoder",
    "observation.umi.robot2_finger_eef_pose",
    "observation.umi.robot2_sensor_magnetic_encoder",
]
VIDEO_KEYS = [
    "observation.images.head_left", "observation.images.head_right",
    "observation.images.wrist_left", "observation.images.wrist_right",
]
LOGICAL_SIGNALS = ["robot1_pose", "robot1_gripper", "robot2_pose", "robot2_gripper"]
LOGICAL_VIDEOS = ["left_head", "right_head", "left_hand", "right_hand"]
SIGNAL_WIDTHS = [7, 1, 7, 1]
DEFAULT_SOURCE_TIMESTAMP_SENTINELS = (-1, 0, int(np.iinfo(np.int64).min))


def validate_mapping(mapping):
    """Fail closed if metadata selects a different representation or ordering."""
    if not isinstance(mapping, dict):
        raise ValueError("modality mapping must be an object")
    for group in ("state", "action"):
        fields = mapping.get(group)
        if not isinstance(fields, dict) or list(fields) != LOGICAL_SIGNALS:
            raise ValueError(f"{group} order must be {LOGICAL_SIGNALS}")
        for logical, source, width in zip(LOGICAL_SIGNALS, SIGNAL_COLUMNS, SIGNAL_WIDTHS):
            spec = fields[logical]
            if not isinstance(spec, dict) or (
                spec.get("original_key") != source
                or type(spec.get("start")) is not int or spec["start"] != 0
                or type(spec.get("end")) is not int or spec["end"] != width
                or spec.get("absolute") is not True
                or spec.get("dtype") != "float32"
            ):
                raise ValueError(
                    f"{group}.{logical} must select raw {source}[0:{width}], "
                    "absolute=true, dtype=float32"
                )
    videos = mapping.get("video")
    if not isinstance(videos, dict) or list(videos) != LOGICAL_VIDEOS:
        raise ValueError(f"video order must be {LOGICAL_VIDEOS}")
    for logical, source in zip(LOGICAL_VIDEOS, VIDEO_KEYS):
        if videos[logical] != {"original_key": source}:
            raise ValueError(f"video.{logical} must select {source}")
    if mapping.get("annotation") != {
        "human.action.task_description": {"original_key": "task_index"}
    }:
        raise ValueError("language annotation must select task_index")


def _scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape in ((), (1,)) else None


def _integer(value):
    value = _scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return None
    if not np.isfinite(value) or int(value) != value:
        return None
    return int(value)


def _true(value):
    value = _scalar(value)
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _distribution(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(finite.min()), "max": float(finite.max()),
        "mean": float(finite.mean()), "std": float(finite.std()),
        "quantiles": dict(zip(
            ["p01", "p05", "p50", "p95", "p99"],
            [float(x) for x in np.quantile(finite, [.01, .05, .5, .95, .99])],
        )),
    }


def select_valid_windows(
    raw, tasks, fps, horizon=HORIZON, *,
    timestamp_relative_tolerance=0.20, timestamp_absolute_tolerance=1e-6,
    source_timestamp_sentinels=DEFAULT_SOURCE_TIMESTAMP_SENTINELS,
):
    """Return (valid_steps, JSON-safe report, float32 values) in original row order.

    Optional source timestamp columns are absolute Unix nanoseconds; available
    columns must contain integer values outside the configured sentinel set.
    Missing columns are reported, not fabricated. No source gap_ns cutoff is used.
    """
    if horizon != HORIZON:
        raise ValueError(f"{DATA_VERSION} requires horizon={HORIZON}")
    if not 1 <= len(raw) <= MAX_FRAMES:
        raise ValueError(f"Use a small nonempty dataset with at most {MAX_FRAMES} frames")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if (not np.isfinite(timestamp_relative_tolerance)
            or not np.isfinite(timestamp_absolute_tolerance)
            or not 0 <= timestamp_relative_tolerance < 1
            or timestamp_absolute_tolerance < 0):
        raise ValueError("timestamp tolerances must be finite and nonnegative; relative < 1")
    if not raw.columns.is_unique:
        raise ValueError("Duplicate dataframe column names are ambiguous")
    n = len(raw)
    values = np.full((n, 16), np.nan, dtype=np.float32)
    reasons = [set() for _ in range(n)]
    episodes, frames, task_ids = [], [], []
    timestamps = np.full(n, np.nan, dtype=np.float64)
    relevant = SIGNAL_COLUMNS + [FRAME_VALIDITY]
    source_columns = ["source_timestamp_ns"] + [
        f"{key}.source_timestamp_ns" for key in relevant
    ]
    present_source_columns = [key for key in source_columns if key in raw]
    sentinels = set(int(x) for x in source_timestamp_sentinels)
    records = raw.to_dict("records")

    for pos, row in enumerate(records):
        episode, frame, task = [_integer(row.get(key)) for key in
                                ("episode_index", "frame_index", "task_index")]
        episodes.append(episode)
        frames.append(frame)
        task_ids.append(task)
        if episode is None or episode < 0 or frame is None or frame < 0:
            reasons[pos].add("invalid_episode_or_frame_index")
        if task is None or task not in tasks or not isinstance(tasks[task], str) or not tasks[task].strip():
            reasons[pos].add("missing_or_empty_task")
        timestamp = _scalar(row.get("timestamp"))
        if isinstance(timestamp, (int, float, np.number)) and not isinstance(timestamp, (bool, np.bool_)):
            timestamps[pos] = timestamp
        if not np.isfinite(timestamps[pos]):
            reasons[pos].add("invalid_timestamp")
        if not _true(row.get(FRAME_VALIDITY)):
            reasons[pos].add("missing_or_false_frame_validity")
        for key in relevant + ["episode_index", "frame_index", "timestamp", "task_index"] + present_source_columns:
            presence = f"schema_presence.{key}"
            if presence in raw and not _true(row.get(presence)):
                reasons[pos].add(f"schema_absent:{key}")
        offset = 0
        for key, width in zip(SIGNAL_COLUMNS, SIGNAL_WIDTHS):
            try:
                arr = np.asarray(row.get(key))
                if width == 1 and arr.shape == ():
                    arr = arr.reshape(1)
                if arr.shape != (width,) or arr.dtype.kind not in "iuf" or not np.isfinite(arr).all():
                    raise ValueError("unavailable numeric signal")
                with np.errstate(over="ignore", invalid="ignore"):
                    converted = arr.astype(np.float32)
                if not np.isfinite(converted).all():
                    raise ValueError("signal outside float32 range")
                values[pos, offset:offset + width] = converted
            except (TypeError, ValueError, OverflowError):
                reasons[pos].add(f"invalid_signal:{key}")
            offset += width
        for offset, label in ((3, "robot1"), (11, "robot2")):
            norm = np.linalg.norm(values[pos, offset:offset + 4].astype(np.float64))
            if not np.isfinite(norm) or norm == 0:
                reasons[pos].add(f"unavailable_quaternion:{label}")
        for key in present_source_columns:
            value = _integer(row.get(key))
            if value is None or value in sentinels:
                reasons[pos].add(f"unavailable_source_timestamp:{key}")

    # A repeated source identity is ambiguous even when its values happen to match.
    identities = Counter((ep, fr) for ep, fr in zip(episodes, frames)
                         if ep is not None and fr is not None)
    for pos, step in enumerate(zip(episodes, frames)):
        if identities[step] > 1:
            reasons[pos].add("duplicate_episode_frame")
    expected_interval = 1.0 / float(fps)
    tolerance = timestamp_relative_tolerance * expected_interval + timestamp_absolute_tolerance
    edges = []
    for pos in range(n - 1):
        issues = set()
        if episodes[pos] != episodes[pos + 1]:
            issues.add("episode_boundary")
        if frames[pos] is None or frames[pos + 1] is None or frames[pos + 1] != frames[pos] + 1:
            issues.add("nonconsecutive_frame_index")
        delta = timestamps[pos + 1] - timestamps[pos]
        if not np.isfinite(delta) or delta <= 0 or abs(delta - expected_interval) > tolerance:
            issues.add("timestamp_gap")
        if "source_timestamp_ns" in raw:
            source_a = _integer(records[pos].get("source_timestamp_ns"))
            source_b = _integer(records[pos + 1].get("source_timestamp_ns"))
            # Subtract integer Unix nanoseconds first to preserve interval precision.
            source_delta = None if source_a is None or source_b is None else (source_b - source_a) / 1e9
            if (source_delta is None or source_delta <= 0
                    or abs(source_delta - expected_interval) > tolerance):
                issues.add("source_timestamp_gap")
        if task_ids[pos] != task_ids[pos + 1]:
            issues.add("task_change")
        edges.append(issues)

    valid_steps, starts = [], []
    episode_row_counts = Counter()
    episode_row_offsets = []
    for episode_id in episodes:
        episode_row_offsets.append(episode_row_counts[episode_id])
        episode_row_counts[episode_id] += 1
    window_counts = Counter()
    for pos in range(n):
        issues = set()
        if pos + horizon >= n:
            issues.add("incomplete_future")
        else:
            if any(reasons[pos:pos + horizon + 1]):
                issues.add("invalid_frame_in_window")
            for edge in edges[pos:pos + horizon]:
                issues.update(edge)
        if issues:
            window_counts.update(issues)
        else:
            valid_steps.append((int(episodes[pos]), episode_row_offsets[pos]))
            starts.append({
                "row_index": pos, "episode_index": int(episodes[pos]),
                "episode_row_index": episode_row_offsets[pos],
                "frame_index": int(frames[pos]),
                "future_frame_indices": [int(x) for x in frames[pos + 1:pos + horizon + 1]],
            })
    quaternion_diagnostics = {}
    for offset, label in ((3, "robot1"), (11, "robot2")):
        quaternions = values[:, offset:offset + 4].astype(np.float64)
        norms = np.linalg.norm(quaternions, axis=1)
        angles = []
        for pos in range(n - 1):
            if edges[pos] or reasons[pos] or reasons[pos + 1]:
                continue
            # Normalize temporary diagnostic copies only; model inputs stay raw.
            dot = np.dot(quaternions[pos], quaternions[pos + 1]) / (norms[pos] * norms[pos + 1])
            angles.append(float(np.degrees(2 * np.arccos(np.clip(abs(dot), 0, 1)))))
        quaternion_diagnostics[label] = {
            "norms": _distribution(norms),
            "absolute_norm_error_from_one": _distribution(abs(norms - 1)),
            "adjacent_rotation_degrees_sign_invariant": _distribution(angles),
        }
    report = {
        "data_version": DATA_VERSION, "horizon": horizon, "total_frames": n,
        "valid_frames": sum(not r for r in reasons), "valid_windows": len(valid_steps),
        "rejected_windows": n - len(valid_steps),
        "frame_rejections": dict(Counter(reason for row in reasons for reason in row)),
        "window_rejections": dict(window_counts),
        "invalid_rows": [
            {"row_index": pos, "episode_index": episodes[pos], "frame_index": frames[pos],
             "reasons": sorted(row_reasons)}
            for pos, row_reasons in enumerate(reasons) if row_reasons
        ],
        "valid_starts": starts,
        "quaternions": quaternion_diagnostics,
        "grippers": {
            "robot1": _distribution(values[:, 7]),
            "robot2": _distribution(values[:, 15]),
        },
        "policy": {
            "row_order": "original parquet order; no filtering or reindexing",
            "window": "current t and all future frames t+1 through t+16 must pass",
            "representation": "world xyz + quaternion xyzw + gripper, left then right",
            "dtype": "float32", "transform": "none", "normalization": "none",
            "quaternion_rejection": "nonfinite or zero norm; no sign changes or norm correction",
            "gripper_rejection": "missing/nonfinite/non-numeric only; no range clipping",
            "fps": float(fps), "timestamp_expected_interval_seconds": expected_interval,
            "timestamp_relative_tolerance": float(timestamp_relative_tolerance),
            "timestamp_absolute_tolerance_seconds": float(timestamp_absolute_tolerance),
            "timestamp_total_tolerance_seconds": tolerance,
            "source_timestamp_sentinels": sorted(sentinels),
            "source_timestamp_columns_checked": present_source_columns,
            "source_timestamp_columns_not_available": [
                key for key in source_columns if key not in raw
            ],
            "root_source_timestamp_interval_checked": "source_timestamp_ns" in raw,
            "root_source_timestamp_interval_policy": "same positive 1/fps tolerance as timestamp; subtract integer nanoseconds before conversion",
            "sensor_source_timestamp_interval_policy": "no cadence constraint",
            "source_gap_ns_limit": None,
            "frame_validity": "required strict boolean true, scalar or one element",
            "schema_presence": "when available, strict boolean true is required",
        },
    }
    return valid_steps, report, values


def _private_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset metadata path escapes private dataset: {relative}")
    return path


def _read_tasks(path):
    tasks_frame = pd.read_parquet(path)
    if "task_index" not in tasks_frame:
        if tasks_frame.index.name != "task_index":
            raise ValueError("Tasks must provide task_index")
        tasks_frame = tasks_frame.reset_index()
    if "task" not in tasks_frame:
        if not all(isinstance(x, str) for x in tasks_frame.index):
            raise ValueError("Tasks must provide task text")
        tasks_frame = tasks_frame.copy()
        tasks_frame["task"] = tasks_frame.index
    tasks = {}
    for raw_id, task in zip(tasks_frame["task_index"], tasks_frame["task"]):
        task_id = _integer(raw_id)
        if task_id is None or task_id in tasks:
            raise ValueError("Tasks contain missing/noninteger or duplicate task_index")
        tasks[task_id] = task
    return tasks


def load_private_umi(root: Path, horizon=HORIZON):
    """Read at most one small complete private episode without modifying it."""
    root = Path(root).resolve()
    if not root.is_relative_to((REPO.parent / "datasets").resolve()):
        raise ValueError("Use a private dataset under REPO.parent / datasets")
    info = json.loads(_private_path(root, "meta/info.json").read_text())
    if info.get("total_episodes") != 1 or not HORIZON + 1 <= info.get("total_frames", 0) <= MAX_FRAMES:
        raise ValueError(f"Use one complete private episode with {HORIZON + 1} to {MAX_FRAMES} frames")
    if info.get("numeric_conversion", {}).get("pose_order") != ["x", "y", "z", "qx", "qy", "qz", "qw"]:
        raise ValueError("numeric_conversion.pose_order must explicitly be x,y,z,qx,qy,qz,qw")
    mapping = json.loads(_private_path(root, "meta/modality.json").read_text())
    validate_mapping(mapping)
    episode_paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise ValueError("No v3 episode metadata found")
    episode_tables = [pq.ParquetFile(_private_path(root, path.relative_to(root))) for path in episode_paths]
    if sum(table.metadata.num_rows for table in episode_tables) != 1:
        raise ValueError("Private smoke dataset must contain exactly one episode metadata row")
    episodes = pd.concat([table.read().to_pandas() for table in episode_tables], ignore_index=True)
    episode = episodes.iloc[0]
    episode_id = _integer(episode.get("episode_index"))
    chunk, file_index = [_integer(episode.get(key))
                         for key in ("data/chunk_index", "data/file_index")]
    if episode_id is None or chunk is None or file_index is None:
        raise ValueError("Episode metadata is missing integer episode/data file identifiers")
    data_path = _private_path(root, info["data_path"].format(
        chunk_index=chunk, file_index=file_index, episode_index=episode_id
    ))
    parquet = pq.ParquetFile(data_path)
    if parquet.metadata.num_rows > MAX_FRAMES:
        raise ValueError("Private parquet exceeds small-data frame limit")
    raw = parquet.read().to_pandas()
    if len(raw) != info["total_frames"] or _integer(episode.get("length")) != len(raw):
        raise ValueError("Episode/info lengths do not match original parquet rows")
    if "episode_index" not in raw or any(_integer(x) != episode_id for x in raw["episode_index"]):
        raise ValueError("Private parquet must contain only the metadata episode")
    tasks = _read_tasks(_private_path(root, "meta/tasks.parquet"))
    valid_steps, report, values = select_valid_windows(raw, tasks, info["fps"], horizon)
    report["dataset"] = str(root)
    return {
        "info": info, "episodes": episodes, "raw": raw, "mapping": mapping,
        "tasks": tasks, "valid_steps": valid_steps, "report": report, "values": values,
    }



def raw_window_statistics(audit):
    """Metadata-only statistics for the union of audited small-data windows.

    These values are not applied to samples or written to a shared/full-data
    statistics cache. The raw-value debug loader still requires metadata stats.
    """
    positions = sorted({
        pos for start in audit["report"]["valid_starts"]
        for pos in range(start["row_index"], start["row_index"] + HORIZON + 1)
    })
    if not positions:
        raise ValueError("Cannot construct metadata statistics without valid windows")
    data = np.asarray(audit["values"][positions], dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 16 or not np.isfinite(data).all():
        raise ValueError("Valid-window values must be finite with 16 dimensions")
    stats = {}
    for source, begin, end in zip(SIGNAL_COLUMNS, (0, 7, 8, 15), (7, 8, 15, 16)):
        values = data[:, begin:end]
        stats[source] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    audit["report"]["metadata_statistics"] = {
        "scope": "unique source rows covered by valid debug windows",
        "source_row_count": len(positions),
        "applied_to_samples": False, "persisted_as_dataset_cache": False,
    }
    return stats


def dataset_indices(dataset, valid_steps):
    """Resolve (episode, unchanged episode row offset) against dataset.all_steps."""
    lookup = {}
    for index, step in enumerate(dataset.all_steps):
        if len(step) != 2:
            raise ValueError("dataset.all_steps must contain episode/row-offset pairs")
        episode, frame = _integer(step[0]), _integer(step[1])
        if episode is None or frame is None:
            raise ValueError("dataset.all_steps has a noninteger source identity")
        key = (episode, frame)
        if key in lookup:
            raise ValueError(f"Duplicate dataset step: {key}")
        lookup[key] = index
    selected, seen = [], set()
    for step in valid_steps:
        key = tuple(step)
        if key in seen:
            raise ValueError(f"Duplicate requested valid step: {key}")
        if key not in lookup:
            raise ValueError(f"Valid source step missing from dataset.all_steps: {key}")
        seen.add(key)
        selected.append(lookup[key])
    return selected
