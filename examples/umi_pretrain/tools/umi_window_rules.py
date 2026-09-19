"""Vectorized raw-row rules for full-scale world-pose UMI window indexing.

These rules mirror umi_valid_windows.select_valid_windows. They never normalize,
clip, reorder, renumber, pad, or repair samples. Only PyArrow and NumPy are used.
Duplicate (episode, frame) identities must be rejected by the caller over the
complete episode; evaluating independent batches cannot detect all duplicates.
"""
from collections.abc import Mapping
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

DATA_VERSION = "world-pose-16D"
HORIZON = 16
FRAME_VALIDITY = "observation.umi.robot0_frame_validity"
SIGNAL_COLUMNS = [
    "observation.umi.robot1_finger_eef_pose",
    "observation.umi.robot1_sensor_magnetic_encoder",
    "observation.umi.robot2_finger_eef_pose",
    "observation.umi.robot2_sensor_magnetic_encoder",
]
SIGNAL_WIDTHS = [7, 1, 7, 1]
SOURCE_COLUMNS = ["source_timestamp_ns"] + [
    f"{key}.source_timestamp_ns" for key in SIGNAL_COLUMNS + [FRAME_VALIDITY]
]
DEFAULT_SOURCE_TIMESTAMP_SENTINELS = (-1, 0, int(np.iinfo(np.int64).min))
INTEGER_MIN = np.iinfo(np.int64).min
INTEGER_MAX = np.iinfo(np.int64).max


def required_columns(schema):
    """Return only existing relevant physical fields, including presence masks."""
    fields = ["episode_index", "frame_index", "task_index", "timestamp",
              FRAME_VALIDITY] + SIGNAL_COLUMNS + SOURCE_COLUMNS
    fields += [f"schema_presence.{key}" for key in fields]
    names = set(schema.names)
    return [key for key in dict.fromkeys(fields) if key in names]


def _column(table, key):
    if key not in table.column_names:
        return None
    array = table.column(key)
    return array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array


def _validity(array):
    return array.is_valid().to_numpy(zero_copy_only=False).copy()


def _list_layout(array):
    """Return child data, start offsets, lengths without expanding Python rows."""
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        return array.values, offsets[:-1], np.diff(offsets)
    if pa.types.is_fixed_size_list(array.type):
        width = array.type.list_size
        starts = (np.arange(len(array), dtype=np.int64) + array.offset) * width
        return array.values, starts, np.full(len(array), width, dtype=np.int64)
    return None


def _scalar_array(array, n):
    """Match old _scalar: primitive scalar or a one-element list only."""
    if array is None:
        return None, np.zeros(n, dtype=bool)
    valid = _validity(array)
    layout = _list_layout(array)
    if layout is None:
        return array, valid
    child, starts, lengths = layout
    valid &= lengths == 1
    # Invalid/null lists must not be used as child indices, including empty child.
    indices = pa.array(starts, mask=~valid, type=pa.int64())
    scalar = pc.take(child, indices)
    return scalar, valid & _validity(scalar)


def _integer_column(table, key):
    n = len(table)
    array, valid = _scalar_array(_column(table, key), n)
    result = np.full(n, INTEGER_MIN, dtype=np.int64)
    if array is None:
        return result, valid
    if pa.types.is_integer(array.type):
        # fill_null avoids Arrow promoting nullable int64 to float64 and losing ns.
        raw = pc.fill_null(array, 0).to_numpy(zero_copy_only=False)
        if pa.types.is_unsigned_integer(array.type):
            valid &= raw <= INTEGER_MAX
        result[valid] = raw[valid].astype(np.int64)
    elif pa.types.is_floating(array.type):
        raw = array.to_numpy(zero_copy_only=False).astype(np.float64)
        # 2**63 is not representable as signed int64; use strict upper bound.
        valid &= (np.isfinite(raw) & (raw >= -(2.0 ** 63))
                  & (raw < 2.0 ** 63) & (np.floor(raw) == raw))
        result[valid] = raw[valid].astype(np.int64)
    else:
        valid[:] = False
    return result, valid


def _strict_true(table, key):
    n = len(table)
    array, valid = _scalar_array(_column(table, key), n)
    if array is None or not pa.types.is_boolean(array.type):
        return np.zeros(n, dtype=bool)
    return valid & pc.fill_null(array, False).to_numpy(zero_copy_only=False)


def _timestamp_column(table):
    n = len(table)
    array, valid = _scalar_array(_column(table, "timestamp"), n)
    result = np.full(n, np.nan, dtype=np.float64)
    if array is not None and (
        pa.types.is_integer(array.type) or pa.types.is_floating(array.type)
    ):
        raw = array.to_numpy(zero_copy_only=False).astype(np.float64)
        result[valid] = raw[valid]
    return result


def _signal(table, key, width):
    """Return FP32[n,width]; a malformed row remains entirely NaN."""
    n = len(table)
    output = np.full((n, width), np.nan, dtype=np.float32)
    array = _column(table, key)
    valid = np.zeros(n, dtype=bool)
    if array is None:
        return output, valid
    valid = _validity(array)
    layout = _list_layout(array)
    if layout is None:
        # Scalar signals are permitted only for a width-one gripper.
        if width != 1 or not (
            pa.types.is_integer(array.type) or pa.types.is_floating(array.type)
        ):
            return output, np.zeros(n, dtype=bool)
        raw = array.to_numpy(zero_copy_only=False).reshape(n, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            converted = raw.astype(np.float32)
        valid &= np.isfinite(raw).all(axis=1) & np.isfinite(converted).all(axis=1)
        output[valid] = converted[valid]
        return output, valid
    child, starts, lengths = layout
    valid &= lengths == width
    if not (pa.types.is_integer(child.type) or pa.types.is_floating(child.type)):
        return output, np.zeros(n, dtype=bool)
    rows = np.flatnonzero(valid)
    if not len(rows):
        return output, valid
    indices = (starts[rows, None] + np.arange(width, dtype=np.int64)).reshape(-1)
    selected = pc.take(child, pa.array(indices, type=pa.int64()))
    child_valid = _validity(selected).reshape(-1, width).all(axis=1)
    raw = selected.to_numpy(zero_copy_only=False).reshape(-1, width)
    with np.errstate(over="ignore", invalid="ignore"):
        converted = raw.astype(np.float32)
    good = child_valid & np.isfinite(raw).all(axis=1) & np.isfinite(converted).all(axis=1)
    valid[rows] = good
    output[rows[good]] = converted[good]
    return output, valid


def evaluate_rows(table, tasks, *,
                  source_timestamp_sentinels=DEFAULT_SOURCE_TIMESTAMP_SENTINELS):
    """Evaluate a bounded Arrow table in physical row order.

    tasks is a set/array of validated IDs whose text is nonempty. A mapping of ID
    to text is also accepted and validated. *_valid denotes integer parse validity,
    not nonnegativity or task membership. Invalid integer outputs use int64 min.
    No frame payloads or Python record dictionaries are retained.
    """
    if len(table.column_names) != len(set(table.column_names)):
        raise ValueError("Duplicate table columns are ambiguous")
    n = len(table)
    episode, episode_valid = _integer_column(table, "episode_index")
    frame, frame_valid = _integer_column(table, "frame_index")
    task, task_valid = _integer_column(table, "task_index")
    timestamp = _timestamp_column(table)
    if isinstance(tasks, Mapping):
        task_ids = np.fromiter(
            (int(k) for k, value in tasks.items()
             if isinstance(value, str) and value.strip()), dtype=np.int64
        )
    elif isinstance(tasks, np.ndarray):
        task_ids = tasks.astype(np.int64, copy=False)
    else:
        task_ids = np.fromiter(tasks, dtype=np.int64)
    reasons = {
        "invalid_episode_or_frame_index": (
            ~episode_valid | ~frame_valid | (episode < 0) | (frame < 0)
        ),
        "missing_or_empty_task": ~task_valid | ~np.isin(task, task_ids),
        "invalid_timestamp": ~np.isfinite(timestamp),
        "missing_or_false_frame_validity": ~_strict_true(table, FRAME_VALIDITY),
    }
    present_source = [key for key in SOURCE_COLUMNS if key in table.column_names]
    relevant = SIGNAL_COLUMNS + [FRAME_VALIDITY]
    for key in relevant + ["episode_index", "frame_index", "timestamp", "task_index"] + present_source:
        presence = f"schema_presence.{key}"
        if presence in table.column_names:
            reasons[f"schema_absent:{key}"] = ~_strict_true(table, presence)
    values = np.full((n, 16), np.nan, dtype=np.float32)
    offset = 0
    for key, width in zip(SIGNAL_COLUMNS, SIGNAL_WIDTHS):
        values[:, offset:offset + width], good = _signal(table, key, width)
        reasons[f"invalid_signal:{key}"] = ~good
        offset += width
    for offset, label in ((3, "robot1"), (11, "robot2")):
        norm = np.linalg.norm(values[:, offset:offset + 4].astype(np.float64), axis=1)
        reasons[f"unavailable_quaternion:{label}"] = ~np.isfinite(norm) | (norm == 0)
    root_ns = None
    root_ns_valid = None
    sentinel_values = np.asarray(tuple(source_timestamp_sentinels), dtype=np.int64)
    for key in present_source:
        value, valid = _integer_column(table, key)
        reasons[f"unavailable_source_timestamp:{key}"] = (
            ~valid | np.isin(value, sentinel_values)
        )
        if key == "source_timestamp_ns":
            root_ns, root_ns_valid = value, valid
    row_ok = np.ones(n, dtype=bool)
    for invalid in reasons.values():
        row_ok &= ~invalid
    return {
        "values": values, "episode": episode, "frame": frame, "task": task,
        "timestamp": timestamp, "root_ns": root_ns, "row_ok": row_ok,
        "row_reasons": reasons, "episode_valid": episode_valid,
        "frame_valid": frame_valid, "task_valid": task_valid,
        "root_ns_valid": root_ns_valid, "source_columns": present_source,
        "missing_source_columns": [key for key in SOURCE_COLUMNS if key not in present_source],
    }


def _different(values, valid):
    """Old integer-or-None equality: two unparsable values compare equal."""
    return ((valid[:-1] != valid[1:])
            | (valid[:-1] & valid[1:] & (values[:-1] != values[1:])))


def evaluate_edges(evaluated, fps, *,
                   timestamp_relative_tolerance=0.20,
                   timestamp_absolute_tolerance=1e-6):
    """Return reason masks for adjacent physical rows; callers retain boundaries."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if (not np.isfinite(timestamp_relative_tolerance)
            or not np.isfinite(timestamp_absolute_tolerance)
            or not 0 <= timestamp_relative_tolerance < 1
            or timestamp_absolute_tolerance < 0):
        raise ValueError("timestamp tolerances must be finite/nonnegative; relative < 1")
    interval = 1.0 / float(fps)
    tolerance = timestamp_relative_tolerance * interval + timestamp_absolute_tolerance
    frame = evaluated["frame"]
    valid = evaluated["frame_valid"]
    with np.errstate(over="ignore", invalid="ignore"):
        next_is_consecutive = (frame[:-1] < INTEGER_MAX) & (frame[1:] == frame[:-1] + 1)
        delta = np.diff(evaluated["timestamp"])
    reasons = {
        "episode_boundary": _different(evaluated["episode"], evaluated["episode_valid"]),
        "nonconsecutive_frame_index": ~(
            valid[:-1] & valid[1:] & next_is_consecutive
        ),
        "timestamp_gap": (
            ~np.isfinite(delta) | (delta <= 0) | (abs(delta - interval) > tolerance)
        ),
        "task_change": _different(evaluated["task"], evaluated["task_valid"]),
    }
    root = evaluated["root_ns"]
    if root is not None:
        valid = evaluated["root_ns_valid"]
        before, after = root[:-1], root[1:]
        # Same-sign differences fit int64. Rare opposite-sign pairs use Python
        # integers, preserving old arbitrary-precision subtraction exactly.
        same_sign = (before >= 0) == (after >= 0)
        delta = np.empty(len(before), dtype=np.float64)
        delta[same_sign] = (after[same_sign] - before[same_sign]).astype(np.float64) / 1e9
        for i in np.flatnonzero(~same_sign):
            delta[i] = (int(after[i]) - int(before[i])) / 1e9
        reasons["source_timestamp_gap"] = (
            ~valid[:-1] | ~valid[1:] | ~np.isfinite(delta) | (delta <= 0)
            | (abs(delta - interval) > tolerance)
        )
    return reasons
