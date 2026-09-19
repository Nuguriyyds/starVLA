"""Mergeable FP64 statistics for read-only UMI quality diagnostics.

This module does not decide whether frames or training windows are valid.
Histogram quantiles are *brackets*, not exact or interpolated quantiles.
No optional model, video, or accelerator dependencies are imported.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
_NEG_INF = "-inf"
_POS_INF = "+inf"

_POSITION_EDGES = [
    _NEG_INF, 0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 50, 100, 300, _POS_INF,
]
_TRANSLATION_EDGES = [
    _NEG_INF, 0, 0.00001, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02,
    0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 50, 100, 300, _POS_INF,
]
_GRIPPER_EDGES = [
    _NEG_INF, 0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.015,
    0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065,
    0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1, _POS_INF,
]
_ANGLE_EDGES = [
    _NEG_INF, 0, 0.1, 0.5, 1, 2, 5, 10, 15, 30, 60, 90, 120, 150,
    180, _POS_INF,
]
_QNORM_EDGES = [
    _NEG_INF, 0, 0.5, 0.9, 0.99, 0.999, 1 - 1e-6, 1 - 1e-7,
    1 - 5e-8, 1 - 1e-8, 1, 1 + 1e-8, 1 + 5e-8, 1 + 1e-7,
    1 + 1e-6, 1.001, 1.01, 1.1, 2, _POS_INF,
]

# Fixed threshold counters use direct comparisons, independently of histogram
# edge conventions. They are diagnostics, never rejection criteria.
_METRICS = {
    "position_norm": (
        "m", _POSITION_EDGES,
        [("gt_5", "gt", 5.0), ("gt_10", "gt", 10.0), ("gt_50", "gt", 50.0)],
    ),
    "gripper": (
        "m", _GRIPPER_EDGES,
        [("lt_0", "lt", 0.0), ("gt_0.06", "gt", 0.06),
         ("gt_0.1", "gt", 0.1), ("gt_0.2", "gt", 0.2)],
    ),
    "quaternion_norm": (
        "dimensionless", _QNORM_EDGES,
        [("eq_0", "eq", 0.0), ("lt_0.99", "lt", 0.99),
         ("gt_1.01", "gt", 1.01),
         ("abs_deviation_from_1_gt_1e-8", "absdev1_gt", 1e-8),
         ("abs_deviation_from_1_gt_1e-7", "absdev1_gt", 1e-7)],
    ),
    "rotation_deg": (
        "deg", _ANGLE_EDGES,
        [("gt_15", "gt", 15.0), ("gt_30", "gt", 30.0), ("gt_90", "gt", 90.0)],
    ),
    "translation": (
        "m", _TRANSLATION_EDGES,
        [("gt_0.1", "gt", 0.1), ("gt_0.5", "gt", 0.5),
         ("gt_1", "gt", 1.0), ("gt_5", "gt", 5.0),
         ("gt_10", "gt", 10.0), ("gt_50", "gt", 50.0)],
    ),
    "negative_dot_rotation_deg": (
        "deg", _ANGLE_EDGES,
        [("gt_15", "gt", 15.0), ("gt_30", "gt", 30.0), ("gt_90", "gt", 90.0)],
    ),
}


def _definition(metric: str):
    try:
        return _METRICS[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown quality metric {metric!r}; choose {sorted(_METRICS)}") from exc


def _numeric_edges(edges):
    return np.asarray(
        [-math.inf if x == _NEG_INF else math.inf if x == _POS_INF else float(x)
         for x in edges], dtype=np.float64,
    )


def _empty(metric: str) -> dict[str, Any]:
    unit, edges, checks = _definition(metric)
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": metric,
        "unit": unit,
        "population": "finite scalar values only; nonfinite inputs counted separately",
        "count": 0,
        "nonfinite_count": 0,
        "nan_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "mean": None,
        "M2": 0.0,
        "min": None,
        "max": None,
        "hist_edges": list(edges),
        "hist_interval": "[lower, upper); infinities encoded as strings",
        "hist_counts": [0] * (len(edges) - 1),
        "diagnostic_definitions": {
            name: {"comparison": op, "value": value} for name, op, value in checks
        },
        "diagnostic_counts": {name: 0 for name, _, _ in checks},
    }


def summarize(values, metric: str) -> dict[str, Any]:
    """Summarize already-computed scalar metric values with FP64 moments.

    The caller must compute quaternion_norm before any normalization. For
    rotation_deg it should supply the sign-invariant principal angle obtained
    using abs(dot(q1_unit, q2_unit)). For negative_dot_rotation_deg, supply those
    same physical angles only for pairs whose original signed dot is negative.
    This function does not compute either angle or filter windows.
    """
    result = _empty(metric)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    result["nan_count"] = int(np.count_nonzero(np.isnan(arr)))
    result["positive_infinity_count"] = int(np.count_nonzero(np.isposinf(arr)))
    result["negative_infinity_count"] = int(np.count_nonzero(np.isneginf(arr)))
    result["nonfinite_count"] = (
        result["nan_count"] + result["positive_infinity_count"]
        + result["negative_infinity_count"]
    )
    arr = arr[np.isfinite(arr)]
    count = int(arr.size)
    result["count"] = count
    if not count:
        return result
    mean = float(np.mean(arr, dtype=np.float64))
    centered = arr - mean
    m2 = float(np.sum(centered * centered, dtype=np.float64))
    if not math.isfinite(mean) or not math.isfinite(m2):
        raise ValueError(f"FP64 moments overflow for {metric}; cannot serialize honest statistics")
    result.update(mean=mean, M2=m2, min=float(arr.min()), max=float(arr.max()))
    # Bins include underflow and overflow; all finite values have one bin.
    bins = np.searchsorted(_numeric_edges(result["hist_edges"]), arr, side="right") - 1
    result["hist_counts"] = np.bincount(
        bins, minlength=len(result["hist_counts"]),
    ).astype(np.int64).tolist()
    for name, op, threshold in _definition(metric)[2]:
        if op == "gt":
            mask = arr > threshold
        elif op == "lt":
            mask = arr < threshold
        elif op == "eq":
            mask = arr == threshold
        elif op == "absdev1_gt":
            mask = np.abs(arr - 1.0) > threshold
        else:
            raise AssertionError(f"Unknown diagnostic comparison: {op}")
        result["diagnostic_counts"][name] = int(np.count_nonzero(mask))
    return result


def merge(target: dict | None, incoming: dict) -> dict:
    """Merge raw summaries in place using Chan's parallel FP64 M2 formula.

    A None or empty-dict target starts a new accumulator. Inputs must be the raw
    summarize/merge structure; save this structure in per-file resume artifacts.
    finalize produces an additional view and is not needed for accumulation.
    """
    if target is incoming:
        raise ValueError("Cannot merge an accumulator into itself")
    if not target:
        copied = copy.deepcopy(incoming)
        if target is None:
            return copied
        target.update(copied)
        return target
    for key in (
        "schema_version", "metric", "unit", "population", "hist_edges",
        "hist_interval", "diagnostic_definitions",
    ):
        if target.get(key) != incoming.get(key):
            raise ValueError(f"Incompatible summary field {key!r} for {target.get('metric')}")
    if len(target["hist_counts"]) != len(incoming["hist_counts"]):
        raise ValueError("Incompatible histogram sizes")
    n_a, n_b = int(target["count"]), int(incoming["count"])
    total = n_a + n_b
    if n_b:
        if not n_a:
            target.update(
                mean=float(incoming["mean"]), M2=float(incoming["M2"]),
                min=float(incoming["min"]), max=float(incoming["max"]),
            )
        else:
            delta = float(incoming["mean"]) - float(target["mean"])
            mean = float(target["mean"]) + delta * (n_b / total)
            m2 = (
                float(target["M2"]) + float(incoming["M2"])
                + delta * delta * (n_a / total) * n_b
            )
            if not math.isfinite(mean) or not math.isfinite(m2):
                raise ValueError(f"FP64 merged moments overflow for {target['metric']}")
            target.update(
                mean=mean, M2=m2, min=min(target["min"], incoming["min"]),
                max=max(target["max"], incoming["max"]),
            )
    target["count"] = total
    for key in (
        "nonfinite_count", "nan_count",
        "positive_infinity_count", "negative_infinity_count",
    ):
        target[key] = int(target[key]) + int(incoming[key])
    target["hist_counts"] = [
        int(a) + int(b) for a, b in zip(target["hist_counts"], incoming["hist_counts"])
    ]
    for name, value in incoming["diagnostic_counts"].items():
        target["diagnostic_counts"][name] += int(value)
    return target


def finalize(stats: dict) -> dict:
    """Return a JSON-safe report without changing the resume/merge accumulator.

    Population standard deviation is sqrt(M2/count). Quantiles are nearest-rank
    empirical quantiles localized to fixed histogram bins, with observed finite
    extrema tightening unbounded end bins. No within-bin interpolation is used.
    """
    result = copy.deepcopy(stats)
    count = int(result["count"])
    if sum(result["hist_counts"]) != count:
        raise ValueError("Histogram counts do not equal finite count")
    result["input_count"] = count + int(result["nonfinite_count"])
    result["std"] = math.sqrt(max(0.0, float(result["M2"])) / count) if count else None
    result["std_definition"] = "population standard deviation, sqrt(M2 / finite_count)"
    result["diagnostic_fractions"] = {
        name: int(value) / count if count else None
        for name, value in result["diagnostic_counts"].items()
    }
    result["quantile_method"] = (
        "nearest-rank empirical quantile bracket from fixed histogram; "
        "not an exact point estimate; no within-bin interpolation; "
        "coarse bins imply coarse uncertainty"
    )
    result["quantile_brackets"] = {}
    cumulative = np.cumsum(np.asarray(result["hist_counts"], dtype=np.int64))
    for label, quantile in (
        ("p01", 0.01), ("p05", 0.05), ("p50", 0.5), ("p95", 0.95),
        ("p99", 0.99), ("p999", 0.999),
    ):
        if not count:
            result["quantile_brackets"][label] = None
            continue
        rank = max(1, math.ceil(quantile * count))
        index = int(np.searchsorted(cumulative, rank, side="left"))
        lower_raw, upper_raw = result["hist_edges"][index:index + 2]
        lower = result["min"] if lower_raw == _NEG_INF else max(float(lower_raw), result["min"])
        upper = result["max"] if upper_raw == _POS_INF else min(float(upper_raw), result["max"])
        # Upper endpoints are inclusive only if tightened to an observed maximum.
        upper_inclusive = upper_raw == _POS_INF or result["max"] < float(upper_raw)
        result["quantile_brackets"][label] = {
            "probability": quantile,
            "nearest_rank": rank,
            "lower": lower,
            "upper": upper,
            "lower_inclusive": True,
            "upper_inclusive": upper_inclusive,
            "histogram_bin_index": index,
            "histogram_bin_count": int(result["hist_counts"][index]),
        }
    return result
