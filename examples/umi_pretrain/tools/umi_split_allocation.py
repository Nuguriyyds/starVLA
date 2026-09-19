"""Deterministic, whole-source-group allocation for UMI manifests.

This module assigns existing valid windows; it neither resamples windows nor
changes validity rules. A group is an indivisible source-collection unit, not
an episode inferred from a file number. Callers must form the source groups.

Public API::

    assignments, report = allocate_groups(
        [{"id": "source-A", "windows": 1234,
          "tasks": {17: 1000, 23: 234}, "sources": {"set-A": 1234}}],
        seed=42, validation_fraction=0.01, stages=5,
    )

``assignments`` maps every group ID to ``validation`` or ``stage_01`` ... .
Task keys are task IDs, not semantic task categories. Source keys identify
known source sets, not scenes, collectors, or devices inferred from names.
Weights are nonnegative integer counts of valid windows. A group may contain
several task/source labels. Labels may overlap; their sums need not equal the
group window count, and the report makes their total weight explicit. Empty
label dictionaries are accepted and provide no balancing information.

Method
------
First assign groups to validation and a training pool, with targets f and
1-f. Then assign the training pool to equally weighted stages. Both passes
use descending group size, with seeded BLAKE2b tie breaking, and a greedy
score combining global window load with task/source proportional load.

The total-load score is the incremental squared-load cost divided by the
group weight: (2 * load + group_weight) / target_weight. Label scores are
the group-label-weighted mean of 2 * already_assigned_label_weight /
target_label_weight. Thus a previously unseen label contributes no preference:
a singleton task is not forced into every split or systematically penalized
for entering validation. No inverse-frequency resampling is introduced.

An upper capacity of target + largest_group bounds global imbalance. A split
can exceed its target by at most the largest group in that allocation pass;
with B bins its shortfall is at most (B-1) * largest_group. These are bounds,
not an assertion that exact targets are attainable. There is no search or
global optimum guarantee. Very large indivisible groups and rare labels can
make the requested validation fraction or task proportions unattainable.

Complexity is O(G log G + G * B * labels_per_group); there is no all-task scan
inside the assignment loop. State is proportional to groups and observed
labels. No random state or Python process-dependent hash() is used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
from typing import Any, Iterable, Mapping


ALGORITHM_VERSION = "whole_group_weighted_greedy_v1"


@dataclass(frozen=True)
class _Group:
    __slots__ = ("gid", "windows", "tasks", "sources")

    gid: str
    windows: int
    tasks: tuple[tuple[str, int], ...]
    sources: tuple[tuple[str, int], ...]


def _digest(seed: int, namespace: str, gid: str) -> bytes:
    payload = f"{seed}\0{namespace}\0{gid}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _count(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return int(value)


def _labels(value: Any, description: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    # Canonicalize IDs loaded either from JSON object keys or integer tables.
    canonical: Counter[str] = Counter()
    for key, count in value.items():
        if isinstance(key, bool) or not isinstance(key, (str, Integral)):
            raise ValueError(f"{description} keys must be strings or integers")
        canonical[str(key)] += _count(count, f"{description}[{key!r}]")
    return tuple(sorted((key, count) for key, count in canonical.items() if count))


def _normalize(groups: Iterable[Mapping[str, Any]]) -> list[_Group]:
    result: list[_Group] = []
    seen: set[str] = set()
    for item in groups:
        gid = item["id"]
        if not isinstance(gid, str) or not gid:
            raise ValueError("Each group id must be a nonempty string")
        if gid in seen:
            raise ValueError(f"Duplicate group id: {gid}")
        seen.add(gid)
        group = _Group(
            gid=gid,
            windows=_count(item["windows"], f"{gid}.windows"),
            tasks=_labels(item.get("tasks", {}), f"{gid}.tasks"),
            sources=_labels(item.get("sources", {}), f"{gid}.sources"),
        )
        if group.windows == 0 and (group.tasks or group.sources):
            raise ValueError(f"Zero-window group {gid} cannot have label weight")
        result.append(group)
    if not result or sum(group.windows for group in result) == 0:
        raise ValueError("At least one positive-window group is required")
    return result


def _totals(groups: Iterable[_Group], field: str) -> Counter[str]:
    totals: Counter[str] = Counter()
    for group in groups:
        for key, count in getattr(group, field):
            totals[key] += count
    return totals


def _allocate_pass(
    groups: list[_Group],
    bins: tuple[tuple[str, float], ...],
    *,
    seed: int,
    namespace: str,
    task_balance: float,
    source_balance: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Allocate positive-weight groups; all sorting is seed-stable."""
    total = sum(group.windows for group in groups)
    if not groups:
        return {}, {"windows": 0, "groups": 0, "splits": {}}
    largest = max(group.windows for group in groups)
    ordered = sorted(
        groups,
        key=lambda group: (-group.windows, _digest(seed, namespace, group.gid), group.gid),
    )
    task_total = _totals(groups, "tasks")
    source_total = _totals(groups, "sources")
    loads = [0] * len(bins)
    task_loads: list[defaultdict[str, int]] = [defaultdict(int) for _ in bins]
    source_loads: list[defaultdict[str, int]] = [defaultdict(int) for _ in bins]
    assignments: dict[str, str] = {}
    for group in ordered:
        w = group.windows
        task_weight = sum(value for _, value in group.tasks)
        source_weight = sum(value for _, value in group.sources)
        # Store shared work once per label rather than once per candidate bin.
        task_terms = tuple(
            (key, value / (task_total[key] * task_weight))
            for key, value in group.tasks
        ) if task_weight else ()
        source_terms = tuple(
            (key, value / (source_total[key] * source_weight))
            for key, value in group.sources
        ) if source_weight else ()
        rotation = int.from_bytes(_digest(seed, namespace + "-bin", group.gid), "big") % len(bins)
        candidates: list[tuple[float, int, int]] = []
        for j, (_, fraction) in enumerate(bins):
            target = fraction * total
            if loads[j] + w > target + largest + 1e-9:
                continue
            score = (2.0 * loads[j] + w) / target
            if task_balance and task_terms:
                score += (2.0 * task_balance / fraction) * sum(
                    task_loads[j].get(key, 0) * factor for key, factor in task_terms
                )
            if source_balance and source_terms:
                score += (2.0 * source_balance / fraction) * sum(
                    source_loads[j].get(key, 0) * factor for key, factor in source_terms
                )
            candidates.append((score, (j - rotation) % len(bins), j))
        if not candidates:
            raise RuntimeError("Allocation capacity invariant failed")
        j = min(candidates)[2]
        assignments[group.gid] = bins[j][0]
        loads[j] += w
        for key, value in group.tasks:
            task_loads[j][key] += value
        for key, value in group.sources:
            source_loads[j][key] += value
    group_counts = Counter(assignments.values())
    return assignments, {
        "groups": len(groups),
        "windows": total,
        "largest_group_windows": largest,
        "maximum_over_target_bound_windows": largest,
        "maximum_under_target_bound_windows": (len(bins) - 1) * largest,
        "splits": {
            name: {
                "groups": group_counts[name],
                "windows": loads[j],
                "target_windows": fraction * total,
                "window_fraction": loads[j] / total,
                "deviation_from_target_windows": loads[j] - fraction * total,
            }
            for j, (name, fraction) in enumerate(bins)
        },
    }


def _total_variation(local: Mapping[str, int], reference: Mapping[str, int]) -> float | None:
    local_total = sum(local.values())
    reference_total = sum(reference.values())
    if not local_total or not reference_total:
        return None
    return 0.5 * math.fsum(
        abs(local.get(key, 0) / local_total - value / reference_total)
        for key, value in sorted(reference.items())
    )


def _distribution_report(
    groups: list[_Group], assignments: Mapping[str, str], names: list[str]
) -> dict[str, Any]:
    task_total = _totals(groups, "tasks")
    source_total = _totals(groups, "sources")
    tasks: dict[str, Counter[str]] = {name: Counter() for name in names}
    sources: dict[str, Counter[str]] = {name: Counter() for name in names}
    counts: Counter[str] = Counter()
    windows: Counter[str] = Counter()
    for group in groups:
        name = assignments[group.gid]
        counts[name] += 1
        windows[name] += group.windows
        for key, value in group.tasks:
            tasks[name][key] += value
        for key, value in group.sources:
            sources[name][key] += value
    return {
        "reference": "all supplied groups; task IDs and source-set IDs are not semantic classes",
        "global_task_id_count": len(task_total),
        "global_source_id_count": len(source_total),
        "global_task_label_weight": sum(task_total.values()),
        "global_source_label_weight": sum(source_total.values()),
        "splits": {
            name: {
                "groups": counts[name],
                "windows": windows[name],
                "task_id_count": len(tasks[name]),
                "source_id_count": len(sources[name]),
                "task_label_weight": sum(tasks[name].values()),
                "source_label_weight": sum(sources[name].values()),
                "task_label_distribution_total_variation": _total_variation(tasks[name], task_total),
                "source_label_distribution_total_variation": _total_variation(sources[name], source_total),
                "source_label_weights": dict(sorted(sources[name].items())),
            }
            for name in names
        },
    }


def allocate_groups(
    groups: Iterable[Mapping[str, Any]],
    *,
    seed: int = 42,
    validation_fraction: float = 0.01,
    stages: int = 5,
    task_balance: float = 0.15,
    source_balance: float = 0.15,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return group-to-split assignments and a JSON-serializable audit report.

    All input groups are assigned exactly once. Zero-window groups remain in
    training, with a stable hashed stage, and contribute no sampling weight.
    They do not create a validation sample merely by being assigned a label.
    Input order does not affect the result. Changing group membership or
    weights can change allocations, so persist and version the result.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(stages, bool) or not isinstance(stages, Integral) or stages < 1:
        raise ValueError("stages must be a positive integer")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    for key, value in (("task_balance", task_balance), ("source_balance", source_balance)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    seed, stages = int(seed), int(stages)
    validation_fraction = float(validation_fraction)
    task_balance, source_balance = float(task_balance), float(source_balance)
    prepared = _normalize(groups)
    positive = [group for group in prepared if group.windows]
    names = [f"stage_{i:02d}" for i in range(1, stages + 1)]
    first, validation_report = _allocate_pass(
        positive,
        (("validation", validation_fraction), ("training_pool", 1.0 - validation_fraction)),
        seed=seed, namespace="validation", task_balance=task_balance,
        source_balance=source_balance,
    )
    training = [group for group in positive if first[group.gid] == "training_pool"]
    assigned, stage_report = _allocate_pass(
        training, tuple((name, 1.0 / stages) for name in names),
        seed=seed, namespace="training-stages", task_balance=task_balance,
        source_balance=source_balance,
    )
    assigned.update({gid: "validation" for gid, name in first.items() if name == "validation"})
    for group in prepared:
        if group.windows == 0:
            index = int.from_bytes(_digest(seed, "zero-window", group.gid), "big") % stages
            assigned[group.gid] = names[index]
    if len(assigned) != len(prepared):
        raise RuntimeError("Not all source groups received an assignment")
    warnings = []
    if not any(name == "validation" for name in assigned.values()):
        warnings.append("Validation has no positive-window group; the target may be too small for whole groups.")
    empty = [name for name in names if not any(group.windows and assigned[group.gid] == name for group in training)]
    if empty:
        warnings.append("Training stages without valid windows: " + ", ".join(empty))
    return assigned, {
        "algorithm": ALGORITHM_VERSION,
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "stages": stages,
        "balance_weights": {"total_windows": 1.0, "task_labels": task_balance, "source_labels": source_balance},
        "group_integrity": "Whole caller-provided source group; never split between validation or stages",
        "zero_window_group_count": len(prepared) - len(positive),
        "zero_window_policy": "Assign to a deterministic training stage with zero sampling weight",
        "validation_allocation": validation_report,
        "training_stage_allocation": stage_report,
        "distributions": _distribution_report(prepared, assigned, ["validation", *names]),
        "warnings": warnings,
        "limitations": [
            "Greedy soft balancing is not globally optimal; indivisible groups limit attainable proportions.",
            "Task IDs are identifiers, not semantic task categories; source IDs are not inferred scenes or devices.",
            "No inverse-frequency sampling, task-uniform sampling, or compulsory rare-task coverage is introduced.",
            "Leakage protection is limited to the grouping information supplied by the caller.",
            "Input changes can change assignment; persist manifests rather than rerunning during training.",
            "Stage balance is by valid windows, not wall-clock duration or physical-file size.",
        ],
    }
