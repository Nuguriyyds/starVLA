"""Build CANDIDATE source-group manifests from verified metadata identifiers.

No numeric payload or video is read. Physical-quality flags are retained, not
converted into new rejection rules. Counts are valid-window weights, not hours.
Task IDs and source IDs do not establish semantic task/scene balance. This tool
does not select the final training split; semantic annotation remains pending.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback

import pyarrow as pa
import pyarrow.parquet as pq

from build_umi_access_index import (
    PRIVATE, fail, integer, now, packed, parquet_rows, progress, sha256,
    verify_inputs, verify_marker, write_json,
)
from umi_split_allocation import allocate_groups
import umi_split_allocation


VERSION = "roban-umi-source-splits-v1"
ASSIGNMENT_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("source_recording_key", pa.string()),
    ("source_set_id", pa.string()), ("split", pa.string()), ("stage", pa.int32()),
    ("valid_windows", pa.int64()), ("window_covered_rows", pa.int64()),
    ("raw_rows", pa.int64()), ("covered_hours", pa.float64()),
    ("task_ids", pa.list_(pa.int64())),
])


def load_episodes(catalog, window, source, manifests, config, report, catalog_report):
    episodes, by_file = {}, defaultdict(dict)
    for name in manifests:
        for row in parquet_rows(catalog / name, [
            "episode_index", "length", "data_path", "task_ids",
            "source_recording_key", "source_set_id",
        ]):
            ep = integer(row["episode_index"], "episode_index")
            length = integer(row["length"], "length")
            if ep < 0 or ep in episodes or length <= 0:
                fail("Invalid/duplicate catalog episode", episode_index=ep)
            # No guessed fallback grouping: missing source identity requires an
            # explicit new policy/version, not silently independent episodes.
            key = row["source_recording_key"]
            if not isinstance(key, str) or not key.strip():
                fail("Missing original recording key", episode_index=ep)
            episodes[ep] = row
            by_file[row["data_path"]][ep] = length
    if len(episodes) != catalog_report["total_episodes"]:
        fail("Catalog episode count mismatch")
    units = {hashlib.sha256(name.encode()).hexdigest()[:24] + ".json" for name in by_file}
    if units != {p.name for p in (window / "scan_progress/completed").glob("*.json")}:
        fail("Frozen completed-marker set differs from catalog")
    if len(by_file) != report["completed_files"] or len(by_file) != report["total_files"]:
        fail("Frozen completed-file count differs from catalog")
    totals, marker_hashes = Counter(), {}
    for number, (data_file, expected) in enumerate(sorted(by_file.items()), 1):
        marker, unit, digest = verify_marker(window, source, data_file, config["rule_fingerprint"])
        marker_hashes[unit + ".json"] = digest
        # Check per-episode weight and de-duplicated covered rows directly
        # against the compact anchor ranges; never multiply windows by horizon.
        ranges = defaultdict(list)
        for row in parquet_rows(window / "valid_anchor_ranges" / (unit + ".parquet")):
            ep = integer(row["episode_index"], "episode_index")
            a = integer(row["anchor_start"], "anchor_start")
            b = integer(row["anchor_end_exclusive"], "anchor_end_exclusive")
            if ep not in expected or not 0 <= a < b <= max(0, expected[ep] - 16):
                fail("Invalid anchor bounds", episode_index=ep)
            if integer(row["num_anchors"], "num_anchors") != b - a:
                fail("Anchor count mismatch", episode_index=ep)
            ranges[ep].append((a, b))
        weights, coverage = Counter(), Counter()
        for ep, intervals in ranges.items():
            previous_end, covered_end = -1, -1
            for a, b in sorted(intervals):
                if a < previous_end:
                    fail("Overlapping anchor ranges", episode_index=ep)
                weights[ep] += b - a
                coverage[ep] += b + 16 - max(a, covered_end)
                previous_end, covered_end = b, b + 16
        seen = set()
        for q in parquet_rows(window / "episode_quality_parts" / (unit + ".parquet"), [
            "episode_index", "data_file", "raw_rows", "valid_windows",
            "window_covered_rows", "task_counts_json",
        ]):
            ep = integer(q["episode_index"], "episode_index")
            if ep not in expected or ep in seen or q["data_file"] != data_file:
                fail("Invalid quality-part identity", episode_index=ep)
            seen.add(ep)
            for field, value in (("raw_rows", expected[ep]), ("valid_windows", weights[ep]),
                                 ("window_covered_rows", coverage[ep])):
                if integer(q[field], field) != value:
                    fail("Episode quality/range disagreement", episode_index=ep, field=field)
            tasks = {}
            for item in json.loads(q["task_counts_json"]):
                task = integer(item["task_index"], "task_index")
                count = integer(item["valid_windows"], "valid_windows")
                if task in tasks or count < 0 or task not in episodes[ep]["task_ids"]:
                    fail("Invalid per-task window weights", episode_index=ep)
                tasks[task] = count
            if sum(tasks.values()) != weights[ep]:
                fail("Task weights do not conserve episode windows", episode_index=ep)
            episodes[ep].update(valid_windows=weights[ep], window_covered_rows=coverage[ep],
                                task_windows={k: v for k, v in tasks.items() if v})
            totals.update(episodes=1, raw_rows=expected[ep], valid_windows=weights[ep],
                          window_covered_rows=coverage[ep])
        if seen != set(expected):
            fail("Quality part does not cover catalog episodes", data_file=data_file)
        if sum(weights.values()) != marker["counts"]["valid_windows"]:
            fail("Marker window total mismatch", data_file=data_file)
        if number % 200 == 0 or number == len(by_file):
            progress("verified_metadata", completed_files=number, total_files=len(by_file))
    for field, value in totals.items():
        if value != report["counts"][field]:
            fail("Frozen report count mismatch", field=field, computed=value)
    return episodes, marker_hashes


def make_groups(episodes):
    grouped = {}
    for row in episodes.values():
        count = row["valid_windows"]
        if not count:
            continue
        key = row["source_recording_key"]
        group = grouped.setdefault(key, {"id": key, "windows": 0, "tasks": Counter(), "sources": Counter()})
        group["windows"] += count
        group["tasks"].update(row["task_windows"])
        # This is a recorded source set, NOT an inferred scene/collector label.
        group["sources"][row["source_set_id"] or "<missing>"] += count
    return list(grouped.values())


def summarize(assignments, episodes, stages, fps, task_catalog):
    labels = ["validation", "train_all"] + [f"stage_{i:02d}" for i in range(1, stages + 1)]
    totals = {label: Counter(episodes=0, valid_windows=0, window_covered_rows=0, raw_rows=0)
              for label in labels + ["excluded_no_valid_windows"]}
    tasks = {label: Counter() for label in labels}
    sources = {label: Counter() for label in labels}
    task_eps = {label: Counter() for label in labels}
    source_eps = {label: Counter() for label in labels}
    group_members = defaultdict(set)
    for row in assignments:
        label, ep = row["split"], row["episode_index"]
        targets = [label, "train_all"] if label.startswith("stage_") else [label]
        if row["valid_windows"]:
            group_members[row["source_recording_key"]].add(label)
        for target in targets:
            totals[target].update(episodes=1, valid_windows=row["valid_windows"],
                                  window_covered_rows=row["window_covered_rows"], raw_rows=row["raw_rows"])
            if target in tasks:
                tasks[target].update(episodes[ep]["task_windows"])
                task_eps[target].update(episodes[ep]["task_windows"].keys())
                key = row["source_set_id"] or "<missing>"
                sources[target][key] += row["valid_windows"]
                source_eps[target][key] += 1
    if any(len(destinations) != 1 for destinations in group_members.values()):
        fail("Original recording group crosses splits")
    eligible = {ep for ep, row in episodes.items() if row["valid_windows"]}
    selected = {row["episode_index"] for row in assignments if row["valid_windows"]}
    if len(assignments) != len(episodes) or len({r["episode_index"] for r in assignments}) != len(episodes) or selected != eligible:
        fail("Split assignment does not conserve episode identities")
    full = sum(row["valid_windows"] for row in episodes.values())
    if totals["train_all"]["valid_windows"] + totals["validation"]["valid_windows"] != full:
        fail("Train and validation do not conserve windows")
    if sum(totals[f"stage_{i:02d}"]["valid_windows"] for i in range(1, stages + 1)) != totals["train_all"]["valid_windows"]:
        fail("Stages do not conserve training windows")
    for label, counts in totals.items():
        if label in tasks and (sum(tasks[label].values()) != counts["valid_windows"] or sum(sources[label].values()) != counts["valid_windows"]):
            fail("Distribution counters do not conserve windows", split=label)
        counts["covered_hours"] = counts["window_covered_rows"] / fps / 3600
        counts["raw_hours"] = counts["raw_rows"] / fps / 3600
        counts["task_ids"] = len(tasks.get(label, {}))
        counts["source_sets"] = len(sources.get(label, {}))
    coverage_rows, coverage_counts = [], Counter(shared=0, train_only=0, validation_only=0, no_valid_windows=0)
    for task in sorted(task_catalog):
        train, validation = tasks["train_all"][task], tasks["validation"][task]
        kind = "shared" if train and validation else "train_only" if train else "validation_only" if validation else "no_valid_windows"
        coverage_counts[kind] += 1
        coverage_rows.append({"task_index": task, "coverage": kind,
                              "train_windows": train, "validation_windows": validation,
                              "train_episodes": task_eps["train_all"][task],
                              "validation_episodes": task_eps["validation"][task]})
    distribution_rows, deviations = [], {}
    for dimension, counters, episode_counters in (("task", tasks, task_eps), ("source_set", sources, source_eps)):
        reference = counters["train_all"]
        denominator = totals["train_all"]["valid_windows"]
        if not denominator:
            fail("Training pool is empty")
        for label in labels:
            count = totals[label]["valid_windows"]
            if not count:
                fail("Split has no valid windows", split=label)
            distance = 0.0
            for key in sorted(set(reference) | set(counters[label]), key=str):
                share, ref_share = counters[label][key] / count, reference[key] / denominator
                distance += abs(share - ref_share)
                if counters[label][key] or reference[key]:
                    distribution_rows.append({"dimension": dimension, "label": str(key), "split": label,
                                              "valid_windows": counters[label][key],
                                              "episodes": episode_counters[label][key],
                                              "window_share": share, "train_reference_share": ref_share})
            deviations.setdefault(label, {})[dimension + "_total_variation_from_train"] = distance / 2
    average = totals["train_all"]["valid_windows"] / stages
    checks = {"unique_episode_assignment": True, "complete_eligible_coverage": True,
              "source_groups_disjoint": True, "train_validation_disjoint": True,
              "stages_disjoint_and_cover_train": True, "window_counts_conserved": True,
              "task_and_source_weights_conserved": True}
    return {
        "counts": {key: dict(value) for key, value in totals.items()}, "checks": checks,
        "eligible_episodes": len(eligible), "eligible_source_groups": len(group_members),
        "valid_windows": full, "actual_validation_fraction": totals["validation"]["valid_windows"] / full,
        "task_coverage": dict(coverage_counts), "distribution_deviation": deviations,
        "stage_max_relative_window_deviation": max(abs(totals[f"stage_{i:02d}"]["valid_windows"] / average - 1) for i in range(1, stages + 1)),
    }, coverage_rows, distribution_rows


def build(args):
    catalog, window, output = [p.expanduser().resolve() for p in (args.catalog, args.window_index, args.output)]
    if output == PRIVATE.resolve() or not output.is_relative_to(PRIVATE.resolve()) or output.exists():
        fail("Output must be a NEW private data_preparation subdirectory", output=str(output))
    for protected in (catalog, window):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            fail("Output overlaps input directory")
    source, info, config, window_report, catalog_report, manifests = verify_inputs(catalog, window)
    if output.is_relative_to(source) or source.is_relative_to(output):
        fail("Output overlaps public source")
    input_paths = {"window_index_config": window / "window_index_config.json",
                   "window_index_report": window / "window_index_report.json",
                   "builder": Path(__file__), "allocator": Path(umi_split_allocation.__file__),
                   "metadata_verifier": Path(__file__).with_name("build_umi_access_index.py")}
    input_hashes = {key: sha256(path) for key, path in input_paths.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".build.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        if output.exists():
            fail("Output appeared while acquiring lock")
        temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".building-", dir=output.parent))
        episodes, marker_hashes = load_episodes(catalog, window, source, manifests, config, window_report, catalog_report)
        progress("allocate", episodes=len(episodes), seed=args.seed, validation_fraction=args.validation_fraction)
        groups = make_groups(episodes)
        allocation, algorithm_report = allocate_groups(groups, seed=args.seed,
                                                       validation_fraction=args.validation_fraction, stages=args.stages)
        if set(allocation) != {group["id"] for group in groups}:
            fail("Allocator does not cover eligible groups")
        allowed = {"validation"} | {f"stage_{i:02d}" for i in range(1, args.stages + 1)}
        if set(allocation.values()) - allowed:
            fail("Allocator returned unknown split")
        rows = []
        for ep, episode in sorted(episodes.items()):
            label = allocation[episode["source_recording_key"]] if episode["valid_windows"] else "excluded_no_valid_windows"
            rows.append({"episode_index": ep, "source_recording_key": episode["source_recording_key"],
                         "source_set_id": episode["source_set_id"], "split": label,
                         "stage": int(label.split("_")[1]) if label.startswith("stage_") else None,
                         "valid_windows": episode["valid_windows"], "window_covered_rows": episode["window_covered_rows"],
                         "raw_rows": episode["length"], "covered_hours": episode["window_covered_rows"] / info["fps"] / 3600,
                         "task_ids": sorted(episode["task_windows"])})
        task_catalog = {r["task_index"] for r in parquet_rows(catalog / "task_catalog.parquet", ["task_index"])}
        summary, coverage, distribution = summarize(rows, episodes, args.stages, info["fps"], task_catalog)
        contract = {"version": VERSION, "rule_fingerprint": config["rule_fingerprint"],
                    "catalog_artifact_sha256": config["contract"]["catalog_artifacts"],
                    "input_sha256": input_hashes, "completed_markers_sha256": marker_hashes,
                    "seed": args.seed, "validation_fraction": args.validation_fraction, "stages": args.stages,
                    "group_key": "source_recording_key", "weight": "valid_windows",
                    "selection_status": "candidate_only",
                    "semantic_balance_assessed": False,
                    "semantic_annotation_status": "pending_user_task_and_scene_labels",
                    "quality_policy": "retain_all_structurally_valid_windows; no new physical filtering",
                    "distribution_target": "natural training-pool window distribution, no task/source resampling"}
        fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write_json(temporary / "split_config.json", {"created_at": now(), "split_fingerprint": fingerprint,
                   "catalog": str(catalog), "window_index": str(window), "contract": contract})
        files = []
        def save(name, selected):
            table = pa.Table.from_pylist(selected, schema=ASSIGNMENT_SCHEMA)
            pq.write_table(table, temporary / name, compression="zstd")
            columns = ["episode_index", "source_recording_key", "split", "valid_windows"]
            if not pq.read_table(temporary / name, columns=columns).equals(table.select(columns)):
                fail("Written manifest failed read-back comparison", artifact=name)
            files.append(name)
        save("episode_assignment.parquet", rows)
        save("train_all_episodes.parquet", [r for r in rows if r["split"].startswith("stage_")])
        for label in ["validation", "excluded_no_valid_windows"] + [f"stage_{i:02d}" for i in range(1, args.stages + 1)]:
            save(label + "_episodes.parquet", [r for r in rows if r["split"] == label])
        for name, records in (("task_coverage.parquet", coverage), ("window_distribution.parquet", distribution)):
            pq.write_table(pa.Table.from_pylist(records), temporary / name, compression="zstd")
            files.append(name)
        summary.update(status="completed", created_at=now(), split_fingerprint=fingerprint,
                       selection_status="candidate_only", semantic_balance_assessed=False,
                       rule_fingerprint=config["rule_fingerprint"], algorithm=algorithm_report,
                       artifacts={name: {"sha256": sha256(temporary / name), "size_bytes": (temporary / name).stat().st_size} for name in files},
                       limits=["Source-key grouping does not detect same-session or content duplicates under different paths.",
                               "source_set_id is a source identifier, not a semantic scene label.",
                               "covered_hours = distinct window-covered rows / recorded FPS / 3600; no multiplication by horizon.",
                               "Task ID coverage is not a semantic generalization classification.",
                               "Pending physical-quality cases remain included; normalization/trainer resume are separate work."])
        summary["checks"]["manifest_readback_matches_assignments"] = True
        write_json(temporary / "split_report.json", summary)
        for key, path in input_paths.items():
            if sha256(path) != input_hashes[key]:
                fail("Input changed during split build", input=key)
        if output.exists():
            fail("Output appeared before publication")
        os.rename(temporary, output)
        temporary = None
        progress("completed", output=str(output), **{k: summary[k] for k in ("eligible_episodes", "valid_windows", "actual_validation_fraction", "stage_max_relative_window_deviation")})
    except BaseException as error:
        if temporary is not None:
            write_json(temporary / "FAILED.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PRIVATE / "roban_umi_world_pose_v1")
    parser.add_argument("--window-index", type=Path, default=PRIVATE / "roban_umi_world_pose_windows_v1")
    parser.add_argument("--output", type=Path, default=PRIVATE / "roban_umi_splits_metadata_candidate_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--stages", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.stages < 1:
        parser.error("Need 0 < validation fraction < 1 and stages >= 1")
    build(args)


if __name__ == "__main__":
    main()
