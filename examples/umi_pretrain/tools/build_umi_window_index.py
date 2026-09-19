"""Build a resumable full-dataset low-dimensional UMI window index.

Public inputs are read-only. No model, video decoding, normalization or sampling.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from umi_valid_windows import (SIGNAL_COLUMNS, SIGNAL_WIDTHS, FRAME_VALIDITY,
                              HORIZON, validate_mapping)
from umi_window_rules import evaluate_rows, evaluate_edges

REPO = Path(__file__).resolve().parents[3]
PRIVATE = REPO.parent / "data_preparation"
VERSION = "roban-umi-window-index-v1"
REL_TOL, ABS_TOL = 0.20, 1e-6
EDGE_PRIORITY = ("episode_boundary", "nonconsecutive_frame_index",
                 "timestamp_gap", "source_timestamp_gap", "task_change")
SEGMENT_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("data_file", pa.string()),
    ("file_row_start", pa.int64()), ("file_row_end_exclusive", pa.int64()),
    ("episode_row_offset_start", pa.int64()), ("episode_row_offset_end_exclusive", pa.int64()),
    ("row_group_start", pa.int64()), ("row_group_end_inclusive", pa.int64()),
])
RANGE_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("anchor_start", pa.int64()),
    ("anchor_end_exclusive", pa.int64()), ("num_anchors", pa.int64()),
])
QUALITY_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("data_file", pa.string()), ("raw_rows", pa.int64()),
    ("valid_rows", pa.int64()), ("candidate_windows", pa.int64()),
    ("incomplete_tail_anchors", pa.int64()), ("valid_windows", pa.int64()),
    ("rejected_candidates", pa.int64()), ("window_covered_rows", pa.int64()),
    ("anchor_range_count", pa.int64()), ("segment_count", pa.int64()),
    ("frame_reasons_json", pa.string()), ("edge_reasons_json", pa.string()),
    ("window_reasons_multilabel_json", pa.string()), ("primary_reasons_json", pa.string()),
    ("task_counts_json", pa.string()), ("diagnostics_json", pa.string()),
])
_WORK = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(path, rows, schema):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    os.replace(tmp, path)


def signature(path):
    s = path.stat()
    return {"size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def unit_id(relative):
    return hashlib.sha256(relative.encode()).hexdigest()[:24]


def selected_columns(schema):
    relevant = SIGNAL_COLUMNS + [FRAME_VALIDITY]
    source = ["source_timestamp_ns"] + [key + ".source_timestamp_ns" for key in relevant]
    checked = relevant + ["episode_index", "frame_index", "timestamp", "task_index"] + source
    wanted = ["index", "episode_index", "frame_index", "timestamp", "task_index"] + relevant + source
    wanted += ["schema_presence." + key for key in checked]
    return [name for name in dict.fromkeys(wanted) if name in schema.names]


def runs(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def span_any(mask, width, count):
    prefix = np.r_[np.int64(0), np.cumsum(mask, dtype=np.int64)]
    return (prefix[width:width + count] - prefix[:count]) > 0


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "min": float(values.min()), "max": float(values.max()),
            "sum": float(values.sum()), "sum_squares": float(np.square(values).sum())}


def merge_stats(target, incoming):
    for key, value in incoming.items():
        if not value.get("count"):
            continue
        if key not in target:
            target[key] = dict(value)
        else:
            t = target[key]
            t["count"] += value["count"]
            t["min"], t["max"] = min(t["min"], value["min"]), max(t["max"], value["max"])
            t["sum"] += value["sum"]
            t["sum_squares"] += value["sum_squares"]


def diagnostics(evaluated, row_ok, edges):
    values = evaluated["values"].astype(np.float64)
    diag = {}
    edge_bad = np.zeros(max(len(values) - 1, 0), dtype=bool)
    for bad in edges.values():
        edge_bad |= bad
    adjacent_ok = row_ok[:-1] & row_ok[1:] & ~edge_bad
    for off, label in ((3, "robot1"), (11, "robot2")):
        q = values[:, off:off+4]
        norms = np.linalg.norm(q, axis=1)
        diag[label + ".quaternion_norm"] = stats(norms)
        diag[label + ".quaternion_norm_error"] = stats(abs(norms - 1))
        good = adjacent_ok & (norms[:-1] > 0) & (norms[1:] > 0)
        a, b = q[:-1][good], q[1:][good]
        if len(a):
            dot = np.sum((a / norms[:-1][good, None]) * (b / norms[1:][good, None]), axis=1)
            angles = np.degrees(2 * np.arccos(np.clip(abs(dot), 0, 1)))
            diag[label + ".adjacent_rotation_deg"] = stats(angles)
            diag[label + ".adjacent_quaternion_negative_dot"] = stats((dot < 0).astype(float))
        begin = off - 3
        xyz = values[:, begin:begin+3]
        valid_xyz = np.isfinite(xyz).all(axis=1)
        diag[label + ".position_norm"] = stats(np.linalg.norm(xyz[valid_xyz], axis=1))
        move_ok = adjacent_ok & valid_xyz[:-1] & valid_xyz[1:]
        diag[label + ".adjacent_translation"] = stats(np.linalg.norm(
            xyz[1:][move_ok] - xyz[:-1][move_ok], axis=1))
    diag["robot1.gripper"] = stats(values[:, 7])
    diag["robot2.gripper"] = stats(values[:, 15])
    return diag


def subset_evaluation(evaluated, positions):
    n = len(evaluated["row_ok"])
    result = {}
    for key, value in evaluated.items():
        if key == "row_reasons":
            result[key] = {name: mask[positions].copy() for name, mask in value.items()}
        elif isinstance(value, np.ndarray) and len(value) == n:
            result[key] = value[positions]
        else:
            result[key] = value
    return result


def worker_init(source, output, tasks, fps, fingerprint, max_file_rows, batch_rows):
    _WORK.update(source=Path(source), output=Path(output), tasks=set(tasks), fps=fps,
                 fingerprint=fingerprint, max_file_rows=max_file_rows, batch_rows=batch_rows)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def scan_file(job):
    started = time.monotonic()
    relative, expected = job["data_file"], job["episodes"]
    ident = unit_id(relative)
    source, output = _WORK["source"], _WORK["output"]
    path = source / relative
    before = signature(path)
    parquet = pq.ParquetFile(path)
    if len(parquet.schema_arrow.names) != len(set(parquet.schema_arrow.names)):
        raise ValueError("Duplicate physical column names")
    actual_rows = parquet.metadata.num_rows
    expected_rows = sum(e["length"] for e in expected)
    if actual_rows != expected_rows:
        raise ValueError(f"Catalog/file row mismatch: expected {expected_rows}, actual {actual_rows}")
    if actual_rows > _WORK["max_file_rows"]:
        raise ValueError(f"File exceeds memory guard {_WORK['max_file_rows']} rows; not silently truncated")
    for key in ("episode_index", "index"):
        if key not in parquet.schema_arrow.names:
            raise ValueError(f"Cannot locate rows without {key}")
    columns = selected_columns(parquet.schema_arrow)
    batches = list(parquet.iter_batches(batch_size=_WORK["batch_rows"], columns=columns, use_threads=False))
    table = pa.Table.from_batches(batches)
    evaluated = evaluate_rows(table, _WORK["tasks"])
    if not np.all(evaluated["episode_valid"] & (evaluated["episode"] >= 0)):
        raise ValueError("Invalid episode identities prevent trustworthy physical row mapping")
    global_column = table["index"].combine_chunks()
    if global_column.null_count or not pa.types.is_integer(global_column.type):
        raise ValueError("Global index must contain non-null integers")
    if pa.types.is_unsigned_integer(global_column.type) and np.any(
            global_column.to_numpy(zero_copy_only=False) > np.iinfo(np.int64).max):
        raise ValueError("Global index exceeds signed int64")
    global_index = global_column.to_numpy(zero_copy_only=False).astype(np.int64)
    ep_values = evaluated["episode"]
    expected_by_id = {e["episode_index"]: e for e in expected}
    actual_ids = set(int(x) for x in np.unique(ep_values))
    if actual_ids != set(expected_by_id):
        raise ValueError("Actual episode IDs differ from catalog file assignment; no cross-file guess")
    group_ends = np.cumsum([parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)])
    starts = np.r_[0, np.flatnonzero(ep_values[1:] != ep_values[:-1]) + 1]
    stops = np.r_[starts[1:], actual_rows]
    positions_by_ep = {}
    segment_rows, offsets = [], Counter()
    for a, b in zip(starts, stops):
        a, b = int(a), int(b)
        ep = int(ep_values[a])
        ep_start = offsets[ep]
        segment_rows.append({"episode_index": ep, "data_file": relative,
            "file_row_start": a, "file_row_end_exclusive": b,
            "episode_row_offset_start": ep_start, "episode_row_offset_end_exclusive": ep_start+b-a,
            "row_group_start": int(np.searchsorted(group_ends, a, side="right")),
            "row_group_end_inclusive": int(np.searchsorted(group_ends, b-1, side="right"))})
        offsets[ep] += b-a
        positions_by_ep.setdefault(ep, []).append((a, b))
    quality_rows, anchor_rows = [], []
    summary = Counter()
    frame_counts, edge_counts, multi_counts, primary_counts = Counter(), Counter(), Counter(), Counter()
    combined_diag = {}
    for ep, spans in positions_by_ep.items():
        positions = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in spans])
        spec = expected_by_id[ep]
        n = len(positions)
        if n != spec["length"]:
            raise ValueError(f"Episode {ep}: actual/catalog length mismatch")
        if not np.array_equal(global_index[positions], np.arange(spec["global_from"], spec["global_to"], dtype=np.int64)):
            raise ValueError(f"Episode {ep}: global index sequence differs from catalog range")
        ev = subset_evaluation(evaluated, positions)
        # A malformed/unknown task remains an invalid row. Known but undeclared task is a catalog mismatch.
        known_task = ev["task_valid"] & ~ev["row_reasons"]["missing_or_empty_task"]
        if np.any(known_task & ~np.isin(ev["task"], spec["task_ids"])):
            raise ValueError(f"Episode {ep}: observed task absent from declared episode tasks")
        valid_identity = ev["frame_valid"]
        _, inverse, counts = np.unique(ev["frame"][valid_identity], return_inverse=True, return_counts=True)
        duplicate = np.zeros(n, dtype=bool)
        duplicate[valid_identity] = counts[inverse] > 1
        if duplicate.any():
            ev["row_reasons"]["duplicate_episode_frame"] = duplicate
        row_ok = ev["row_ok"].copy() & ~duplicate
        ev["row_ok"] = row_ok
        edges = evaluate_edges(ev, _WORK["fps"],
                               timestamp_relative_tolerance=REL_TOL,
                               timestamp_absolute_tolerance=ABS_TOL)
        candidate = max(n - HORIZON, 0)
        bad_windows = {"invalid_frame_in_window": span_any(~row_ok, HORIZON+1, candidate)}
        bad_windows.update({reason: span_any(mask, HORIZON, candidate) for reason, mask in edges.items()})
        okay = np.ones(candidate, dtype=bool)
        primary = Counter()
        for reason in ("invalid_frame_in_window",) + EDGE_PRIORITY:
            mask = bad_windows.get(reason)
            if mask is None:
                continue
            primary[reason] += int(np.count_nonzero(okay & mask))
            okay &= ~mask
        unknown_edge = set(edges) - set(EDGE_PRIORITY)
        if unknown_edge:
            raise ValueError(f"Unaccounted edge rejection rules: {unknown_edge}")
        intervals = runs(okay)
        coverage = 0
        covered_end = -1
        for a, b in intervals:
            anchor_rows.append({"episode_index": ep, "anchor_start": a,
                "anchor_end_exclusive": b, "num_anchors": b-a})
            cover_end = b + HORIZON
            coverage += max(cover_end - max(a, covered_end), 0)
            covered_end = max(covered_end, cover_end)
        accepted = int(np.count_nonzero(okay))
        rejected = candidate - accepted
        if sum(primary.values()) != rejected:
            raise RuntimeError("Non-exclusive rejection accounting")
        frame = {k:int(np.count_nonzero(v)) for k,v in ev["row_reasons"].items() if np.any(v)}
        edge = {k:int(np.count_nonzero(v)) for k,v in edges.items() if np.any(v)}
        multi = {k:int(np.count_nonzero(v)) for k,v in bad_windows.items() if np.any(v)}
        primary = {k:v for k,v in primary.items() if v}
        task_stats = []
        for task in np.unique(ev["task"][ev["task_valid"]]):
            task_stats.append({"task_index": int(task),
                "raw_rows": int(np.count_nonzero(ev["task_valid"] & (ev["task"] == task))),
                "valid_rows": int(np.count_nonzero(row_ok & (ev["task"] == task))),
                "valid_windows": int(np.count_nonzero(okay & (ev["task"][:candidate] == task)))})
        diag = diagnostics(ev, row_ok, edges)
        quality_rows.append({"episode_index": ep, "data_file": relative, "raw_rows": n,
            "valid_rows": int(row_ok.sum()), "candidate_windows": candidate,
            "incomplete_tail_anchors": n-candidate, "valid_windows": accepted,
            "rejected_candidates": rejected, "window_covered_rows": coverage,
            "anchor_range_count": len(intervals), "segment_count": len(spans),
            "frame_reasons_json": packed(frame), "edge_reasons_json": packed(edge),
            "window_reasons_multilabel_json": packed(multi), "primary_reasons_json": packed(primary),
            "task_counts_json": packed(task_stats), "diagnostics_json": packed(diag)})
        summary.update(episodes=1, raw_rows=n, valid_rows=int(row_ok.sum()),
                       candidate_windows=candidate, incomplete_tail_anchors=n-candidate,
                       valid_windows=accepted, rejected_candidates=rejected,
                       window_covered_rows=coverage, anchor_ranges=len(intervals), segments=len(spans))
        frame_counts.update(frame)
        edge_counts.update(edge)
        multi_counts.update(multi)
        primary_counts.update(primary)
        merge_stats(combined_diag, diag)
    if signature(path) != before:
        raise ValueError("Source file changed during scan; results not committed")
    output_files = []
    for folder, rows, schema in (
        ("episode_segments", segment_rows, SEGMENT_SCHEMA),
        ("valid_anchor_ranges", anchor_rows, RANGE_SCHEMA),
        ("episode_quality_parts", quality_rows, QUALITY_SCHEMA),
    ):
        dest = output / folder / (ident + ".parquet")
        atomic_parquet(dest, rows, schema)
        output_files.append({"path": str(dest.relative_to(output)), "sha256": digest(dest)})
    marker = {"status": "completed", "rule_fingerprint": _WORK["fingerprint"],
        "data_file": relative, "source_identity": before, "source_identity_kind": "path+size+mtime_ns",
        "actual_rows": actual_rows, "row_groups": parquet.num_row_groups,
        "schema_sha256": hashlib.sha256(str(parquet.schema_arrow).encode()).hexdigest(),
        "selected_columns": columns, "source_timestamp_columns": ev.get("source_columns", []),
        "missing_source_timestamp_columns": ev.get("missing_source_columns", []),
        "outputs": output_files, "counts": dict(summary),
        "frame_reasons": dict(frame_counts), "edge_reasons": dict(edge_counts),
        "window_reasons_multilabel": dict(multi_counts), "primary_reasons": dict(primary_counts),
        "diagnostics": combined_diag, "seconds": time.monotonic()-started, "completed_at": now()}
    atomic_json(output / "scan_progress" / "completed" / (ident + ".json"), marker)
    return marker


def load_plan(catalog):
    report = json.loads((catalog / "catalog_report.json").read_text())
    cfg = json.loads((catalog / "catalog_config.json").read_text())
    if not report.get("scan_complete") or report.get("status") != "completed" or report.get("issue_counts"):
        raise ValueError("Require a complete catalog without unresolved metadata issues")
    source = Path(cfg["source"]).resolve()
    info = json.loads((source / "meta/info.json").read_text())
    if digest(source / "meta/info.json") != cfg["info_sha256"]:
        raise ValueError("Source info changed since catalog creation")
    if info.get("numeric_conversion", {}).get("pose_order") != ["x","y","z","qx","qy","qz","qw"]:
        raise ValueError("Pose order is not the established xyzw world-pose contract")
    mapping_path = REPO / "examples/umi_pretrain/train_files/modality.json"
    mapping = json.loads(mapping_path.read_text())
    validate_mapping(mapping)
    manifest_files = sorted((catalog / "episode_manifest").glob("*.parquet"))
    if not manifest_files:
        raise ValueError("No manifest partitions")
    jobs_by_path = {}
    identities = set()
    for part in manifest_files:
        for batch in pq.ParquetFile(part).iter_batches(batch_size=4096,
                columns=["episode_index","length","dataset_from_index","dataset_to_index","data_path","task_ids","issues"]):
            for row in batch.to_pylist():
                ep = row["episode_index"]
                if row["issues"] or ep in identities:
                    raise ValueError("Catalog contains invalid/duplicate episode")
                identities.add(ep)
                relative = row["data_path"]
                rp = Path(relative)
                if rp.is_absolute() or ".." in rp.parts:
                    raise ValueError("Unsafe relative payload path")
                jobs_by_path.setdefault(relative, []).append({
                    "episode_index": ep, "length": row["length"],
                    "global_from": row["dataset_from_index"], "global_to": row["dataset_to_index"],
                    "task_ids": row["task_ids"]})
    tasks = set()
    for batch in pq.ParquetFile(catalog / "task_catalog.parquet").iter_batches():
        for row in batch.to_pylist():
            if row["issues"] or not isinstance(row["task"],str) or not row["task"].strip():
                raise ValueError("Task catalog has unresolved labels")
            if row["task_index"] in tasks:
                raise ValueError("Duplicate task identity")
            tasks.add(row["task_index"])
    if len(identities) != report["total_episodes"] or sum(e["length"] for es in jobs_by_path.values() for e in es) != report["total_frames"]:
        raise ValueError("Manifest totals differ from catalog report")
    scripts = [Path(__file__), Path(__file__).with_name("umi_window_rules.py"),
               Path(__file__).with_name("umi_valid_windows.py")]
    contract = {"index_version": VERSION, "catalog": str(catalog), "source": str(source),
        "catalog_artifacts": {str(p.relative_to(catalog)):digest(p) for p in
            [catalog/"catalog_config.json",catalog/"catalog_report.json",catalog/"task_catalog.parquet"]+manifest_files},
        "code_sha256": {p.name:digest(p) for p in scripts}, "mapping": mapping,
        "source_info_sha256": cfg["info_sha256"], "horizon": HORIZON, "fps": info["fps"],
        "timestamp_relative_tolerance": REL_TOL, "timestamp_absolute_tolerance": ABS_TOL,
        "source_timestamp_sentinels": [-9223372036854775808,-1,0],
        "signals": SIGNAL_COLUMNS, "signal_widths": SIGNAL_WIDTHS,
        "row_order": "original physical order within each episode, never sort by frame_index",
        "representation": "robot1 xyz+xyzw+gripper then robot2; raw float32; no normalization",
        "video_validation": "none; low-dimensional candidate windows only",
        "source_gap_ns_limit": None, "sensor_timestamp_cadence_constraint": None,
        "file_assignment": "verified against catalog; inconsistent/cross-file assignments are errors",
        "primary_rejection_order": ["invalid_frame_in_window"]+list(EDGE_PRIORITY)}
    fingerprint = hashlib.sha256(packed(contract).encode()).hexdigest()
    jobs = [{"data_file":path,"episodes":es} for path,es in sorted(jobs_by_path.items())]
    return source, info, tasks, jobs, contract, fingerprint


def verified_completed(job, output, source, fingerprint):
    path = output / "scan_progress/completed" / (unit_id(job["data_file"]) + ".json")
    if not path.exists():
        return None
    marker = json.loads(path.read_text())
    if marker["rule_fingerprint"] != fingerprint or marker["data_file"] != job["data_file"]:
        raise ValueError("Completed unit belongs to another input/rule version")
    if marker["source_identity"] != signature(source/job["data_file"]):
        raise ValueError("Completed source file changed; use a new output version")
    for item in marker["outputs"]:
        p = output / item["path"]
        if not p.is_file() or digest(p) != item["sha256"]:
            raise ValueError("Completed output missing/corrupt; refusing silent resume")
    return marker


def aggregate(markers):
    counts, frames, edges, multi, primary = Counter(), Counter(), Counter(), Counter(), Counter()
    diag = {}
    for m in markers.values():
        counts.update(m["counts"])
        frames.update(m["frame_reasons"])
        edges.update(m["edge_reasons"])
        multi.update(m["window_reasons_multilabel"])
        primary.update(m["primary_reasons"])
        merge_stats(diag, m["diagnostics"])
    for v in diag.values():
        mean = v["sum"]/v["count"]
        v["mean"] = mean
        v["std"] = math.sqrt(max(v["sum_squares"]/v["count"]-mean*mean,0))
    return {"counts":dict(counts), "frame_reasons":dict(frames), "edge_reasons":dict(edges),
            "window_reasons_multilabel":dict(multi), "primary_reasons":dict(primary), "diagnostics":diag}


def finalize_quality(output, jobs):
    temp = output / "episode_quality.parquet.tmp"
    task_counts = {}
    with pq.ParquetWriter(temp, QUALITY_SCHEMA, compression="zstd") as writer:
        for job in jobs:
            p = output / "episode_quality_parts" / (unit_id(job["data_file"])+".parquet")
            for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
                writer.write_batch(batch)
                for value in batch.column(batch.schema.get_field_index("task_counts_json")).to_pylist():
                    for item in json.loads(value):
                        task = item["task_index"]
                        t = task_counts.setdefault(task, Counter())
                        t.update({key:item[key] for key in ("raw_rows","valid_rows","valid_windows")})
    os.replace(temp, output/"episode_quality.parquet")
    schema = pa.schema([("task_index",pa.int64()),("raw_rows",pa.int64()),
                        ("valid_rows",pa.int64()),("valid_windows",pa.int64())])
    atomic_parquet(output/"task_window_distribution.parquet",
                   [{"task_index":k,**dict(v)} for k,v in sorted(task_counts.items())], schema)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PRIVATE/"roban_umi_world_pose_v1")
    parser.add_argument("--output", type=Path, default=PRIVATE/"roban_umi_world_pose_windows_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-rows",type=int,default=32768)
    parser.add_argument("--max-file-rows",type=int,default=1000000)
    args = parser.parse_args()
    if min(args.workers,args.batch_rows,args.max_file_rows) < 1:
        parser.error("Resource limits must be positive")
    output, catalog = args.output.resolve(), args.catalog.resolve()
    if output == PRIVATE.resolve() or not output.is_relative_to(PRIVATE.resolve()):
        parser.error("--output must be a private data_preparation subdirectory")
    if output == catalog or output.is_relative_to(catalog) or catalog.is_relative_to(output):
        parser.error("Index and catalog must have separate output trees")
    print("Loading fixed metadata catalog and rules", flush=True)
    source, info, tasks, jobs, contract, fingerprint = load_plan(catalog)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Source and output trees must be separate")
    if output.exists() and not args.resume:
        parser.error("Output exists; use --resume or a new output version")
    if not output.exists() and args.resume:
        parser.error("--resume requires an existing index output")
    output.mkdir(parents=True,exist_ok=True)
    lock_path = output/"scan_progress"/"scan.lock"
    (output/"scan_progress").mkdir(exist_ok=True)
    import fcntl
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("This index output already has an active scanner")
    cfg_path = output/"window_index_config.json"
    if args.resume:
        old = json.loads(cfg_path.read_text())
        if old["rule_fingerprint"] != fingerprint:
            raise ValueError("Rule, code or catalog changed; resume requires identical version")
    else:
        try:
            commit = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        except (OSError,subprocess.SubprocessError):
            commit = None
        atomic_json(cfg_path, {"created_at":now(),"rule_fingerprint":fingerprint,
            "contract":contract,"code_commit":commit,
            "source_identity_check":"size+mtime_ns before/after scan and resume; not a payload content hash"})
    for folder in ("episode_segments","valid_anchor_ranges","episode_quality_parts",
                   "scan_progress/completed","scan_progress/failed"):
        (output/folder).mkdir(parents=True,exist_ok=True)
    markers, pending = {}, []
    for job in jobs:
        marker = verified_completed(job,output,source,fingerprint)
        if marker is None:
            pending.append(job)
        else:
            markers[job["data_file"]] = marker
    invocation_start = time.monotonic()
    started_rows = sum(m["counts"]["raw_rows"] for m in markers.values())
    failed = {}
    stop = {"requested":False}
    def handle_stop(signum,frame):
        stop["requested"] = True
        print("Stop requested; committing active files before exit",flush=True)
    signal.signal(signal.SIGTERM,handle_stop)
    signal.signal(signal.SIGINT,handle_stop)
    def progress(status):
        count = Counter()
        for m in markers.values():
            count.update(m["counts"])
        elapsed = time.monotonic()-invocation_start
        rows_per_second = (count["raw_rows"]-started_rows)/max(elapsed,1e-6)
        value = {"status":status,"updated_at":now(),"pid":os.getpid(),
            "rule_fingerprint":fingerprint,"total_files":len(jobs),"completed_files":len(markers),
            "failed_files":len(failed),"remaining_files":len(jobs)-len(markers),
            "counts":dict(count),"invocation_seconds":elapsed,
            "invocation_rows_per_second":rows_per_second,
            "workers":args.workers,"batch_rows":args.batch_rows,"max_file_rows":args.max_file_rows}
        atomic_json(output/"scan_progress/progress.json",value)
        return value
    progress("running")
    print(f"Full scan: {len(jobs)} files; {len(markers)} already committed; {len(pending)} pending",flush=True)
    # Only workers in flight are queued; no unbounded future/task payload accumulation.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,
            initializer=worker_init,initargs=(str(source),str(output),sorted(tasks),info["fps"],
                                             fingerprint,args.max_file_rows,args.batch_rows)) as pool:
        iterator = iter(pending)
        active = {}
        def submit_one():
            if stop["requested"]:
                return
            job = next(iterator,None)
            if job is not None:
                active[pool.submit(scan_file,job)] = job
        for _ in range(args.workers):
            submit_one()
        since_print = time.monotonic()
        while active:
            done,_ = wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    marker = future.result()
                    markers[job["data_file"]] = marker
                except Exception as error:
                    record = {"status":"failed","data_file":job["data_file"],
                        "rule_fingerprint":fingerprint,"error":f"{type(error).__name__}: {error}",
                        "traceback":traceback.format_exc(),"at":now()}
                    failed[job["data_file"]] = record
                    atomic_json(output/"scan_progress/failed"/(unit_id(job["data_file"])+".json"),record)
                    print(f"FAILED {job['data_file']}: {record['error']}",flush=True)
                submit_one()
            if done or time.monotonic()-since_print >= 30:
                p = progress("stopping" if stop["requested"] else "running")
                if len(markers) % 25 == 0 or time.monotonic()-since_print >= 30 or failed:
                    print(f"files={len(markers)}/{len(jobs)} failed={len(failed)} "
                          f"rows={p['counts'].get('raw_rows',0)} valid_windows={p['counts'].get('valid_windows',0)} "
                          f"rows/s={p['invocation_rows_per_second']:.0f}",flush=True)
                    since_print = time.monotonic()
    complete = len(markers) == len(jobs) and not failed
    if complete:
        print("Writing consolidated episode quality and filtered task distribution",flush=True)
        finalize_quality(output,jobs)
    result = aggregate(markers)
    counts = result["counts"]
    if complete:
        assert counts["episodes"] == info["total_episodes"]
        assert counts["raw_rows"] == info["total_frames"]
        assert counts["candidate_windows"] == counts["valid_windows"]+counts["rejected_candidates"]
        assert counts["raw_rows"] == counts["candidate_windows"]+counts["incomplete_tail_anchors"]
    result.update(status="completed" if complete else ("paused" if stop["requested"] else "incomplete"),
        scan_complete=complete, rule_fingerprint=fingerprint, updated_at=now(),
        total_files=len(jobs),completed_files=len(markers),failed_files=list(failed.values()),
        nominal_raw_hours=counts.get("raw_rows",0)/info["fps"]/3600,
        nominal_valid_row_hours=counts.get("valid_rows",0)/info["fps"]/3600,
        unique_window_covered_hours=counts.get("window_covered_rows",0)/info["fps"]/3600,
        scope="low-dimensional candidate windows; video content and synchronization not validated",
        source_content_hash_verified=False,
        note="Multi-label reasons overlap; primary candidate rejection counts are exclusive. Tail anchors are separate.",
        outputs=["window_index_config.json","episode_segments/","valid_anchor_ranges/",
                 "episode_quality_parts/","episode_quality.parquet","task_window_distribution.parquet",
                 "window_index_report.json","scan_progress/"])
    atomic_json(output/"window_index_report.json",result)
    progress(result["status"])
    print(json.dumps({k:result[k] for k in ("status","scan_complete","counts","nominal_raw_hours",
                                          "nominal_valid_row_hours","unique_window_covered_hours")}),flush=True)
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
