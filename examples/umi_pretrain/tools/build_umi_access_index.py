"""Compile a completed UMI catalog/window index into a compact read-only runtime index.

Only metadata and source file identities are read: no frame payload, video decode,
model, normalization, new quality threshold, or modification of the frozen index.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tempfile
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "roban-umi-access-v1"
REPO = Path(__file__).resolve().parents[3]
PRIVATE = REPO.parent / "data_preparation"
VIDEO_KEYS = ["observation.images." + name for name in
              ("head_left", "head_right", "wrist_left", "wrist_right")]
SEGMENT_KEYS = ("episode_index", "data_file", "file_row_start", "file_row_end_exclusive",
                "episode_row_offset_start", "episode_row_offset_end_exclusive",
                "row_group_start", "row_group_end_inclusive")


def fail(message, **context):
    raise ValueError(message + (" | " + json.dumps(context, ensure_ascii=False) if context else ""))


def now():
    return datetime.now(timezone.utc).isoformat()


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compute_view_fingerprint(meta):
    """Identify the selected window view, independently of its current location.

    Keep this stdlib-only formula identical to umi_indexed_dataset.py. Hashes
    identify the compiled artifacts; this function does not rehash their bytes.
    Source/catalog paths, build times and selection-list filenames are excluded.
    """
    identity = {
        "identity_version": "umi-access-view-identity-v1",
        "version": meta["version"],
        "rule_fingerprint": meta["rule_fingerprint"],
        "catalog_artifacts_sha256": meta["catalog_artifacts_sha256"],
        "artifacts_sha256": {
            name: meta["artifacts"][name]["sha256"]
            for name in ("ranges.npy", "cumulative.npy", "metadata.sqlite3")
        },
        "horizon": meta["horizon"], "fps": meta["fps"],
        "signals": meta["signals"], "signal_widths": meta["signal_widths"],
        "video_keys": meta["video_keys"],
        "representation": meta["representation"],
        "index_order": meta["index_order"],
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def progress(stage, **fields):
    print(packed({"time": now(), "stage": stage, **fields}), flush=True)


def integer(value, field, **context):
    if isinstance(value, bool) or not isinstance(value, int):
        fail("Expected integer", field=field, value=value, **context)
    return value


def relative_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        fail("Invalid relative path", value=relative)
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative:
        fail("Unsafe relative path", value=relative)
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        fail("Relative path escapes its root", root=str(root), value=relative)
    return result


def parquet_rows(path, columns=None):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def episode_selection(path):
    if path is None:
        return None
    schema = pq.ParquetFile(path).schema_arrow
    if "episode_index" not in schema.names or not pa.types.is_integer(schema.field("episode_index").type):
        fail("Episode list requires an integer episode_index column", path=str(path))
    result = set()
    for row in parquet_rows(path, ["episode_index"]):
        ep = integer(row["episode_index"], "episode_index", path=str(path))
        if ep < 0 or ep in result:
            fail("Negative or duplicate episode in selection", episode_index=ep)
        result.add(ep)
    if not result:
        fail("Episode selection is empty", path=str(path))
    return result


def verify_inputs(catalog, window):
    catalog_config = read_json(catalog / "catalog_config.json")
    catalog_report = read_json(catalog / "catalog_report.json")
    config = read_json(window / "window_index_config.json")
    report = read_json(window / "window_index_report.json")
    if catalog_report.get("status") != "completed" or not catalog_report.get("scan_complete") or catalog_report.get("issue_counts"):
        fail("Catalog must be completed without unresolved metadata issues")
    if report.get("status") != "completed" or not report.get("scan_complete") or report.get("failed_files"):
        fail("Window index must be completed without failed files")
    contract = config["contract"]
    fingerprint = hashlib.sha256(packed(contract).encode()).hexdigest()
    if fingerprint != config["rule_fingerprint"] or fingerprint != report["rule_fingerprint"]:
        fail("Frozen window contract fingerprint mismatch")
    if Path(contract["catalog"]).resolve() != catalog:
        fail("Catalog path differs from frozen window contract")
    if integer(contract["horizon"], "horizon") != 16:
        fail("This access version requires horizon=16")
    source = Path(catalog_config["source"]).resolve()
    if Path(contract["source"]).resolve() != source:
        fail("Source path differs between catalog and window contract")
    artifacts = contract["catalog_artifacts"]
    required = {"catalog_config.json", "catalog_report.json", "task_catalog.parquet"}
    actual_manifests = {p.relative_to(catalog).as_posix() for p in (catalog / "episode_manifest").glob("*.parquet")}
    frozen_manifests = {name for name in artifacts if name.startswith("episode_manifest/")}
    if not actual_manifests or actual_manifests != frozen_manifests or not required.issubset(artifacts):
        fail("Catalog artifact list differs from frozen contract")
    for name, expected_hash in artifacts.items():
        if sha256(relative_path(catalog, name)) != expected_hash:
            fail("Catalog artifact hash mismatch", artifact=name)
    info_path = source / "meta/info.json"
    source_info_hash = sha256(info_path)
    if source_info_hash != catalog_config["info_sha256"] or source_info_hash != contract["source_info_sha256"]:
        fail("Source info changed since catalog/window creation")
    info = read_json(info_path)
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0 or fps != float(contract["fps"]):
        fail("Invalid or inconsistent FPS", fps=fps)
    mapped_video = [item["original_key"] for item in contract["mapping"]["video"].values()]
    if mapped_video != VIDEO_KEYS:
        fail("Frozen four-camera order differs from this access contract", video_keys=mapped_video)
    if info.get("numeric_conversion", {}).get("pose_order") != ["x", "y", "z", "qx", "qy", "qz", "qw"]:
        fail("Source pose order differs from world-pose xyzw contract")
    return source, info, config, report, catalog_report, sorted(actual_manifests)


def create_database(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE episodes(episode_index INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE segments(episode_index INTEGER NOT NULL, data_file TEXT NOT NULL,
            file_row_start INTEGER NOT NULL, file_row_end_exclusive INTEGER NOT NULL,
            episode_row_offset_start INTEGER NOT NULL, episode_row_offset_end_exclusive INTEGER NOT NULL,
            row_group_start INTEGER NOT NULL, row_group_end_inclusive INTEGER NOT NULL,
            PRIMARY KEY(episode_index, episode_row_offset_start));
        CREATE TABLE tasks(task_index INTEGER PRIMARY KEY, task TEXT NOT NULL);
        CREATE TABLE data_files(data_file TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL);
    """)
    return connection


def load_catalog(connection, catalog, manifests, selected, fps, catalog_report):
    task_ids = set()
    for row in parquet_rows(catalog / "task_catalog.parquet"):
        task = integer(row["task_index"], "task_index")
        if task < 0 or task in task_ids or row.get("issues") or not isinstance(row["task"], str) or not row["task"].strip():
            fail("Invalid task catalog row", task_index=task)
        task_ids.add(task)
        connection.execute("INSERT INTO tasks VALUES (?,?)", (task, row["task"]))
    episodes, by_file = {}, defaultdict(dict)
    total_rows = selected_count = 0
    for part_number, name in enumerate(manifests, 1):
        for row in parquet_rows(catalog / name):
            ep = integer(row["episode_index"], "episode_index", manifest=name)
            length = integer(row["length"], "length", episode_index=ep)
            data_file = row["data_path"]
            relative_path(catalog, data_file)  # syntax/path traversal validation only
            if ep < 0 or length <= 0 or ep in episodes or row.get("issues"):
                fail("Invalid or duplicate episode catalog row", episode_index=ep, manifest=name)
            if float(row["fps"]) != fps:
                fail("Episode FPS differs from source", episode_index=ep)
            if integer(row["dataset_to_index"], "dataset_to_index") - integer(row["dataset_from_index"], "dataset_from_index") != length:
                fail("Episode length differs from global span", episode_index=ep)
            ids = row["task_ids"]
            if not isinstance(ids, list) or not ids or any(t not in task_ids for t in ids):
                fail("Episode task reference missing from task catalog", episode_index=ep)
            cameras = row["cameras"]
            if not isinstance(cameras, list) or len(cameras) != len(VIDEO_KEYS) or {c.get("camera") for c in cameras} != set(VIDEO_KEYS):
                fail("Episode must have exactly the four expected camera records", episode_index=ep)
            for camera in cameras:
                relative_path(catalog, camera["video_path"])
                begin, end = camera["from_timestamp"], camera["to_timestamp"]
                if begin is None or end is None or not math.isfinite(begin) or not math.isfinite(end) or begin < 0 or end <= begin:
                    fail("Invalid camera time locator", episode_index=ep, camera=camera["camera"])
            keep = selected is None or ep in selected
            episodes[ep] = (length, data_file, keep)
            by_file[data_file][ep] = length
            total_rows += length
            if keep:
                selected_count += 1
                connection.execute("INSERT INTO episodes VALUES (?,?)", (ep, packed(row)))
        connection.commit()
        progress("catalog", completed_parts=part_number, total_parts=len(manifests), episodes=len(episodes), selected_episodes=selected_count)
    if len(episodes) != catalog_report["total_episodes"] or total_rows != catalog_report["total_frames"]:
        fail("Catalog totals differ from manifest", episodes=len(episodes), rows=total_rows)
    if selected is not None and selected - episodes.keys():
        fail("Selected episodes do not exist in catalog", missing=sorted(selected - episodes.keys())[:20])
    return episodes, by_file, selected_count, len(task_ids)


def verify_marker(window, source, data_file, fingerprint):
    unit = hashlib.sha256(data_file.encode()).hexdigest()[:24]
    marker_path = window / "scan_progress/completed" / (unit + ".json")
    marker = read_json(marker_path)
    if marker.get("status") != "completed" or marker.get("rule_fingerprint") != fingerprint or marker.get("data_file") != data_file:
        fail("Invalid completed marker", data_file=data_file, marker=str(marker_path))
    source_stat = relative_path(source, data_file).stat()
    identity = {"size_bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns}
    if marker["source_identity"] != identity:
        fail("Source data file changed since index creation", data_file=data_file)
    expected_names = {folder + "/" + unit + ".parquet" for folder in
                      ("episode_segments", "valid_anchor_ranges", "episode_quality_parts")}
    outputs = marker["outputs"]
    if len(outputs) != 3 or {entry["path"] for entry in outputs} != expected_names:
        fail("Completed marker has unexpected output set", data_file=data_file)
    for entry in outputs:
        if sha256(relative_path(window, entry["path"])) != entry["sha256"]:
            fail("Window part hash mismatch", data_file=data_file, artifact=entry["path"])
    return marker, unit, sha256(marker_path)


def consume_file(connection, window, marker, unit, expected, episodes, horizon):
    data_file = marker["data_file"]
    actual_rows = integer(marker["actual_rows"], "actual_rows", data_file=data_file)
    group_count = integer(marker["row_groups"], "row_groups", data_file=data_file)
    segment_rows = list(parquet_rows(window / "episode_segments" / (unit + ".parquet")))
    offsets, segment_counts = Counter(), Counter()
    file_cursor = 0
    for row in sorted(segment_rows, key=lambda item: item["file_row_start"]):
        ep = integer(row["episode_index"], "episode_index", data_file=data_file)
        if ep not in expected or row["data_file"] != data_file:
            fail("Segment points outside expected episode/file", episode_index=ep, data_file=data_file)
        for key in SEGMENT_KEYS:
            if key != "data_file":
                integer(row[key], key, episode_index=ep, data_file=data_file)
        a, b = row["file_row_start"], row["file_row_end_exclusive"]
        ea, eb = row["episode_row_offset_start"], row["episode_row_offset_end_exclusive"]
        if a != file_cursor or not a < b <= actual_rows or ea != offsets[ep] or eb - ea != b - a or eb > expected[ep]:
            fail("Segment has a gap, overlap, wrong length or bounds", episode_index=ep, data_file=data_file, segment=row)
        if not 0 <= row["row_group_start"] <= row["row_group_end_inclusive"] < group_count:
            fail("Segment row-group range is invalid", episode_index=ep, data_file=data_file)
        offsets[ep] = eb
        segment_counts[ep] += 1
        file_cursor = b
        if episodes[ep][2]:
            connection.execute("INSERT INTO segments VALUES (?,?,?,?,?,?,?,?)", tuple(row[key] for key in SEGMENT_KEYS))
    if dict(offsets) != expected or file_cursor != actual_rows or actual_rows != sum(expected.values()):
        fail("Segments do not cover each episode and physical file exactly", data_file=data_file)
    ranges = list(parquet_rows(window / "valid_anchor_ranges" / (unit + ".parquet")))
    ranges.sort(key=lambda row: (row["episode_index"], row["anchor_start"]))
    last_end, accepted, range_counts = {}, Counter(), Counter()
    kept_ranges = []
    for row in ranges:
        ep = integer(row["episode_index"], "episode_index", data_file=data_file)
        if ep not in expected:
            fail("Range references episode outside file", episode_index=ep, data_file=data_file)
        start = integer(row["anchor_start"], "anchor_start", episode_index=ep)
        end = integer(row["anchor_end_exclusive"], "anchor_end_exclusive", episode_index=ep)
        count = integer(row["num_anchors"], "num_anchors", episode_index=ep)
        if not 0 <= start < end <= max(expected[ep] - horizon, 0) or count != end - start or start < last_end.get(ep, 0):
            fail("Invalid, overlapping or out-of-bounds anchor range", episode_index=ep, data_file=data_file, interval=row)
        last_end[ep] = end
        accepted[ep] += count
        range_counts[ep] += 1
        if episodes[ep][2]:
            kept_ranges.append((ep, start, end))
    qualities = list(parquet_rows(window / "episode_quality_parts" / (unit + ".parquet"), columns=[
        "episode_index", "data_file", "raw_rows", "valid_windows", "anchor_range_count", "segment_count"]))
    seen = set()
    for row in qualities:
        ep = integer(row["episode_index"], "episode_index", data_file=data_file)
        if ep in seen or ep not in expected or row["data_file"] != data_file:
            fail("Invalid episode quality part identity", episode_index=ep, data_file=data_file)
        seen.add(ep)
        for key, value in (("raw_rows", expected[ep]), ("valid_windows", accepted[ep]),
                           ("anchor_range_count", range_counts[ep]), ("segment_count", segment_counts[ep])):
            if integer(row[key], key, episode_index=ep) != value:
                fail("Episode quality counters disagree with ranges/segments", episode_index=ep, data_file=data_file, field=key)
    if seen != set(expected):
        fail("Episode quality part is incomplete", data_file=data_file)
    counts = {"episodes": len(expected), "raw_rows": actual_rows, "valid_windows": sum(accepted.values()),
              "anchor_ranges": len(ranges), "segments": len(segment_rows)}
    for key, value in counts.items():
        if marker["counts"].get(key) != value:
            fail("Completed marker counters disagree with its parts", data_file=data_file, field=key, computed=value)
    if kept_ranges or any(episodes[ep][2] for ep in expected):
        identity = marker["source_identity"]
        connection.execute("INSERT INTO data_files VALUES (?,?,?)", (data_file, identity["size_bytes"], identity["mtime_ns"]))
    return np.asarray(kept_ranges, dtype=np.int64).reshape((-1, 3)), counts


def build(args):
    catalog = args.catalog.expanduser().resolve()
    window = args.window_index.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private = PRIVATE.resolve()
    if output == private or not output.is_relative_to(private):
        fail("Output must be a new private data_preparation subdirectory", output=str(output), private_root=str(private))
    if output.exists():
        fail("Output already exists; choose a new version, never overwrite a runtime index", output=str(output))
    for protected in (catalog, window):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            fail("Output overlaps input tree", output=str(output), input=str(protected))
    progress("verify_inputs", catalog=str(catalog), window_index=str(window))
    source, info, config, report, catalog_report, manifests = verify_inputs(catalog, window)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        fail("Output overlaps source data tree")
    selected = episode_selection(args.episode_list)
    selection_hash = sha256(args.episode_list) if args.episode_list else None
    frozen_input_hashes = {
        "catalog_config.json": config["contract"]["catalog_artifacts"]["catalog_config.json"],
        "catalog_report.json": config["contract"]["catalog_artifacts"]["catalog_report.json"],
        "window_index_config.json": sha256(window / "window_index_config.json"),
        "window_index_report.json": sha256(window / "window_index_report.json"),
        "source_info.json": config["contract"]["source_info_sha256"],
        "builder.py": sha256(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".build.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    connection = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
            lock.write(packed({"pid": os.getpid(), "created_at": now(), "output": str(output)}))
        if output.exists():
            fail("Output appeared while acquiring build lock", output=str(output))
        temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".building-", dir=output.parent))
        database = temporary / "metadata.sqlite3"
        connection = create_database(database)
        episodes, by_file, selected_count, task_count = load_catalog(connection, catalog, manifests, selected, float(info["fps"]), catalog_report)
        marker_names = {hashlib.sha256(name.encode()).hexdigest()[:24] + ".json" for name in by_file}
        actual_names = {path.name for path in (window / "scan_progress/completed").glob("*.json")}
        if actual_names != marker_names or len(by_file) != report["total_files"] or report["completed_files"] != len(by_file):
            fail("Completed marker set or report file count differs from catalog")
        chunks, totals, marker_hashes = [], Counter(), {}
        for number, (data_file, expected) in enumerate(sorted(by_file.items()), 1):
            marker, unit, marker_hash = verify_marker(window, source, data_file, config["rule_fingerprint"])
            chunk, counts = consume_file(connection, window, marker, unit, expected, episodes, 16)
            if len(chunk):
                chunks.append(chunk)
            totals.update(counts)
            marker_hashes[unit + ".json"] = marker_hash
            if number % 100 == 0 or number == len(by_file):
                connection.commit()
                progress("compile", completed_files=number, total_files=len(by_file), verified_windows=totals["valid_windows"])
        for key, value in totals.items():
            if report["counts"].get(key) != value:
                fail("Full window report counter mismatch", field=key, computed=value, reported=report["counts"].get(key))
        if not chunks:
            fail("Selected episodes contain no valid windows")
        ranges = np.concatenate(chunks, axis=0)
        ranges = ranges[np.lexsort((ranges[:, 1], ranges[:, 0]))]
        counts = ranges[:, 2] - ranges[:, 1]
        exact_windows = sum(int(value) for value in counts)
        if exact_windows > np.iinfo(np.int64).max:
            fail("Valid window count exceeds int64 indexing capacity")
        cumulative = np.cumsum(counts, dtype=np.int64)
        np.save(temporary / "ranges.npy", ranges, allow_pickle=False)
        np.save(temporary / "cumulative.npy", cumulative, allow_pickle=False)
        connection.commit()
        database_counts = {table: connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                           for table in ("episodes", "segments", "tasks", "data_files")}
        if database_counts["episodes"] != selected_count or database_counts["tasks"] != task_count:
            fail("Runtime SQLite count mismatch", counts=database_counts)
        progress("verify_database", size_bytes=database.stat().st_size)
        # Sequentially read the committed database before SQLite's page-wise
        # quick_check. On the current network mount, cold 4-KiB page reads were
        # much slower than sequential reads. Keep the integrity check itself.
        sha256(database)
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            fail("Runtime SQLite integrity check failed")
        connection.close()
        connection = None
        outputs = {name: {"sha256": sha256(temporary / name), "size_bytes": (temporary / name).stat().st_size}
                   for name in ("ranges.npy", "cumulative.npy", "metadata.sqlite3")}
        meta = {"version": VERSION, "status": "completed", "created_at": now(),
                "source_path": str(source), "catalog_path": str(catalog), "window_index_path": str(window),
                "rule_fingerprint": config["rule_fingerprint"], "horizon": 16, "fps": float(info["fps"]),
                "video_keys": VIDEO_KEYS, "total_windows": exact_windows, "total_episodes": selected_count,
                "trainable_episodes": int(np.unique(ranges[:, 0]).size), "total_ranges": int(len(ranges)),
                "database_counts": database_counts, "verified_full_index_counts": dict(totals),
                "episode_list": str(args.episode_list.resolve()) if args.episode_list else None,
                "episode_list_sha256": selection_hash, "source_artifact_sha256": frozen_input_hashes,
                "catalog_artifacts_sha256": config["contract"]["catalog_artifacts"],
                "completed_markers_sha256": marker_hashes, "artifacts": outputs,
                "signals": config["contract"]["signals"], "signal_widths": config["contract"]["signal_widths"],
                "representation": config["contract"]["representation"],
                "window_contract": config["contract"],
                "index_order": "episode_index ascending, then episode-local anchor offset ascending",
                "range_semantics": "ranges[i]=[episode_index,start,end_exclusive]; cumulative[i]=number of windows through range i",
                "scope": "Existing structural-validity rules only; no new physical quality threshold or normalization",
                "source_content_hash_verified": False,
                "source_identity_check": "data file size+mtime_ns; this does not establish payload content identity",
                "video_content_validation": "not performed; per-camera locators copied from frozen catalog"}
        meta["view_fingerprint"] = compute_view_fingerprint(meta)
        write_json(temporary / "meta.json", meta)
        # Protect against changed small inputs or selection while compilation was running.
        input_paths = {"catalog_config.json": catalog / "catalog_config.json",
                       "catalog_report.json": catalog / "catalog_report.json",
                       "window_index_config.json": window / "window_index_config.json",
                       "window_index_report.json": window / "window_index_report.json",
                       "source_info.json": source / "meta/info.json", "builder.py": Path(__file__)}
        for name, path in input_paths.items():
            if sha256(path) != frozen_input_hashes[name]:
                fail("Input configuration changed during compilation", artifact=name)
        if args.episode_list and sha256(args.episode_list) != selection_hash:
            fail("Episode selection changed during compilation")
        if output.exists():
            fail("Output appeared before atomic publication", output=str(output))
        os.rename(temporary, output)
        temporary = None
        progress("completed", output=str(output), total_windows=exact_windows, total_ranges=len(ranges), total_episodes=selected_count)
    except BaseException as error:
        if connection is not None:
            connection.close()
        if temporary is not None:
            write_json(temporary / "FAILED.json", {"time": now(), "error": str(error), "traceback": traceback.format_exc()})
            progress("failed", error=str(error), incomplete_output=str(temporary))
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PRIVATE / "roban_umi_world_pose_v1")
    parser.add_argument("--window-index", type=Path, default=PRIVATE / "roban_umi_world_pose_windows_v1")
    parser.add_argument("--output", type=Path, default=PRIVATE / "roban_umi_access_v1")
    parser.add_argument("--episode-list", type=Path, help="Optional Parquet containing unique episode_index values; default selects all episodes")
    args = parser.parse_args()
    try:
        build(args)
    except Exception as error:
        print(packed({"stage": "error", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
