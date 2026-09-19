"""Build a metadata-only Roban UMI catalog; never read frame payloads or decode video."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path("/mnt/nas/public/roban_umi/restricted_data/derived/umi_v30_curated_v1")
PRIVATE_OUTPUT_ROOT = REPO.parent / "data_preparation"
CAMERAS = tuple("observation.images." + name for name in
                ("head_left", "head_right", "wrist_left", "wrist_right"))
VERSION = "roban-umi-world-pose-catalog-v1"
CAMERA_TYPE = pa.struct([
    ("camera", pa.string()), ("video_path", pa.string()),
    ("chunk_index", pa.int64()), ("file_index", pa.int64()),
    ("from_timestamp", pa.float64()), ("to_timestamp", pa.float64()),
    ("source_frame_count", pa.int64()),
])
MANIFEST_SCHEMA = pa.schema([
    ("catalog_version", pa.string()), ("metadata_file", pa.string()),
    ("metadata_row", pa.int64()), ("episode_index", pa.int64()), ("length", pa.int64()),
    ("fps", pa.float64()), ("duration_seconds", pa.float64()), ("duration_basis", pa.string()),
    ("dataset_from_index", pa.int64()), ("dataset_to_index", pa.int64()),
    ("data_path", pa.string()), ("data_locator", pa.string()),
    ("data_local_row_from", pa.int64()), ("data_local_row_to", pa.int64()),
    ("task_ids", pa.list_(pa.int64())), ("task_texts", pa.list_(pa.string())),
    ("task_mapping_basis", pa.string()), ("source_mcap", pa.string()),
    ("source_recording_key", pa.string()), ("source_key_basis", pa.string()),
    ("source_path_parent", pa.string()), ("source_set_id", pa.string()),
    ("source_global_episode_index", pa.int64()), ("source_local_episode_index", pa.int64()),
    ("source_trim_frame_start", pa.int64()), ("source_trim_frame_stop", pa.int64()),
    ("scene", pa.string()), ("session_id", pa.string()), ("collector_id", pa.string()),
    ("task_category", pa.string()), ("cameras", pa.list_(CAMERA_TYPE)),
    ("curation_mapping_json", pa.string()), ("merge_mapping_json", pa.string()),
    ("metadata_reconstruction", pa.string()), ("max_required_gap_ns", pa.int64()),
    ("issues", pa.list_(pa.string())), ("warnings", pa.list_(pa.string())),
])
TASK_SCHEMA = pa.schema([("task_index", pa.int64()), ("task", pa.string()),
                         ("issues", pa.list_(pa.string()))])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def text(value):
    return value if isinstance(value, str) and value.strip() else None


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def relative_asset(pattern, **values):
    if not isinstance(pattern, str):
        raise ValueError("Missing path pattern")
    result = pattern.format(**values)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or "\\" in result:
        raise ValueError("Asset path must remain relative to the source dataset")
    return str(path)


def recording_key(path):
    """Use the documented OSS mount mapping; other paths remain explicit raw keys."""
    path = text(path)
    if path is None:
        return None, "missing"
    if path.startswith("/mnt/oss-data/"):
        return "oss://ruoban-pai-wl-oss/" + path[len("/mnt/oss-data/"):], "documented_oss_mount"
    if path.startswith("oss://"):
        return path, "explicit_oss_uri"
    return path, "unmapped_raw_path"


class Parts:
    def __init__(self, directory, schema, part_rows):
        self.directory, self.schema, self.limit = directory, schema, part_rows
        self.directory.mkdir()
        self.rows, self.count = [], 0

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self):
        if self.rows:
            pq.write_table(pa.Table.from_pylist(self.rows, schema=self.schema),
                           self.directory / f"part-{self.count:05d}.parquet",
                           compression="zstd")
            self.rows.clear()
            self.count += 1


def build_catalog(source, output, *, max_meta_files=None, batch_size=4096,
                  part_rows=8192, check_files=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output and public source must be separate directory trees")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing catalog: {output}")
    if batch_size < 1 or part_rows < 1 or (max_meta_files is not None and max_meta_files < 1):
        raise ValueError("Batch, part and optional file limits must be positive")
    info = json.loads((source / "meta/info.json").read_text())
    fps = real(info.get("fps"))
    if fps is None or fps <= 0:
        raise ValueError("info.json must contain a positive finite fps")
    all_paths = sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not all_paths:
        raise FileNotFoundError("No v3 episode metadata files found")
    paths = all_paths if max_meta_files is None else all_paths[:max_meta_files]
    output.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"],
                                             text=True))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    config = {
        "catalog_version": VERSION, "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "output": str(output), "code_commit": commit,
        "working_tree_dirty": dirty, "scanner_sha256": digest(Path(__file__)),
        "data_representation": "world-pose-16D; current state; future actions 1..16",
        "info_sha256": digest(source / "meta/info.json"),
        "source_info": info,
        "options": {"max_meta_files": max_meta_files, "batch_size": batch_size,
                    "part_rows": part_rows, "check_files": bool(check_files)},
        "scope": "Metadata only. No low-dimensional frame payload, MP4 decoding, validity filtering or normalization.",
        "recording_key_policy": "/mnt/oss-data/<key> -> oss://ruoban-pai-wl-oss/<key>; raw value retained",
        "recording_key_evidence": "/mnt/nas/public/roban_umi/UMI_NUMERIC_EPISODE_TO_OSS.md",
    }
    write_json(output / "catalog_config.json", config)
    db = sqlite3.connect(output / "catalog_index.sqlite3")
    db.executescript("""
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-32768;
        CREATE TABLE tasks(task_id INTEGER, task TEXT);
        CREATE INDEX tasks_id ON tasks(task_id);
        CREATE INDEX tasks_text ON tasks(task);
        CREATE TABLE lineage(kind TEXT, episode_id INTEGER, payload TEXT,
                             PRIMARY KEY(kind, episode_id));
        CREATE TABLE episodes(seq INTEGER PRIMARY KEY, episode_id INTEGER, length INTEGER,
            global_from INTEGER, global_to INTEGER, data_path TEXT,
            source_key TEXT, source_parent TEXT, source_set TEXT, issue_count INTEGER);
        CREATE INDEX episode_ids ON episodes(episode_id);
        CREATE INDEX global_ranges ON episodes(global_from);
        CREATE TABLE task_usage(episode_seq INTEGER, task_id INTEGER, length INTEGER);
        CREATE TABLE files(path TEXT PRIMARY KEY, kind TEXT, refs INTEGER, present INTEGER);
    """)
    issues, warnings = Counter(), Counter()
    inputs, schema_types, schema_samples = [], Counter(), {}
    presence = Counter()
    mapping_stats = {}
    total_rows = total_frames = multi_task = 0
    seen_episode_ids = set()  # Episode-level IDs only; never one entry per frame.
    sink = Parts(output / "episode_manifest", MANIFEST_SCHEMA, part_rows)

    def parquet_batches(path):
        # All callers pass metadata files, never a data/ or videos/ payload.
        relative = str(path.relative_to(source))
        if not relative.startswith("meta/"):
            raise ValueError(f"Refusing to read non-metadata parquet: {relative}")
        parquet = pq.ParquetFile(path)
        schema_text = str(parquet.schema_arrow)
        schema_hash = hashlib.sha256(schema_text.encode()).hexdigest()
        inputs.append({"path": relative, "rows": parquet.metadata.num_rows,
                       "bytes": path.stat().st_size, "sha256": digest(path),
                       "schema_sha256": schema_hash})
        schema_types[schema_hash] += 1
        schema_samples.setdefault(schema_hash, {
            "example_file": relative, "fields": [
                {"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow],
            "schema_metadata": {key.decode(errors="replace"): value.decode(errors="replace")
                                for key, value in (parquet.schema_arrow.metadata or {}).items()},
        })
        if relative.startswith("meta/episodes/"):
            presence.update(parquet.schema_arrow.names)
        yield from parquet.iter_batches(batch_size=batch_size)

    def count_row_errors(row_errors):
        issues.update(set(row_errors))

    def register_file(path, kind):
        if path:
            db.execute("""INSERT INTO files VALUES(?,?,1,NULL)
                          ON CONFLICT(path) DO UPDATE SET refs=refs+1""", (path, kind))

    @lru_cache(maxsize=16384)
    def ids_for_text(value):
        return tuple(r[0] for r in db.execute(
            "SELECT DISTINCT task_id FROM tasks WHERE task=? ORDER BY task_id", (value,)))

    @lru_cache(maxsize=16384)
    def texts_for_id(value):
        return tuple(r[0] for r in db.execute("SELECT task FROM tasks WHERE task_id=?", (value,)))

    def mapping(kind, episode_id):
        result = db.execute("SELECT payload FROM lineage WHERE kind=? AND episode_id=?",
                            (kind, episode_id)).fetchone()
        return json.loads(result[0]) if result else {}

    try:
        print("Reading task metadata", flush=True)
        with pq.ParquetWriter(output / "task_catalog.parquet", TASK_SCHEMA, compression="zstd") as writer:
            for batch in parquet_batches(source / "meta/tasks.parquet"):
                normalized = []
                for row in batch.to_pylist():
                    task_id, task = integer(row.get("task_index")), text(row.get("task"))
                    errors = []
                    if task_id is None or task_id < 0:
                        errors.append("invalid_task_id")
                    if task is None:
                        errors.append("missing_task_text")
                    if not errors:
                        previous = db.execute("SELECT task FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                        if previous:
                            errors.append("duplicate_task_id")
                        db.execute("INSERT INTO tasks VALUES(?,?)", (task_id, task))
                    normalized.append({"task_index": task_id, "task": task, "issues": errors})
                    count_row_errors(errors)
                writer.write_table(pa.Table.from_pylist(normalized, schema=TASK_SCHEMA))
        for kind, relative, key in (
            ("merge", "meta/merge/episode_mapping.parquet", "global_episode_index"),
            ("curation", "meta/curation/episode_mapping.parquet", "curated_global_episode_index"),
        ):
            path = source / relative
            mapping_stats[kind] = {"available": path.is_file(), "rows": 0}
            if not path.is_file():
                continue
            print("Reading lineage metadata:", relative, flush=True)
            for batch in parquet_batches(path):
                for row in batch.to_pylist():
                    mapping_stats[kind]["rows"] += 1
                    episode_id = integer(row.get(key))
                    if episode_id is None:
                        issues["invalid_" + kind + "_mapping_id"] += 1
                        continue
                    try:
                        db.execute("INSERT INTO lineage VALUES(?,?,?)",
                                   (kind, episode_id, json.dumps(row, ensure_ascii=False)))
                    except sqlite3.IntegrityError:
                        issues["duplicate_" + kind + "_mapping_id"] += 1
                db.commit()

        for file_index, path in enumerate(paths):
            row_number = 0
            for batch in parquet_batches(path):
                for raw in batch.to_pylist():
                    errors, notes = [], []
                    ep, length = integer(raw.get("episode_index")), integer(raw.get("length"))
                    start, stop = integer(raw.get("dataset_from_index")), integer(raw.get("dataset_to_index"))
                    if ep is None or ep < 0:
                        errors.append("invalid_episode_id")
                    elif ep in seen_episode_ids:
                        errors.append("duplicate_episode_id")
                    if ep is not None:
                        seen_episode_ids.add(ep)
                    if length is None or length <= 0:
                        errors.append("invalid_episode_length")
                    if start is None or stop is None or start < 0 or stop <= start:
                        errors.append("invalid_global_range")
                    elif length != stop - start:
                        errors.append("length_global_span_mismatch")
                    merged = mapping("merge", ep) if ep is not None else {}
                    curated = mapping("curation", ep) if ep is not None else {}
                    for kind, mapping_row in (("merge", merged), ("curation", curated)):
                        if mapping_stats[kind]["available"] and not mapping_row:
                            errors.append("missing_" + kind + "_mapping")
                    for mapping_row, prefix in ((merged, "global"), (curated, "curated_global")):
                        if mapping_row and (mapping_row.get(prefix + "_dataset_from_index"),
                                            mapping_row.get(prefix + "_dataset_to_index")) != (start, stop):
                            errors.append(prefix + "_lineage_range_mismatch")
                    chunk, file_id = integer(raw.get("data/chunk_index")), integer(raw.get("data/file_index"))
                    data_path = None
                    if chunk is None or file_id is None or chunk < 0 or file_id < 0:
                        errors.append("missing_data_file_indices")
                    else:
                        try:
                            data_path = relative_asset(info.get("data_path"), chunk_index=chunk,
                                                       file_index=file_id, episode_index=ep)
                        except (ValueError, KeyError, TypeError, IndexError):
                            errors.append("invalid_data_path_pattern")
                    if curated.get("merged_data_file_path") and curated["merged_data_file_path"] != data_path:
                        errors.append("curation_data_path_mismatch")
                    if merged and (merged.get("global_data_chunk_index"), merged.get("global_data_file_index")) != (chunk, file_id):
                        errors.append("merge_data_indices_mismatch")
                    register_file(data_path, "data")

                    tasks = raw.get("tasks")
                    if not isinstance(tasks, list) or not tasks or any(text(x) is None for x in tasks):
                        errors.append("missing_or_invalid_episode_tasks")
                        tasks = [x for x in tasks if text(x)] if isinstance(tasks, list) else []
                    task_ids, task_basis = [], "exact_text_join"
                    if merged and isinstance(merged.get("global_task_indices"), list):
                        task_basis = "merge_mapping_global_task_indices"
                        for value in merged["global_task_indices"]:
                            task_id = integer(value)
                            matched_texts = texts_for_id(task_id) if task_id is not None else ()
                            if len(matched_texts) != 1:
                                errors.append("unknown_or_ambiguous_task_id")
                            else:
                                task_ids.append(task_id)
                        linked_texts = [texts_for_id(i)[0] for i in task_ids]
                        if set(linked_texts) != set(tasks) or set(merged.get("tasks") or []) != set(tasks):
                            errors.append("task_mapping_text_mismatch")
                    else:
                        for task in tasks:
                            candidates = ids_for_text(task)
                            if len(candidates) != 1:
                                errors.append("unknown_or_ambiguous_task_text")
                            else:
                                task_ids.append(candidates[0])
                    task_ids = sorted(set(task_ids))
                    multi_task += int(len(tasks) > 1)

                    raw_source = text(raw.get("source_mcap"))
                    mapped_source = text(merged.get("source_mcap"))
                    if raw_source and mapped_source and recording_key(raw_source)[0] != recording_key(mapped_source)[0]:
                        errors.append("source_mcap_conflict")
                    source_mcap = raw_source or mapped_source
                    source_key, key_basis = recording_key(source_mcap)
                    if source_key is None:
                        notes.append("missing_source_recording")
                    parent = source_key.rsplit("/", 1)[0] if source_key else None
                    source_set = text(merged.get("set_id")) or text(curated.get("dataset_id"))
                    if merged.get("set_id") and curated.get("dataset_id") and merged["set_id"] != curated["dataset_id"]:
                        errors.append("source_set_mapping_mismatch")
                    trim_start, trim_stop = integer(curated.get("source_trim_frame_start")), integer(curated.get("source_trim_frame_stop"))
                    if curated and (trim_start is None or trim_stop is None or trim_start < 0 or trim_stop - trim_start != length):
                        errors.append("source_trim_length_mismatch")
                    cameras = []
                    for camera in CAMERAS:
                        prefix = "videos/" + camera + "/"
                        video_chunk = integer(raw.get(prefix + "chunk_index"))
                        video_file = integer(raw.get(prefix + "file_index"))
                        begin, end = real(raw.get(prefix + "from_timestamp")), real(raw.get(prefix + "to_timestamp"))
                        video_path = None
                        if video_chunk is None or video_file is None or video_chunk < 0 or video_file < 0:
                            errors.append("missing_video_file_indices:" + camera)
                        else:
                            try:
                                video_path = relative_asset(info.get("video_path"), video_key=camera,
                                    chunk_index=video_chunk, file_index=video_file, episode_index=ep)
                            except (ValueError, KeyError, TypeError, IndexError):
                                errors.append("invalid_video_path_pattern:" + camera)
                        if begin is None or end is None or begin < 0 or end <= begin:
                            errors.append("invalid_video_time_range:" + camera)
                        elif length is not None and abs((end - begin) - length / fps) > 1 / fps + 1e-6:
                            notes.append("video_span_differs_from_length_fps:" + camera)
                        register_file(video_path, "video")
                        cameras.append({"camera": camera, "video_path": video_path,
                            "chunk_index": video_chunk, "file_index": video_file,
                            "from_timestamp": begin, "to_timestamp": end,
                            "source_frame_count": integer(raw.get(prefix + "source_frame_count"))})
                    seq = total_rows
                    count_row_errors(errors)
                    warnings.update(set(notes))
                    db.execute("INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (seq, ep, length, start, stop, data_path, source_key, parent, source_set, len(set(errors))))
                    db.executemany("INSERT INTO task_usage VALUES(?,?,?)",
                                   ((seq, task_id, length) for task_id in task_ids))
                    sink.append({
                        "catalog_version": VERSION, "metadata_file": str(path.relative_to(source)),
                        "metadata_row": row_number, "episode_index": ep, "length": length,
                        "fps": fps, "duration_seconds": length / fps if length is not None and length > 0 else None,
                        "duration_basis": "metadata length / info.fps, not valid-window hours",
                        "dataset_from_index": start, "dataset_to_index": stop,
                        "data_path": data_path, "data_locator": "episode_index_filter",
                        "data_local_row_from": None, "data_local_row_to": None,
                        "task_ids": task_ids, "task_texts": tasks, "task_mapping_basis": task_basis,
                        "source_mcap": source_mcap, "source_recording_key": source_key,
                        "source_key_basis": key_basis, "source_path_parent": parent, "source_set_id": source_set,
                        "source_global_episode_index": integer(curated.get("source_global_episode_index")),
                        "source_local_episode_index": integer(curated.get("source_local_episode_index")),
                        "source_trim_frame_start": trim_start, "source_trim_frame_stop": trim_stop,
                        "scene": text(raw.get("scene")), "session_id": text(raw.get("session_id")),
                        "collector_id": text(raw.get("collector_id")), "task_category": text(raw.get("task_category")),
                        "cameras": cameras,
                        "curation_mapping_json": json.dumps(curated, ensure_ascii=False) if curated else None,
                        "merge_mapping_json": json.dumps(merged, ensure_ascii=False) if merged else None,
                        "metadata_reconstruction": text(raw.get("metadata_reconstruction")),
                        "max_required_gap_ns": integer(raw.get("max_required_gap_ns")),
                        "issues": sorted(set(errors)), "warnings": sorted(set(notes)),
                    })
                    total_rows += 1
                    total_frames += length if length is not None and length > 0 else 0
                    row_number += 1
                db.commit()
            if file_index == 0 or (file_index + 1) % 25 == 0 or file_index + 1 == len(paths):
                print(f"Metadata {file_index + 1}/{len(paths)}; episodes={total_rows}", flush=True)
        sink.flush()

        print("Summarizing episode-level metadata", flush=True)
        duplicate_ids = db.execute("SELECT COUNT(*) FROM (SELECT episode_id FROM episodes WHERE episode_id IS NOT NULL GROUP BY episode_id HAVING COUNT(*)>1)").fetchone()[0]
        previous_stop, gap_count, overlap_count = None, 0, 0
        for start, stop in db.execute("SELECT global_from,global_to FROM episodes WHERE global_from IS NOT NULL AND global_to>global_from ORDER BY global_from,global_to"):
            if previous_stop is not None:
                gap_count += int(start > previous_stop)
                overlap_count += int(start < previous_stop)
            previous_stop = max(previous_stop or 0, stop)
        if overlap_count:
            issues["overlapping_global_ranges"] += overlap_count
        complete = len(paths) == len(all_paths)
        expected_episodes, expected_frames = integer(info.get("total_episodes")), integer(info.get("total_frames"))
        first_start = db.execute("SELECT MIN(global_from) FROM episodes WHERE global_from IS NOT NULL AND global_to>global_from").fetchone()[0]
        if complete:
            if gap_count:
                issues["global_range_gaps"] += gap_count
            if first_start != 0:
                issues["global_range_start_mismatch"] += 1
            if previous_stop != expected_frames:
                issues["global_range_end_mismatch"] += 1
        if complete and expected_episodes != total_rows:
            issues["info_episode_count_mismatch"] += 1
        if complete and expected_frames != total_frames:
            issues["info_frame_count_mismatch"] += 1
        actual_tasks = db.execute("SELECT COUNT(DISTINCT task_id) FROM tasks").fetchone()[0]
        if integer(info.get("total_tasks")) is not None and actual_tasks != info["total_tasks"]:
            issues["info_task_count_mismatch"] += 1
        file_total = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        missing_files = 0
        if check_files:
            print(f"Checking existence of {file_total} unique referenced files (no payload reads)", flush=True)
            cursor = db.execute("SELECT path FROM files")
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                states = [(int((source / relative).is_file()), relative) for (relative,) in batch]
                missing_files += sum(not present for present, _ in states)
                db.executemany("UPDATE files SET present=? WHERE path=?", states)
                db.commit()
            if missing_files:
                issues["missing_referenced_file"] += missing_files

        def export_query(filename, query, schema):
            cursor = db.execute(query)
            names = [item[0] for item in cursor.description]
            with pq.ParquetWriter(output / filename, schema, compression="zstd") as writer:
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    writer.write_table(pa.Table.from_pylist(
                        [dict(zip(names, row)) for row in rows], schema=schema))

        export_query("task_distribution.parquet", """
            SELECT t.task_id AS task_index,MIN(t.task) AS task,
                   COALESCE(u.episode_count,0) AS episode_count,
                   COALESCE(u.episode_frames,0) AS associated_episode_frames
            FROM tasks t LEFT JOIN (
                SELECT task_id,COUNT(*) AS episode_count,SUM(length) AS episode_frames
                FROM task_usage GROUP BY task_id
            ) u ON u.task_id=t.task_id GROUP BY t.task_id
        """, pa.schema([("task_index",pa.int64()),("task",pa.string()),
                         ("episode_count",pa.int64()),("associated_episode_frames",pa.int64())]))
        export_query("source_groups.parquet", """
            SELECT source_key AS source_recording_key,COUNT(*) AS episode_count,
                   SUM(length) AS frames FROM episodes
            WHERE source_key IS NOT NULL GROUP BY source_key
        """, pa.schema([("source_recording_key",pa.string()),("episode_count",pa.int64()),("frames",pa.int64())]))
        export_query("file_inventory.parquet", "SELECT path,kind,refs,present FROM files",
                     pa.schema([("path",pa.string()),("kind",pa.string()),("refs",pa.int64()),("present",pa.int64())]))
        source_groups = db.execute("SELECT COUNT(DISTINCT source_key) FROM episodes").fetchone()[0]
        reused_groups = db.execute("SELECT COUNT(*) FROM (SELECT source_key FROM episodes WHERE source_key IS NOT NULL GROUP BY source_key HAVING COUNT(*)>1)").fetchone()[0]
        missing_sources = db.execute("SELECT COUNT(*) FROM episodes WHERE source_key IS NULL").fetchone()[0]
        top_tasks = [dict(zip(("task_index","episodes"), row)) for row in db.execute(
            "SELECT task_id,COUNT(*) FROM task_usage GROUP BY task_id ORDER BY COUNT(*) DESC LIMIT 20")]
        source_sets = [dict(zip(("source_set_id","episodes","frames"),row)) for row in db.execute(
            "SELECT source_set,COUNT(*),SUM(length) FROM episodes GROUP BY source_set ORDER BY COUNT(*) DESC")]
        required_fields = ["episode_index","tasks","length","dataset_from_index","dataset_to_index",
                           "data/chunk_index","data/file_index"]
        required_fields += ["videos/"+camera+"/"+key for camera in CAMERAS
                            for key in ("chunk_index","file_index","from_timestamp","to_timestamp")]
        schemas = {
            "inputs": inputs,
            "schema_variants": [{**schema_samples[key], "sha256": key, "file_count": count}
                                for key, count in schema_types.items()],
            "episode_field_file_presence": dict(presence),
            "required_episode_fields_missing_in_files": {
                field: len(paths) - presence[field] for field in required_fields if presence[field] < len(paths)},
            "declared_semantic_fields": {field: {
                "metadata_files_with_field": presence[field],
                "inference_from_task_text_or_directory": False,
            } for field in ("scene","session_id","collector_id","task_category")},
            "physical_row_offset_note": "dataset_from/to are global; curated_local_dataset offsets are set-local, not file-local. Select data_path by episode_index until frame-index preparation establishes exact file-local rows.",
        }
        report = {
            "status": "completed_with_issues" if issues else "completed",
            "catalog_version": VERSION, "scan_complete": complete,
            "metadata_files_scanned": len(paths), "metadata_files_available": len(all_paths),
            "total_episodes": total_rows, "total_frames": total_frames,
            "nominal_hours": total_frames / fps / 3600,
            "duration_basis": "metadata frame counts / fps; no validity filtering",
            "info_expected_episodes": expected_episodes, "info_expected_frames": expected_frames,
            "tasks_in_task_catalog": actual_tasks, "episodes_with_multiple_tasks": multi_task,
            "duplicate_episode_ids": duplicate_ids, "global_range_gaps": gap_count,
            "global_range_overlaps": overlap_count,
            "issue_counts": dict(issues), "warning_counts": dict(warnings),
            "manifest_parts": sink.count, "lineage_mappings": mapping_stats,
            "source_recording_groups": source_groups,
            "source_groups_with_multiple_episodes": reused_groups,
            "episodes_missing_source": missing_sources, "source_set_distribution": source_sets,
            "top_tasks_by_episode_count": top_tasks,
            "task_distribution_note": "Multi-task episodes count once for each associated task; associated_episode_frames are not per-task labeled frame counts.",
            "referenced_files": {"unique": file_total, "existence_checked": bool(check_files),
                                 "missing": missing_files if check_files else None,
                                 "content_validated": False},
            "not_established": ["frame/window validity", "scene/session/collector semantics absent from explicit metadata",
                                "content-duplicate or same-session leakage across different source paths",
                                "physical-file-local row offsets", "training/validation split", "normalization"],
            "outputs": ["catalog_config.json","episode_manifest/","task_catalog.parquet",
                        "schema_report.json","catalog_report.json","task_distribution.parquet",
                        "source_groups.parquet","file_inventory.parquet","catalog_index.sqlite3"],
        }
        write_json(output / "schema_report.json", schemas)
        write_json(output / "catalog_report.json", report)
        config["status"] = report["status"]
        write_json(output / "catalog_config.json", config)
        db.commit()
        print(json.dumps({key:report[key] for key in
            ("status","scan_complete","total_episodes","total_frames","nominal_hours","issue_counts")}, ensure_ascii=False), flush=True)
        return report
    except Exception as error:
        sink.flush()
        config["status"] = "failed"
        write_json(output / "catalog_config.json", config)
        write_json(output / "catalog_report.json", {
            "status": "failed", "scan_complete": False, "total_episodes": total_rows,
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-meta-files", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--part-rows", type=int, default=8192)
    parser.add_argument("--check-files", action="store_true", help="Stat each referenced file; never read its payload")
    args = parser.parse_args()
    output = args.output.resolve()
    private_root = PRIVATE_OUTPUT_ROOT.resolve()
    if not output.is_relative_to(private_root) or output == private_root:
        parser.error(f"--output must be a new subdirectory of {private_root}")
    report = build_catalog(args.source, output, max_meta_files=args.max_meta_files,
                          batch_size=args.batch_size, part_rows=args.part_rows,
                          check_files=args.check_files)
    print("Catalog output:", output, flush=True)
    if report["issue_counts"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
