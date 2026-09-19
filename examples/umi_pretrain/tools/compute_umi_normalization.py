"""Fit low-dimensional statistics from one explicit immutable UMI access view.

The scan reads selected Parquet columns in bounded batches, never decodes video
or enumerates windows. A file result is committed atomically, so resume cannot
count a partially processed file twice. Unselected invalid rows are ignored.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from starVLA.dataloader import umi_normalization as normalization
from build_umi_access_index import compute_view_fingerprint

VERSION = "umi-weighted-row-statistics-v1"
COLUMNS = ("episode_index", "frame_index", "timestamp", "task_index")
WEIGHTING = {"state": "one observation per selected anchor",
             "action": "each future offset 1..16 per selected anchor; overlap multiplicity retained"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_path(root, relative):
    item = PurePosixPath(relative)
    if item.is_absolute() or ".." in item.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe source relative path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source file escapes source root: {relative}")
    return path


class AnchorWeights:
    """Compute row frequencies using compact intervals, not expanded windows."""
    def __init__(self, intervals, horizon=16):
        self.intervals = np.asarray(intervals, dtype=np.int64).reshape(-1, 2)
        self.horizon = horizon
        self.starts, self.ends = self.intervals.T
        if (not len(self.intervals) or np.any(self.starts < 0)
                or np.any(self.ends <= self.starts)
                or np.any(self.starts[1:] < self.ends[:-1])):
            raise ValueError("Anchor ranges must be nonempty, sorted and nonoverlapping")
        self.prefix = np.concatenate(([0], np.cumsum(self.ends - self.starts)))

    def before(self, positions):
        positions = np.asarray(positions, dtype=np.int64)
        index = np.searchsorted(self.starts, positions, side="right") - 1
        chosen = np.maximum(index, 0)
        count = self.prefix[chosen] + np.clip(positions - self.starts[chosen], 0,
                                              self.ends[chosen] - self.starts[chosen])
        return np.where(index < 0, 0, count)

    def weights(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        state = self.before(rows + 1) - self.before(rows)
        action = self.before(rows) - self.before(rows - self.horizon)
        return state, action


def checked_source(path, expected):
    stat = path.stat()
    actual = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if actual != expected:
        raise ValueError(f"Source identity changed since access indexing: {path}")
    return actual


def load_view(index_dir, source_root=None):
    index_dir = Path(index_dir).resolve()
    meta = read_json(index_dir / "meta.json")
    if meta.get("status") != "completed" or meta.get("version") != "roban-umi-access-v1":
        raise ValueError("Statistics require a completed roban-umi-access-v1 view")
    normalization.build_representation(meta)
    fingerprint = compute_view_fingerprint(meta)
    if meta.get("view_fingerprint", fingerprint) != fingerprint:
        raise ValueError("View fingerprint disagrees with access metadata")
    for name in ("ranges.npy", "cumulative.npy", "metadata.sqlite3"):
        path = index_dir / name
        artifact = meta["artifacts"][name]
        if path.stat().st_size != artifact["size_bytes"] or file_hash(path) != artifact["sha256"]:
            raise ValueError(f"Access artifact bytes disagree with metadata: {name}")
    ranges = np.load(index_dir / "ranges.npy", mmap_mode="r", allow_pickle=False)
    cumulative = np.load(index_dir / "cumulative.npy", mmap_mode="r", allow_pickle=False)
    if (ranges.dtype != np.dtype("int64") or cumulative.dtype != np.dtype("int64")
            or ranges.shape != (meta["total_ranges"], 3)
            or cumulative.shape != (len(ranges),) or not len(ranges)):
        raise ValueError("Invalid compact access arrays")
    lengths = ranges[:, 2] - ranges[:, 1]
    if (np.any(lengths <= 0) or not np.array_equal(np.cumsum(lengths), cumulative)
            or int(cumulative[-1]) != meta["total_windows"]
            or np.any(ranges[1:, 0] < ranges[:-1, 0])):
        raise ValueError("Invalid compact access counts/order")
    starts = np.r_[0, np.flatnonzero(ranges[1:, 0] != ranges[:-1, 0]) + 1]
    ends = np.r_[starts[1:], len(ranges)]
    anchors = {int(ranges[a, 0]): AnchorWeights(ranges[a:b, 1:]) for a, b in zip(starts, ends)}
    source = Path(source_root or meta["source_path"]).resolve()
    connection = sqlite3.connect((index_dir / "metadata.sqlite3").as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-8192")
    jobs, sources, episode_lengths = defaultdict(list), {}, {}
    try:
        tasks = {int(row[0]) for row in connection.execute("SELECT task_index FROM tasks")}
        for episode, payload in connection.execute("SELECT episode_index,payload FROM episodes"):
            if episode not in anchors:
                continue
            length = int(json.loads(payload)["length"])
            if int(anchors[episode].ends[-1]) - 1 + 16 >= length:
                raise ValueError(f"Incomplete window in episode {episode}")
            episode_lengths[episode] = length
        if set(episode_lengths) != set(anchors):
            raise ValueError("Access ranges reference missing episode metadata")
        expected_offset = defaultdict(int)
        query = ("SELECT episode_index,data_file,file_row_start,file_row_end_exclusive,"
                 "episode_row_offset_start,episode_row_offset_end_exclusive "
                 "FROM segments ORDER BY episode_index,episode_row_offset_start")
        for ep, relative, a, b, ea, eb in connection.execute(query):
            if ep not in anchors:
                continue
            if not (0 <= a < b and ea == expected_offset[ep] and b-a == eb-ea):
                raise ValueError(f"Invalid physical segment map for episode {ep}")
            expected_offset[ep] = eb
            jobs[relative].append({"episode": ep, "file_start": a, "file_end": b,
                                   "episode_start": ea, "episode_end": eb})
        if dict(expected_offset) != episode_lengths:
            raise ValueError("Episode segments do not cover indexed episode lengths")
        for relative, size, mtime in connection.execute("SELECT data_file,size_bytes,mtime_ns FROM data_files"):
            if relative in jobs:
                sources[relative] = {"size_bytes": size, "mtime_ns": mtime}
        if set(sources) != set(jobs):
            raise ValueError("Missing source file identities")
    finally:
        connection.close()
    for relative, segments in jobs.items():
        segments.sort(key=lambda item: item["file_start"])
        if any(a["file_end"] > b["file_start"] for a, b in zip(segments, segments[1:])):
            raise ValueError(f"Overlapping physical segments in {relative}")
        checked_source(safe_path(source, relative), sources[relative])
    return meta, fingerprint, source, anchors, dict(jobs), sources, tasks


def check_edge(previous, current, context):
    if (current[0] != previous[0] + 1 or current[1] != previous[1]):
        raise ValueError(f"Indexed frame/task continuity changed: {context}")


def scan_file(path, segments, anchors, tasks, signals, widths, batch_rows):
    """Stream one physical file; memory is bounded by batch_rows and metadata."""
    state, action = normalization.WeightedMoments(), normalization.WeightedMoments()
    boundaries, last = [], {}
    selected_rows, file_offset = 0, 0
    with pq.ParquetFile(path) as reader:
        schema = reader.schema_arrow
        for key in COLUMNS + tuple(signals):
            if key not in schema.names:
                raise ValueError(f"Missing low-dimensional column {key} in {path}")
        for key in ("episode_index", "frame_index", "task_index"):
            if not pa.types.is_integer(schema.field(key).type):
                raise ValueError(f"Identity column must be integer: {key}")
        if any(s["file_end"] > reader.metadata.num_rows for s in segments):
            raise ValueError(f"Segment outside physical file: {path}")
        segment_cursor = 0
        for batch in reader.iter_batches(batch_size=batch_rows, columns=list(COLUMNS) + list(signals),
                                         use_threads=False):
            batch_end = file_offset + len(batch)
            while segment_cursor < len(segments) and segments[segment_cursor]["file_end"] <= file_offset:
                segment_cursor += 1
            cursor = segment_cursor
            while cursor < len(segments) and segments[cursor]["file_start"] < batch_end:
                segment = segments[cursor]
                cursor += 1
                a, b = max(file_offset, segment["file_start"]), min(batch_end, segment["file_end"])
                if a >= b:
                    continue
                ep = segment["episode"]
                offsets = np.arange(a, b, dtype=np.int64) - segment["file_start"] + segment["episode_start"]
                sw, aw = anchors[ep].weights(offsets)
                chosen = (sw + aw) > 0
                if not chosen.any():
                    continue
                # Filter before converting signals: invalid unselected rows do not participate.
                table = pa.Table.from_batches([batch.slice(a-file_offset, b-a)]).filter(pa.array(chosen))
                offsets, sw, aw = offsets[chosen], sw[chosen], aw[chosen]
                for key in COLUMNS:
                    if table[key].null_count:
                        raise ValueError(f"Null selected identity {key}: {path}, episode={ep}")
                eps, frames, task = (table[key].to_numpy() for key in
                                      ("episode_index", "frame_index", "task_index"))
                times = table["timestamp"].to_numpy()
                if (not np.all(eps == ep) or np.any(frames < 0) or np.any(task < 0)
                        or not np.isfinite(times).all() or not set(task.tolist()).issubset(tasks)):
                    raise ValueError(f"Invalid selected row identity: {path}, episode={ep}")
                segment_id = (ep, segment["episode_start"])
                previous = last.get(segment_id)
                first_record = [int(frames[0]), int(task[0])]
                if aw[0] > 0:
                    if previous is not None and previous[0] == offsets[0] - 1:
                        check_edge(previous[1], first_record, f"{path}:episode={ep},offset={offsets[0]}")
                    elif offsets[0] == segment["episode_start"] and offsets[0] > 0:
                        boundaries.append({"kind": "entry", "episode": ep, "offset": int(offsets[0]),
                                           "identity": first_record})
                    else:
                        raise ValueError(f"Missing selected predecessor: {path}, {ep}, {offsets[0]}")
                required_edges = aw[1:] > 0
                valid_edges = ((np.diff(offsets) == 1) & (np.diff(frames) == 1) & (np.diff(task) == 0))
                if np.any(required_edges & ~valid_edges):
                    raise ValueError(f"Indexed frame/task continuity changed: {path}, episode={ep}")
                last_record = [int(frames[-1]), int(task[-1])]
                if offsets[-1] == segment["episode_end"] - 1:
                    boundaries.append({"kind": "exit", "episode": ep, "offset": int(offsets[-1]),
                                       "identity": last_record})
                last[segment_id] = (int(offsets[-1]), last_record)
                values = []
                for key, width in zip(signals, widths):
                    part = np.asarray(table[key].to_pylist(), dtype=np.float32)
                    if width == 1 and part.shape == (len(table),):
                        part = part[:, None]
                    if part.shape != (len(table), width) or not np.isfinite(part).all():
                        raise ValueError(f"Invalid selected FP32 signal {key}: {path}, episode={ep}")
                    values.append(part)
                values = np.concatenate(values, axis=1)
                state.update(values, sw)
                action.update(values, aw)
                selected_rows += len(values)
            file_offset = batch_end
    return {"state": state.state_dict(), "action": action.state_dict(),
            "boundaries": boundaries, "selected_physical_rows": selected_rows}


def _compute_impl(index_dir, output, *, source_root=None, resume=False, batch_rows=65536,
            min_std=1e-6, purpose="engineering", experiment_contract=None, after_file=None):
    """Compute or resume. after_file is a test hook called after atomic commit."""
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std must be finite and positive")
    if purpose not in ("engineering", "formal"):
        raise ValueError("purpose must be engineering or formal")
    index_dir, output = Path(index_dir).resolve(), Path(output).resolve()
    meta, view, source, anchors, jobs, identities, tasks = load_view(index_dir, source_root)
    if output.is_relative_to(index_dir) or output.is_relative_to(source):
        raise ValueError("Statistics output must not be inside source data or immutable index")
    if output.exists():
        raise FileExistsError(f"Statistics output already exists: {output}")
    code_files = {"scanner": Path(__file__), "normalizer": Path(normalization.__file__),
                  "access_identity": Path(__file__).with_name("build_umi_access_index.py")}
    code = {name: file_hash(path) for name, path in code_files.items()}
    contract = {"version": VERSION, "view_fingerprint": view,
                "meta_sha256": file_hash(index_dir / "meta.json"), "source_identities": identities,
                "source_root": str(source), "index_dir": str(index_dir), "code_sha256": code,
                "batch_rows": batch_rows, "min_std": min_std, "purpose": purpose,
                "experiment_contract": experiment_contract, "weighting": WEIGHTING,
                "input_dtype": "float32", "accumulation_dtype": "float64"}
    contract_hash = digest(contract)
    if purpose == "formal":
        normalization.validate_experiment_contract(experiment_contract, access_meta=meta,
                                                   fit_view_fingerprint=view)
    elif experiment_contract is not None:
        raise ValueError("Engineering statistics must not claim a formal experiment contract")
    work = output.with_name(output.name + ".work")
    if work.exists():
        if not resume:
            raise FileExistsError(f"Progress exists; pass --resume explicitly: {work}")
        old = read_json(work / "contract.json")
        if old != contract:
            raise ValueError("Cannot resume: view/source/code/parameters changed")
    else:
        if resume:
            raise FileNotFoundError(f"No progress to resume: {work}")
        work.mkdir(parents=True)
        atomic_json(work / "contract.json", contract)
    results_dir = work / "completed"
    results_dir.mkdir(exist_ok=True)
    expected_results = {digest(relative) + ".json" for relative in jobs}
    if any(path.name not in expected_results for path in results_dir.glob("*.json")):
        raise ValueError("Unexpected completed-file result in statistics progress")
    for number, relative in enumerate(sorted(jobs), 1):
        result_path = results_dir / (digest(relative) + ".json")
        if result_path.exists():
            continue
        path = safe_path(source, relative)
        checked_source(path, identities[relative])
        result = scan_file(path, jobs[relative], anchors, tasks, meta["signals"],
                           meta["signal_widths"], batch_rows)
        checked_source(path, identities[relative])
        result.update({"contract_fingerprint": contract_hash, "data_file": relative,
                       "source_identity": identities[relative]})
        result["fingerprint"] = digest(result)
        atomic_json(result_path, result)
        print(json.dumps({"completed_file": number, "total_files": len(jobs),
                          "data_file": relative, "purpose": purpose}), flush=True)
        if after_file is not None:
            after_file(relative)
    state, action = normalization.WeightedMoments(), normalization.WeightedMoments()
    exits, entries, physical_rows = {}, [], 0
    for relative in sorted(jobs):
        result = read_json(results_dir / (digest(relative) + ".json"))
        fingerprint = result.pop("fingerprint", None)
        if fingerprint != digest(result):
            raise ValueError(f"Completed-file result content fingerprint mismatch: {relative}")
        if (result["contract_fingerprint"] != contract_hash or result["data_file"] != relative
                or result["source_identity"] != identities[relative]):
            raise ValueError("Completed-file result identity differs from statistics contract")
        state.merge(normalization.WeightedMoments.from_state_dict(result["state"]))
        action.merge(normalization.WeightedMoments.from_state_dict(result["action"]))
        physical_rows += result["selected_physical_rows"]
        for boundary in result["boundaries"]:
            key = (boundary["episode"], boundary["offset"])
            if boundary["kind"] == "exit":
                if key in exits:
                    raise ValueError(f"Duplicate segment boundary: {key}")
                exits[key] = boundary["identity"]
            else:
                entries.append(boundary)
    for boundary in entries:
        key = (boundary["episode"], boundary["offset"] - 1)
        if key not in exits:
            raise ValueError(f"Missing cross-segment predecessor: {key}")
        check_edge(exits[key], boundary["identity"], f"cross-segment episode={key[0]},offset={key[1]+1}")
    if (state.state_dict()["count"] != meta["total_windows"]
            or action.state_dict()["count"] != meta["total_windows"] * 16):
        raise ValueError("Statistics frequencies do not conserve N states and 16N action positions")
    for relative in jobs:
        checked_source(safe_path(source, relative), identities[relative])
    if (file_hash(index_dir / "meta.json") != contract["meta_sha256"]
            or any(file_hash(path) != code[name] for name, path in code_files.items())):
        raise ValueError("Code or view metadata changed during statistics scan")
    artifact = normalization.build_statistics(state, action, access_meta=meta,
        fit_view_fingerprint=view, purpose=purpose,
        fit_details={"index_dir": str(index_dir), "source_root": str(source),
                     "fit_windows": meta["total_windows"], "completed_files": len(jobs),
                     "selected_physical_rows": physical_rows, "weighting": WEIGHTING,
                     "scanner_version": VERSION, "scan_contract_fingerprint": contract_hash,
                     "source_identity_check": "size+mtime_ns; source payload hashes not available",
                     "normalization_policy": "no clipping, quaternion sign changes or physical filtering"},
        code_version=digest(code), min_std=min_std, experiment_contract=experiment_contract)
    atomic_json(output, artifact)
    return artifact


@contextmanager
def exclusive_lock(path):
    """One writer per output; stale locks after SIGKILL require manual inspection."""
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Statistics build lock exists: {path}. Check the recorded PID; "
                           "remove a stale lock only after confirming no writer is active.") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink()


def compute(index_dir, output, **kwargs):
    """Lock the private output before starting or resuming the bounded scan."""
    index_dir, output = Path(index_dir).resolve(), Path(output).resolve()
    meta = read_json(index_dir / "meta.json")
    source = Path(kwargs.get("source_root") or meta["source_path"]).resolve()
    if output.is_relative_to(index_dir) or output.is_relative_to(source):
        raise ValueError("Statistics output must not be inside source data or immutable index")
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output.with_name(output.name + ".lock")):
        return _compute_impl(index_dir, output, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Statistics JSON file; progress uses <output>.work")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument("--min-std", type=float, default=1e-6)
    parser.add_argument("--purpose", choices=("engineering", "formal"), default="engineering")
    parser.add_argument("--experiment-contract", type=Path)
    args = parser.parse_args()
    experiment = read_json(args.experiment_contract) if args.experiment_contract else None
    compute(args.index_dir, args.output, source_root=args.source_root, resume=args.resume,
            batch_rows=args.batch_rows, min_std=args.min_std, purpose=args.purpose,
            experiment_contract=experiment)
    print(f"Statistics saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
