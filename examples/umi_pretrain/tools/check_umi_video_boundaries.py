"""Offline diagnostics of the unchanged 48-request video-boundary workload.

Errors are records ONLY in this tool. The training Dataset remains fail-fast.
Candidate timestamps describe the current policy; they do not prove ownership.
No model, complete video scan, replacement sampling or index rebuilding is used.
"""
import argparse
from collections import defaultdict, deque
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from starVLA.dataloader.umi_indexed_dataset import CAMERAS, UMIIndexedDataset, collate_umi_samples
from starVLA.dataloader.umi_video_time import VideoTimeWindow, seconds
from starVLA.training.trainer_utils.umi_checkpoint import write_json


class RecordEveryRequest(Dataset):
    """Diagnostic adapter, deliberately not registered with training factories."""
    def __init__(self, raw):
        self.raw = raw

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        result = {"dataset_index": int(index)}
        try:
            episode, offset = self.raw.locate(index)
            result.update(episode_index=episode, episode_row_offset=offset)
            sample = self.raw[index]  # Exercise the actual, unchanged training reader.
            result.update(status="passed", metadata={k: sample["umi_metadata"][k] for k in
                          ("dataset_index", "episode_index", "frame_index", "video_decode")},
                          pixels_sha256=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                                         for im in sample["image"]])
        except Exception as exc:
            result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        return result


def physical_video(root, camera, chunk, file):
    relative = f"videos/{camera}/chunk-{int(chunk):03d}/file-{int(file):03d}.mp4"
    return relative, str((root / relative).resolve())


def collect_memberships(root, targets):
    """One metadata-only pass; retain rows for the requested physical MP4s.

    Filter by camera/chunk/file first, then verify resolved physical paths. The
    two flattened views use global file coordinates; mismatches are reported as
    missing evidence, never silently interpreted as physical neighbours.
    """
    groups = defaultdict(list)
    target_coordinates = defaultdict(set)
    wanted = set()
    for c in targets:
        target_coordinates[c["camera"]].add((c["chunk_index"], c["file_index"]))
        wanted.add(c["physical_video"])
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise ValueError(f"No episode metadata in {root}")
    fields = ["episode_index", "length", "source_mcap", "initial_h264_packets_skipped"]
    fields += [f"videos/{c}/{f}" for c in CAMERAS for f in
               ("chunk_index", "file_index", "from_timestamp", "to_timestamp", "source_frame_count")]
    resolved = {}
    scanned = 0
    for path in files:
        with pq.ParquetFile(path) as reader:
            columns = [f for f in fields if f in reader.schema_arrow.names]
            for batch in reader.iter_batches(batch_size=2048, columns=columns, use_threads=False):
                scanned += batch.num_rows
                table = pa.Table.from_batches([batch])
                keep = pa.array([False] * batch.num_rows)
                for cam, coords in target_coordinates.items():
                    chunk_col, file_col = table[f"videos/{cam}/chunk_index"], table[f"videos/{cam}/file_index"]
                    for chunk in {c for c, _ in coords}:
                        matching = pc.and_(pc.equal(chunk_col, chunk), pc.is_in(file_col,
                            value_set=pa.array([f for c, f in coords if c == chunk], type=file_col.type)))
                        keep = pc.or_(keep, matching)
                for row in table.filter(keep).to_pylist():
                    for cam in CAMERAS:
                        prefix = f"videos/{cam}/"
                        coord = row[prefix + "chunk_index"], row[prefix + "file_index"]
                        if coord not in target_coordinates[cam]:
                            continue
                        key = cam, *coord
                        if key not in resolved:
                            resolved[key] = physical_video(root, cam, *coord)
                        relative, physical = resolved[key]
                        if physical not in wanted:
                            continue
                        groups[physical].append({"episode_index": row["episode_index"],
                            "length": row["length"], "source_mcap": row.get("source_mcap"),
                            "initial_h264_packets_skipped": row.get("initial_h264_packets_skipped"),
                            "metadata_file": str(path.relative_to(root)), "camera": cam,
                            "video_path": relative, "from_timestamp": row[prefix + "from_timestamp"],
                            "to_timestamp": row[prefix + "to_timestamp"],
                            "source_frame_count": row.get(prefix + "source_frame_count")})
    for rows in groups.values():
        rows.sort(key=lambda r: (r["from_timestamp"], r["to_timestamp"], r["episode_index"]))
    return groups, {"root": str(root), "metadata_files": len(files), "metadata_rows": scanned,
                    "matched_physical_files": len(groups), "requested_physical_files": len(wanted)}


def neighbours(rows, episode):
    matches = [i for i, r in enumerate(rows) if r["episode_index"] == episode]
    if len(matches) != 1:
        return {"status": "missing_or_ambiguous", "matches": len(matches)}
    i = matches[0]
    return {"status": "found", "previous_by_start": rows[i-1] if i else None,
            "current": rows[i], "next_by_start": rows[i+1] if i+1 < len(rows) else None,
            "note": "Sorted in the same resolved MP4; metadata intervals are not a PTS ownership oracle."}


def timestamp_at(raw, segments, row):
    for relative, start, _, ep_start, ep_stop in segments:
        if ep_start <= row < ep_stop:
            local = start + row - ep_start
            return float(raw._file_slice(relative, local, local+1)["timestamp"][0].as_py())
    raise ValueError(f"No low-dimensional row {row}")


def candidate_probe(raw, camera, timestamp, count=3):
    """Seek locally; retain both sides independently of the current bounds.

    A bounded decode may not recover every preceding frame (e.g. a broken seek).
    Record that limitation explicitly instead of declaring a new nearest rule.
    """
    import av
    result = {"camera": camera["camera"], "from_timestamp": camera["from_timestamp"],
              "to_timestamp": camera["to_timestamp"], "episode_timestamp": timestamp,
              "physical_video": str(raw._path(camera["video_path"], video=True))}
    try:
        _, selected = raw._video_frame(camera, timestamp)
        result.update(reader_status="passed", selected=selected)
    except Exception as exc:
        result.update(reader_status="failed", reader_error=str(exc))
    try:
        with av.open(result["physical_video"]) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            base = Fraction(stream.time_base)
            window = VideoTimeWindow(camera["from_timestamp"], camera["to_timestamp"], timestamp,
                                     base, raw.decode_tolerance_seconds)
            seek_time = window.target - Fraction(count + 2, 1) / seconds(raw.fps)
            container.seek(max(0, seek_time // base), stream=stream, backward=True, any_frame=False)
            before, after = deque(maxlen=count), []
            reached = False
            decoded = 0
            for frame in container.decode(video=0):
                decoded += 1
                if decoded > 600:  # Bounded local inspection, not full video decoding.
                    break
                if frame.pts is None:
                    continue
                time = window.frame_time(frame.pts, frame.time_base)
                inside = window.contains(time)
                close = abs(time-window.target) <= window.tolerance
                reasons = []
                if not inside:
                    reasons.append("rejected_by_current_metadata_boundary_policy")
                if not close:
                    reasons.append("outside_request_matching_tolerance")
                record = {"pts": frame.pts, "time_base": str(frame.time_base),
                          "time_fraction": str(time), "seconds": float(time),
                          "offset_seconds": float(time-window.target),
                          "inside_strict_metadata_bounds": window.start <= time < window.end,
                          "inside_current_policy_bounds": inside,
                          "within_matching_tolerance": close, "rejection_reasons": reasons,
                          "selected_by_reader": result.get("selected", {}).get("decoded_pts") == frame.pts}
                if time < window.target:
                    before.append(record)
                else:
                    after.append(record)
                    if len(after) == count:
                        reached = True
                        break
            candidates = list(before) + after
            closest = min(candidates, key=lambda r: abs(Fraction(r["time_fraction"])-window.target)) if candidates else None
            result.update(probe_status="recorded", query_fraction=str(window.target),
                          stream_time_base=str(base), stream_frames=stream.frames,
                          decoded_frames=decoded, found_both_sides=bool(before and after),
                          reached_requested_after_count=reached, candidates=candidates,
                          nearest_candidate_ignoring_bounds=closest,
                          warning="Nearest among captured candidates is diagnostic, not approved for training or proof of source ownership.")
    except Exception as exc:
        result.update(probe_status="failed", probe_error=str(exc))
    return result


def upstream_evidence(source, curated, set_ids):
    evidence = []
    paths = [(source / "FLATTEN_COMPLETE.json", ("video_mode", "status")),
             (curated / "FLATTEN_COMPLETE.json", ("video_mode", "status"))]
    for identity in sorted(set_ids):
        paths.extend([
            (source / f"meta/merge/input_artifacts/markers/{identity}/aggregate_complete.json",
             ("timestamp_mode", "video_mode", "set_id", "status")),
            (curated / f"meta/merge/input_artifacts/curation/{identity}/provenance.json",
             ("schema", "source_root", "video_copy_policy", "video_policy", "resolved_config"))])
    for path, keys in paths:
        if not path.is_file():
            evidence.append({"path": str(path), "status": "absent"})
            continue
        content = path.read_bytes()
        obj = json.loads(content)
        selected = {k: obj[k] for k in keys if k in obj}
        if "resolved_config" in selected:
            config = selected["resolved_config"]
            selected["resolved_config"] = {k: config.get(k) for k in ("motion", "materialization", "source")}
        evidence.append({"path": str(path), "status": "read", "sha256": hashlib.sha256(content).hexdigest(),
                         "selected_fields": selected})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Uncurated merged source/umi root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.baseline.read_text())
    provenance = previous["provenance"]
    indices = previous["sequences"]["cross_file_stress"]
    if len(indices) != 48 or len(set(indices)) != 16 or 126293227 not in indices:
        raise ValueError("Require the original 48-request / 16-episode sequence")
    options = {k: provenance[k] for k in ("source_root", "allowed_video_roots", "image_size",
                                         "source_identity", "decode_tolerance_seconds")}
    def make_raw():
        return UMIIndexedDataset(provenance["index_dir"], return_metadata=True, **options)
    report = {"schema": "umi-video-time-diagnosis-v2", "status": "running", "original_indices": indices,
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(), "cases": [],
              "interpretation": "Read success is not proof of correct physical frame ownership."}
    for workers in (0, 2):
        raw = make_raw()
        report["provenance"] = raw.provenance()
        kwargs = dict(batch_size=1, sampler=indices, collate_fn=collate_umi_samples,
                      num_workers=workers, generator=torch.Generator().manual_seed(42))
        if workers:
            kwargs.update(multiprocessing_context="spawn", prefetch_factor=2)
        case = {"workers": workers, "requested": len(indices), "samples": []}
        report["cases"].append(case)
        iterator = None
        try:
            iterator = iter(DataLoader(RecordEveryRequest(raw), **kwargs))
            for position, batch in enumerate(iterator):
                case["samples"].append(dict(batch[0], request_position=position))
                write_json(args.output, report)
                print(f"workers={workers} request={position+1}/48 {batch[0]['status']}", flush=True)
        except Exception as exc:
            case["loader_error"] = str(exc)  # Worker crash is not an ordinary per-request failure.
        finally:
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
            raw.close()
        case["complete_original_sequence"] = [r["dataset_index"] for r in case["samples"]] == indices
        case["passed"] = sum(r["status"] == "passed" for r in case["samples"])
        case["failed"] = sum(r["status"] == "failed" for r in case["samples"])
    report["worker_records_equal"] = report["cases"][0]["samples"] == report["cases"][1]["samples"]
    raw = make_raw()
    raw._worker()
    episodes, targets = [], []
    for index in dict.fromkeys(indices):
        ep, anchor = raw.locate(index)
        metadata, segments = raw._episode(ep)
        episodes.append((index, anchor, metadata, segments))
        for cam in metadata["cameras"]:
            targets.append(dict(cam, physical_video=str(raw._path(cam["video_path"], video=True))))
    print("Reading source/curated EPISODE METADATA ONLY for same-file neighbours", flush=True)
    source_members, source_scan = collect_memberships(args.source_root.resolve(), targets)
    curated_members, curated_scan = collect_memberships(raw.source_root, targets)
    report["metadata_scans"] = [source_scan, curated_scan]
    report["upstream_evidence"] = upstream_evidence(args.source_root.resolve(), raw.source_root,
                                                    {m["source_set_id"] for _, _, m, _ in episodes})
    report["episode_probes"] = []
    for index, anchor, meta, segments in episodes:
        ep = {k: meta[k] for k in ("episode_index", "length", "metadata_file", "source_mcap",
              "source_set_id", "source_global_episode_index", "source_local_episode_index",
              "source_trim_frame_start", "source_trim_frame_stop", "curation_mapping_json")}
        ep.update(dataset_index=index, cameras=[], positions=[])
        for camera in meta["cameras"]:
            physical = str(raw._path(camera["video_path"], video=True))
            source_neighbours = neighbours(source_members.get(physical, []), meta["source_global_episode_index"])
            relation = {"camera": camera["camera"], "curated": camera, "physical_video": physical,
                        "source_neighbours": source_neighbours,
                        "curated_neighbours": neighbours(curated_members.get(physical, []), meta["episode_index"])}
            if source_neighbours["status"] == "found":
                original = source_neighbours["current"]
                relation["mapping_comparison"] = {
                    "same_source_mcap": original["source_mcap"] == meta["source_mcap"],
                    "start_residual_seconds": camera["from_timestamp"] - (original["from_timestamp"] + meta["source_trim_frame_start"]/raw.fps),
                    "end_residual_seconds": camera["to_timestamp"] - (original["from_timestamp"] + meta["source_trim_frame_stop"]/raw.fps),
                    "formula_under_test": "curated from/to = source from + trim start/stop / dataset fps; agreement is observational, not recovered converter code"}
            ep["cameras"].append(relation)
        for label, row in (("original_request", anchor), ("middle", meta["length"]//2), ("last_observation", meta["length"]-1)):
            position = {"label": label, "episode_row": row}
            try:
                timestamp = timestamp_at(raw, segments, row)
                position.update(episode_timestamp=timestamp,
                    cameras=[candidate_probe(raw, c, timestamp) for c in meta["cameras"]])
            except Exception as exc:
                position.update(error=str(exc))
            ep["positions"].append(position)
        report["episode_probes"].append(ep)
        write_json(args.output, report)
        print(f"Probed episode={meta['episode_index']} first/middle/last; all four cameras", flush=True)
    raw.close()
    diagnostic_errors = sum("error" in p or any(c.get("probe_status") != "recorded" for c in p.get("cameras", []))
                            for e in report["episode_probes"] for p in e["positions"])
    reader_probe_failures = sum(c.get("reader_status") == "failed" for e in report["episode_probes"]
                               for p in e["positions"] for c in p.get("cameras", []))
    report["summary"] = {"diagnostic_position_errors": diagnostic_errors,
                         "reader_camera_probe_failures": reader_probe_failures,
                         "all_original_requests_recorded": all(c["complete_original_sequence"] for c in report["cases"]),
                         "training_reader_changed": False, "physical_ownership_verified": False}
    passed = (report["summary"]["all_original_requests_recorded"] and report["worker_records_equal"]
              and not diagnostic_errors and not reader_probe_failures and all(c["failed"] == 0 for c in report["cases"]))
    report["status"] = "passed" if passed else "failed"
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], **report["summary"]}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
