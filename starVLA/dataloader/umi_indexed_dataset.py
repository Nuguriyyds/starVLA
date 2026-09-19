"""Read immutable UMI window indexes without enumerating individual windows.

The raw Dataset performs no normalization, relabeling, padding, replacement
sampling, or quality filtering. The factory can add an explicit normalization
wrapper; read_lowdim() on the raw Dataset always retains its original meaning.
The access index is built offline by build_umi_access_index.py.
"""
from collections import OrderedDict
import hashlib
import json
import math
import operator
import os
from pathlib import Path, PurePosixPath
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset


CAMERAS = tuple("observation.images." + name for name in
                ("head_left", "head_right", "wrist_left", "wrist_right"))
SIGNALS = (
    "observation.umi.robot1_finger_eef_pose",
    "observation.umi.robot1_sensor_magnetic_encoder",
    "observation.umi.robot2_finger_eef_pose",
    "observation.umi.robot2_sensor_magnetic_encoder",
)
WIDTHS = (7, 1, 7, 1)
COLUMNS = ("episode_index", "frame_index", "timestamp", "task_index") + SIGNALS


def compute_view_fingerprint(meta):
    """Identify the selected window view, independently of its current location.

    Keep this stdlib-only formula identical to build_umi_access_index.py. Hashes
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


class _BoundedCache:
    def __init__(self, max_items, max_bytes=None, close=None):
        self.max_items, self.max_bytes, self.closer = max_items, max_bytes, close
        self.items, self.bytes = OrderedDict(), 0

    def get(self, key):
        if key not in self.items:
            return None
        value, size = self.items.pop(key)
        self.items[key] = (value, size)
        return value

    def put(self, key, value, size=0):
        if self.max_items == 0 or (self.max_bytes is not None and size > self.max_bytes):
            return value
        if key in self.items:
            self._remove(key)
        while self.items and (len(self.items) >= self.max_items or
                              (self.max_bytes is not None and self.bytes + size > self.max_bytes)):
            self._remove(next(iter(self.items)))
        self.items[key] = (value, size)
        self.bytes += size
        return value

    def _remove(self, key):
        value, size = self.items.pop(key)
        self.bytes -= size
        if self.closer is not None:
            self.closer(value)

    def clear(self):
        while self.items:
            self._remove(next(iter(self.items)))


class UMIIndexedDataset(Dataset):
    """Map each integer to one complete current+16-future window.

    Metadata stays in read-only SQLite; ranges are memory mapped. Per-worker
    caches have explicit bounds. SQLite/video handles are opened lazily in the
    worker and discarded on fork/pickle. This class never writes source files.
    """
    def __init__(self, index_dir, *, source_root=None, image_size=(224, 224),
                 row_cache_mib=128, row_cache_items=32, file_cache_items=4,
                 video_cache_items=8, episode_cache_items=128,
                 source_identity="size_mtime", decode_tolerance_seconds=None,
                 return_metadata=False, allowed_video_roots=()):
        self.index_dir = Path(index_dir).resolve()
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        if (self.meta.get("version") != "roban-umi-access-v1" or self.meta.get("horizon") != 16
                or self.meta.get("status") != "completed"):
            raise ValueError("Require a completed roban-umi-access-v1 index with horizon=16")
        # Legacy full indexes derive the same identity without rewriting meta.
        # A stored identity is checked against the declared artifact hashes;
        # payload hash verification remains an offline index-validation step.
        self.view_fingerprint = compute_view_fingerprint(self.meta)
        if ("view_fingerprint" in self.meta
                and self.meta["view_fingerprint"] != self.view_fingerprint):
            raise ValueError("Access view fingerprint disagrees with index metadata")
        if tuple(self.meta["signals"]) != SIGNALS or tuple(self.meta["signal_widths"]) != WIDTHS:
            raise ValueError("Access index differs from robot1/robot2 world-pose-16D contract")
        for name in ("ranges.npy", "cumulative.npy", "metadata.sqlite3"):
            if (self.index_dir / name).stat().st_size != self.meta["artifacts"][name]["size_bytes"]:
                raise ValueError(f"Incomplete runtime index file: {name}")
        if tuple(self.meta["video_keys"]) != CAMERAS:
            raise ValueError("Access index does not use the established four-view order")
        self.source_root = Path(source_root or self.meta["source_path"]).resolve()
        # Curated datasets may link videos to a sibling source-data directory.
        # Extra read roots are explicit configuration, never inferred from an
        # arbitrary link target. Numeric Parquet remains confined to source_root.
        if isinstance(allowed_video_roots, (str, Path)):
            raise TypeError("allowed_video_roots must be a sequence of directory paths")
        self.allowed_video_roots = tuple(Path(path).resolve() for path in allowed_video_roots)
        self.fps = float(self.meta["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid FPS in access index")
        if source_identity not in ("size_mtime", "size"):
            raise ValueError("source_identity must be size_mtime, or explicitly size after relocation")
        self.source_identity = source_identity
        self.image_size = tuple(operator.index(x) for x in image_size)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        self.row_cache_mib = float(row_cache_mib)
        if not math.isfinite(self.row_cache_mib) or self.row_cache_mib < 0:
            raise ValueError("row_cache_mib must be finite and nonnegative")
        for name, value, minimum in (
            ("row_cache_items", row_cache_items, 0), ("file_cache_items", file_cache_items, 1),
            ("video_cache_items", video_cache_items, 1), ("episode_cache_items", episode_cache_items, 1),
        ):
            if operator.index(value) < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            setattr(self, name, operator.index(value))
        self.decode_tolerance_seconds = (1 / self.fps + 1e-6 if decode_tolerance_seconds is None
                                         else float(decode_tolerance_seconds))
        if not math.isfinite(self.decode_tolerance_seconds) or self.decode_tolerance_seconds <= 0:
            raise ValueError("decode_tolerance_seconds must be finite and positive")
        self.return_metadata = bool(return_metadata)
        self._pid = None
        self._db = self._rows = self._files = self._videos = self._episodes = self._tasks = None
        self._ranges = self._cumulative = None
        self._open_arrays()

    def _open_arrays(self):
        self._ranges = np.load(self.index_dir / "ranges.npy", mmap_mode="r", allow_pickle=False)
        self._cumulative = np.load(self.index_dir / "cumulative.npy", mmap_mode="r", allow_pickle=False)
        n = int(self.meta["total_ranges"])
        if (self._ranges.shape != (n, 3) or self._cumulative.shape != (n,) or n == 0
                or self._ranges.dtype != np.dtype("int64")
                or self._cumulative.dtype != np.dtype("int64")
                or int(self._cumulative[-1]) != int(self.meta["total_windows"])):
            raise ValueError("Access index shape/count mismatch")
        lengths = self._ranges[:, 2] - self._ranges[:, 1]
        if np.any(lengths <= 0) or not np.array_equal(np.cumsum(lengths, dtype=np.int64), self._cumulative):
            raise ValueError("Invalid range cumulative counts")

    def _worker(self):
        if self._pid == os.getpid():
            return
        self.close()
        if self._ranges is None:
            self._open_arrays()
        self._db = sqlite3.connect((self.index_dir / "metadata.sqlite3").as_uri() + "?mode=ro", uri=True)
        self._db.execute("PRAGMA query_only=ON")
        self._db.execute("PRAGMA cache_size=-8192")
        self._db.execute("PRAGMA mmap_size=0")
        self._rows = _BoundedCache(self.row_cache_items, int(self.row_cache_mib * 1024**2))
        self._files = _BoundedCache(self.file_cache_items, close=lambda item: item[0].close())
        self._videos = _BoundedCache(self.video_cache_items, close=lambda item: item[0].close())
        self._episodes = _BoundedCache(self.episode_cache_items)
        self._tasks = _BoundedCache(1024)
        self._pid = os.getpid()

    def close(self):
        for name in ("_videos", "_files", "_rows", "_episodes", "_tasks"):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()
            setattr(self, name, None)
        if getattr(self, "_db", None) is not None:
            self._db.close()
        self._db = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_pid", "_db", "_rows", "_files", "_videos", "_episodes", "_tasks", "_ranges", "_cumulative"):
            state[key] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return int(self.meta["total_windows"])

    def locate(self, index):
        """Return episode ID and episode-local row offset, not a global file row."""
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(f"UMI index {index} outside [0,{len(self)})")
        if self._ranges is None:
            self._open_arrays()
        position = int(np.searchsorted(self._cumulative, index, side="right"))
        base = int(self._cumulative[position - 1]) if position else 0
        episode, begin, _ = self._ranges[position]
        return int(episode), int(begin) + index - base

    def _path(self, relative, *, video=False):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid relative asset path: {relative}")
        resolved = (self.source_root / relative).resolve()
        roots = (self.source_root,) + (self.allowed_video_roots if video else ())
        if not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError(f"Asset escapes source root: {relative}")
        return resolved

    def _episode(self, episode):
        cached = self._episodes.get(episode)
        if cached is not None:
            return cached
        row = self._db.execute("SELECT payload FROM episodes WHERE episode_index=?", (episode,)).fetchone()
        if row is None:
            raise ValueError(f"No metadata for episode {episode}")
        metadata = json.loads(row[0])
        segments = self._db.execute(
            "SELECT data_file,file_row_start,file_row_end_exclusive,episode_row_offset_start,"
            "episode_row_offset_end_exclusive FROM segments WHERE episode_index=? "
            "ORDER BY episode_row_offset_start", (episode,)).fetchall()
        return self._episodes.put(episode, (metadata, segments))

    def _parquet(self, relative):
        cached = self._files.get(relative)
        if cached is not None:
            return cached
        expected = self._db.execute("SELECT size_bytes,mtime_ns FROM data_files WHERE data_file=?",
                                    (relative,)).fetchone()
        if expected is None:
            raise ValueError(f"Unindexed data file {relative}")
        path = self._path(relative)
        stat = path.stat()
        if stat.st_size != expected[0] or (self.source_identity == "size_mtime" and stat.st_mtime_ns != expected[1]):
            raise ValueError(f"Source file changed since indexing: {relative}")
        reader = pq.ParquetFile(path)
        missing = set(COLUMNS) - set(reader.schema_arrow.names)
        if missing:
            reader.close()
            raise ValueError(f"Missing columns in {relative}: {sorted(missing)}")
        ends = np.cumsum([reader.metadata.row_group(i).num_rows for i in range(reader.num_row_groups)],
                         dtype=np.int64)
        return self._files.put(relative, (reader, ends))

    def _file_slice(self, relative, begin, end):
        reader, ends = self._parquet(relative)
        if not 0 <= begin < end <= int(ends[-1]):
            raise ValueError(f"File row range outside {relative}: [{begin},{end})")
        pieces = []
        for group in range(int(np.searchsorted(ends, begin, side="right")),
                           int(np.searchsorted(ends, end - 1, side="right")) + 1):
            key = (relative, group)
            table = self._rows.get(key)
            if table is None:
                table = reader.read_row_group(group, columns=list(COLUMNS), use_threads=False)
                self._rows.put(key, table, table.nbytes)
            group_start = int(ends[group - 1]) if group else 0
            a, b = max(begin, group_start) - group_start, min(end, int(ends[group])) - group_start
            pieces.append(table.slice(a, b - a))
        return pa.concat_tables(pieces)

    def read_lowdim(self, index):
        """Return raw state/action and trace metadata; no images or model needed."""
        self._worker()
        episode, anchor = self.locate(index)
        metadata, segments = self._episode(episode)
        stop = anchor + 17
        if stop > metadata["length"]:
            raise ValueError(f"Incomplete indexed window: episode={episode}, anchor={anchor}")
        pieces, locations, next_row = [], [], anchor
        for relative, file_a, file_b, ep_a, ep_b in segments:
            a, b = max(anchor, ep_a), min(stop, ep_b)
            if a >= b:
                continue
            if a != next_row or file_b - file_a != ep_b - ep_a:
                raise ValueError(f"Broken segment map: episode={episode}, anchor={anchor}")
            local_a, local_b = file_a + a - ep_a, file_a + b - ep_a
            pieces.append(self._file_slice(relative, local_a, local_b))
            locations.append({"data_file": relative, "file_row_start": local_a, "file_row_end_exclusive": local_b})
            next_row = b
        if next_row != stop:
            raise ValueError(f"Missing indexed rows: episode={episode}, anchor={anchor}")
        table = pa.concat_tables(pieces)
        identities = table["episode_index"].to_numpy()
        frames = table["frame_index"].to_numpy()
        task_ids = table["task_index"].to_numpy()
        timestamps = table["timestamp"].to_numpy()
        if (len(table) != 17 or not np.all(identities == episode) or not np.all(np.diff(frames) == 1)
                or not np.all(task_ids == task_ids[0]) or not np.isfinite(timestamps).all()):
            raise ValueError(f"Indexed row identities no longer match: episode={episode}, anchor={anchor}")
        chunks = []
        for key, width in zip(SIGNALS, WIDTHS):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if width == 1 and values.shape == (17,):
                values = values[:, None]
            if values.shape != (17, width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid indexed signal {key}: episode={episode}, anchor={anchor}")
            chunks.append(values)
        values = np.concatenate(chunks, axis=1)
        task_id = int(task_ids[0])
        text = self._tasks.get(task_id)
        if text is None:
            record = self._db.execute("SELECT task FROM tasks WHERE task_index=?", (task_id,)).fetchone()
            if record is None or not isinstance(record[0], str) or not record[0].strip():
                raise ValueError(f"No language for task_index={task_id}")
            text = self._tasks.put(task_id, record[0])
        return {"state": values[:1].copy(), "action": values[1:].copy(), "lang": text,
                "trace": {"dataset_index": int(index), "episode_index": episode,
                          "episode_row_offset": anchor, "frame_index": int(frames[0]),
                          "future_frame_indices": frames[1:].tolist(), "timestamp": float(timestamps[0]),
                          "task_index": task_id, "locations": locations,
                          "rule_fingerprint": self.meta["rule_fingerprint"],
                          "view_fingerprint": self.view_fingerprint},
                "cameras": metadata["cameras"]}

    def _video_frame(self, camera, timestamp):
        import av
        relative = camera["video_path"]
        start, end = float(camera["from_timestamp"]), float(camera["to_timestamp"])
        target = start + timestamp
        if not start <= target < end:
            raise ValueError(f"Requested timestamp outside episode video span: {relative}, {target}")
        cached = self._videos.get(relative)
        if cached is None:
            container = av.open(str(self._path(relative, video=True)))
            stream = container.streams.video[0]
            stream.thread_count = 1
            cached = self._videos.put(relative, (container, stream))
        container, stream = cached
        time_base = float(stream.time_base)
        container.seek(math.floor(target / time_base), stream=stream, backward=True, any_frame=False)
        closest, closest_time, distance = None, None, math.inf
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            if start <= time < end:
                error = abs(time - target)
                if error < distance:
                    closest, closest_time, distance = frame, time, error
            if time >= target:
                break
        if closest is None or distance > self.decode_tolerance_seconds:
            raise ValueError(f"Cannot locate image near {target:.6f}s in {relative}; closest error={distance}")
        image = Image.fromarray(closest.to_ndarray(format="rgb24")).resize(self.image_size)
        return image, {"camera": camera["camera"], "video_path": relative,
                       "requested_seconds": target, "decoded_seconds": closest_time,
                       "offset_seconds": closest_time - target}

    def __getitem__(self, index):
        try:
            data = self.read_lowdim(index)
            cameras = {camera["camera"]: camera for camera in data["cameras"]}
            images, decoded = [], []
            for key in CAMERAS:
                image, timing = self._video_frame(cameras[key], data["trace"]["timestamp"])
                images.append(image)
                decoded.append(timing)
            sample = {"image": images, "lang": data["lang"], "state": data["state"],
                      "action": data["action"], "robot_tag": "new_embodiment"}
            if self.return_metadata:
                sample["umi_metadata"] = dict(data["trace"], video_decode=decoded, reader_pid=os.getpid())
            return sample
        except Exception as error:
            raise RuntimeError(f"UMI read failed at dataset_index={index}: {error}") from error

    def provenance(self):
        return {"dataset_type": "umi_indexed", "index_dir": str(self.index_dir),
                "source_root": str(self.source_root), "rule_fingerprint": self.meta["rule_fingerprint"],
                "view_fingerprint": self.view_fingerprint,
                "allowed_video_roots": [str(path) for path in self.allowed_video_roots],
                "num_windows": len(self), "num_episodes": self.meta["total_episodes"],
                "trainable_episodes": self.meta["trainable_episodes"],
                "state_shape": [1, 16], "action_shape": [16, 16], "dtype": "float32",
                "camera_order": list(CAMERAS), "image_size": list(self.image_size),
                "normalization": "none", "source_identity": self.source_identity,
                "decode_tolerance_seconds": self.decode_tolerance_seconds,
                "note": "Source checks and decode tolerance do not prove physical label quality or sensor synchronization."}


def collate_umi_samples(batch):
    # QwenPI accepts a list of per-example dictionaries and handles PIL images.
    return batch


class UMINormalizedDataset(Dataset):
    """Apply fixed parameters outside the raw reader, without mutating samples.

    Access raw low-dimensional values through raw_dataset.read_lowdim(). The
    wrapper deliberately does not alias that method to a transformed result.
    Normalization parameters are picklable; raw worker handles remain managed
    by UMIIndexedDataset's existing process-local lifecycle.
    """
    def __init__(self, raw_dataset, normalizer):
        self.raw_dataset = raw_dataset
        self.normalizer = normalizer

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        raw = self.raw_dataset[index]
        sample = dict(raw)
        sample["state"] = self.normalizer.normalize_state(raw["state"])
        sample["action"] = self.normalizer.normalize_action(raw["action"])
        if "umi_metadata" in raw:
            sample["umi_metadata"] = dict(raw["umi_metadata"], normalization=self.normalizer.provenance())
        return sample

    def provenance(self):
        raw = self.raw_dataset.provenance()
        return dict(raw, raw_reader_normalization="none", normalization="mean_std",
                    model_lowdim_space="fixed_mean_std",
                    normalization_details=self.normalizer.provenance())

    def close(self):
        self.raw_dataset.close()


def apply_umi_normalization(raw_dataset, data):
    """Resolve an explicit transform; never fit statistics at training startup."""
    mode = data.get("normalization", "none")
    if mode == "none":
        if data.get("normalization_statistics") is not None:
            raise ValueError("normalization_statistics was supplied with normalization=none")
        return raw_dataset
    if mode != "mean_std":
        raise ValueError("UMI normalization must be none or mean_std")
    path = data.get("normalization_statistics")
    if not path or not isinstance(path, (str, Path)):
        raise ValueError("mean_std requires an explicit normalization_statistics file")
    purpose = data.get("normalization_purpose", "engineering")
    if purpose not in ("engineering", "formal"):
        raise ValueError("normalization_purpose must be engineering or formal")
    contract_path = data.get("normalization_experiment_contract")
    contract = None
    if contract_path is not None:
        if not isinstance(contract_path, (str, Path)):
            raise TypeError("normalization_experiment_contract must be a JSON file path")
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    from starVLA.dataloader.umi_normalization import UMINormalizer
    normalizer = UMINormalizer(
        "mean_std", statistics_path=path, access_meta=raw_dataset.meta,
        current_view_fingerprint=raw_dataset.view_fingerprint,
        purpose=purpose, experiment_contract=contract,
    )
    return UMINormalizedDataset(raw_dataset, normalizer)


def make_umi_dataloader(cfg):
    """Factory used by the existing trainer; Accelerate shards the loader once."""
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from starVLA.dataloader.umi_sampler import UMIBlockShuffleSampler

    data = cfg.datasets.vla_data
    for key, expected in (("action_dim", 16), ("state_dim", 16), ("action_horizon", 16)):
        if int(cfg.framework.action_model[key]) != expected:
            raise ValueError(f"Indexed UMI requires framework.action_model.{key}={expected}")
    options = {key: data[key] for key in (
        "source_root", "allowed_video_roots", "image_size", "row_cache_mib", "row_cache_items", "file_cache_items",
        "video_cache_items", "episode_cache_items", "source_identity", "decode_tolerance_seconds",
        "return_metadata") if key in data and data[key] is not None}
    dataset = UMIIndexedDataset(data.index_dir, **options)
    dataset = apply_umi_normalization(dataset, data)
    sampler = UMIBlockShuffleSampler(len(dataset), block_size=int(data.get("shuffle_block_size", 4096)),
                                     seed=int(data.get("seed", 42)), shuffle=data.get("shuffle", True))
    workers = int(data.get("num_workers", 2))
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    kwargs = dict(dataset=dataset, batch_size=int(data.get("per_device_batch_size", 1)),
                  sampler=sampler, collate_fn=collate_umi_samples, num_workers=workers,
                  pin_memory=data.get("pin_memory", False), drop_last=data.get("drop_last", True),
                  generator=torch.Generator().manual_seed(int(data.get("seed", 42))))
    if workers:
        kwargs.update(persistent_workers=data.get("persistent_workers", True),
                      prefetch_factor=int(data.get("prefetch_factor", 2)),
                      multiprocessing_context=data.get("multiprocessing_context", "spawn"))
    loader = DataLoader(**kwargs)
    if not dist.is_initialized() or dist.get_rank() == 0:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        record = dict(dataset.provenance(), sampler=sampler.state_dict(),
                      num_workers=workers, per_device_batch_size=kwargs["batch_size"],
                      drop_last=kwargs["drop_last"],
                      resume_status="sampler API available; trainer checkpoint integration is separate")
        temporary = output / "dataset_access.json.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "dataset_access.json")
    return loader
