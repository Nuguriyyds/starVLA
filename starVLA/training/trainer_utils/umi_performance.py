"""Opt-in measurements only: no training state, seeding or sampler advancement.

Host sections overlap device work. Only synchronized, coordinated segments are
throughput measurements; detail CUDA events are restricted to warmup updates.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist


def distribution(values):
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), mean=float(array.mean()), p50=float(np.percentile(array, 50)),
                p95=float(np.percentile(array, 95)), maximum=float(array.max()))


def process_resources():
    # psutil is optional unless performance collection is explicitly enabled.
    import psutil
    parent = psutil.Process()
    records = []
    for process in [parent] + parent.children(recursive=True):
        try:
            cpu = process.cpu_times()
            records.append(dict(pid=process.pid, role="parent" if process.pid == parent.pid else "child",
                                rss_bytes=process.memory_info().rss, fds=process.num_fds(),
                                cpu_seconds=cpu.user + cpu.system))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return records


class Performance:
    def __init__(self, config=None, accelerator=None):
        self.config = dict(config or {})
        self.enabled = self.config.get("enabled", False)
        self.accelerator = accelerator
        self.origin = time.monotonic()
        self.sections, self.updates, self.resources, self.segments = [], [], [], []
        self.pending_events = []
        self.detail = False
        self.segment = None
        self.completed = 0
        self.training_start = None
        self.device = accelerator.device if accelerator else torch.device("cpu")
        self.warmup = int(self.config.get("warmup_updates", 2))
        self.detail_count = int(self.config.get("detail_updates", 0))
        if self.warmup < self.detail_count or self.detail_count < 0:
            raise ValueError("detail_updates must be nonnegative and contained in warmup_updates")

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def timing_probe(self):
        if not self.enabled or self.device.type != "cuda":
            return
        # Deterministic tensor: never consumes the model RNG.
        self.sync()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tensor = torch.ones((256, 256), device=self.device)
        wall = time.monotonic()
        start.record()
        result = tensor @ tensor
        end.record()
        self.sync()
        elapsed = start.elapsed_time(end)
        if not torch.isfinite(result).all() or not np.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Device timing probe failed")
        self.probe = dict(event_ms=elapsed, synchronized_wall_seconds=time.monotonic() - wall,
                          device=str(self.device), device_name=torch.cuda.get_device_name(self.device))

    @contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        events = None
        if self.detail and self.device.type == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        try:
            yield
        finally:
            record = dict(name=name, host_seconds=time.monotonic() - start)
            self.sections.append(record)
            if events:
                events[1].record()
                self.pending_events.append((record, events))

    def call(self, name, function, *args, **kwargs):
        with self.span(name):
            return function(*args, **kwargs)

    def begin_update(self):
        if not self.enabled:
            return
        if self.training_start is None:
            self.training_start = time.monotonic()
        self.detail = self.completed < self.detail_count
        if self.completed >= self.warmup and self.segment is None:
            self.sync()
            if self.accelerator:
                self.accelerator.wait_for_everyone()
            self.segment = dict(start=time.monotonic(), updates=0)
        self.update_start = time.monotonic()
        self.section_start = len(self.sections)

    def end_update(self, step, windows):
        if not self.enabled:
            return
        if self.detail:
            self.sync()
            for record, (start, end) in self.pending_events:
                record["device_event_seconds"] = start.elapsed_time(end) / 1000
            self.pending_events.clear()
        self.updates.append(dict(global_update=step, warmup=self.completed < self.warmup,
                                 host_seconds=time.monotonic() - self.update_start,
                                 sections=self.sections[self.section_start:]))
        if self.segment is not None:
            self.segment["updates"] += 1
            self.segment["windows_per_update"] = windows
        self.completed += 1
        self.detail = False
        self.sample_resources()

    def sample_resources(self):
        if not self.enabled:
            return
        record = dict(elapsed_seconds=time.monotonic() - self.origin, processes=process_resources())
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            record["device"] = dict(allocated_bytes=torch.cuda.memory_allocated(self.device),
                                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                                    device_used_bytes=total-free, device_total_bytes=total)
        self.resources.append(record)

    def suspend(self):
        if not self.enabled or self.segment is None:
            return
        self.sync()
        segment, self.segment = self.segment, None
        seconds = time.monotonic() - segment.pop("start")
        if dist.is_initialized():
            value = torch.tensor(seconds, dtype=torch.float64, device=self.device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            seconds = value.item()
        segment["coordinated_seconds"] = seconds
        self.segments.append(segment)

    def expired(self):
        return bool(self.enabled and self.training_start is not None and
                    time.monotonic() - self.training_start >= self.config.get("max_training_seconds", float("inf")))

    def finish(self, path):
        if not self.enabled:
            return
        self.suspend()
        self.sample_resources()
        seconds = sum(s["coordinated_seconds"] for s in self.segments)
        windows = sum(s["updates"] * s["windows_per_update"] for s in self.segments)
        total = time.monotonic() - self.origin
        report = dict(version="umi-performance-v1", config=self.config, pid=os.getpid(),
                      timing_probe=getattr(self, "probe", None), elapsed_seconds=total,
                      completed_updates=self.completed, segments=self.segments,
                      measured_windows=windows, stable_seconds=seconds,
                      stable_windows_per_second=windows / seconds if seconds else None,
                      sections=self.sections, updates=self.updates, resources=self.resources,
                      caveats=["Host sections overlap async device work; do not sum them as device time.",
                               "RSS sums may double-count shared pages; device_used includes non-PyTorch use.",
                               "Stable interval includes existing checks/logging and lightweight resource collection."])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


class DecodedReplayDataset:
    """Bounded cache of an explicit upcoming stream; uncached samples fail closed.

    The original dataset length, sampler order and per-sample identities remain
    unchanged. Only decoded PIL/NumPy samples are cached, never model features.
    Used solely by opt-in engineering benchmarks, with a single-process loader.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(dict.fromkeys(int(i) for i in indices))
        self.samples = {i: dataset[i] for i in self.indices}
        self.dataset.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index not in self.samples:
            raise ValueError(f"Replay sample {index} was not explicitly preloaded; increase bounded replay_samples")
        return deepcopy(self.samples[index])

    def provenance(self):
        return self.dataset.provenance()

    def close(self):
        self.dataset.close()
