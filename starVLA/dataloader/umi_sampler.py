"""Bounded-memory shuffling of the global UMI sample-index stream.

The sampler deliberately does not partition indices by distributed rank. Pass it
to the ordinary DataLoader and let the training framework (e.g. Accelerate)
partition that loader exactly once. Applying a DistributedSampler as well would
partition the data twice.

``start_index`` is an offset in this epoch's *global sample stream*, not an
episode/frame identifier, per-rank counter, batch counter or optimizer step.
The trainer must explicitly record the globally consumed offset at a coordinated
checkpoint boundary. Iteration never advances this offset: DataLoader workers
can prefetch samples that have not yet contributed to a completed training step.
Saving this sampler alone is not a complete distributed-training resume scheme.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from numbers import Integral
from typing import Any

import torch
from torch.utils.data import Sampler


_UINT64_MASK = (1 << 64) - 1
_STATE_VERSION = 1


def _integer(name: str, value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {result}")
    return result


def _mix64(value: int) -> int:
    """Stable SplitMix64 mixing; independent of Python's randomized hash()."""
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


class UMIBlockShuffleSampler(Sampler[int]):
    """Shuffle blocks, then the indices within each block, without an N-sized permutation.

    At offset zero, every dataset index occurs exactly once in an epoch. The
    final, possibly short block keeps its actual length when shuffled. Random
    state for a block depends only on seed, epoch and block ID; resuming does not
    require replaying permutations of preceding blocks. This is a locality-aware
    shuffle, not a uniformly random permutation over all N! possible orders.

    Working memory is O(ceil(size / block_size) + block_size), rather than O(size)
    at the default block size. With ``shuffle=False``, iteration uses range() and
    constant auxiliary memory. Determinism assumes the same PyTorch RNG behavior;
    keep the PyTorch version fixed when resuming an existing run.

    Args:
        size: Number of indexed training windows, not raw data rows.
        block_size: Number of consecutive dataset indices in each shuffle block.
        seed: Unsigned 64-bit base seed.
        shuffle: Whether to shuffle both block order and indices within blocks.
        epoch: Nonnegative epoch number.
        start_index: Number of samples already consumed from the global stream
            in this epoch, explicitly supplied by the trainer.

    ``state_dict()`` records configuration and the explicit cursor; it does not
    infer progress from yielded indices. When restoring, construct a sampler for
    the same dataset/configuration and call ``load_state_dict()``. The trainer
    must also validate the dataset/index version: equal lengths do not guarantee
    that two datasets contain the same windows.
    """

    def __init__(
        self,
        size: int,
        block_size: int = 4096,
        seed: int = 42,
        shuffle: bool = True,
        *,
        epoch: int = 0,
        start_index: int = 0,
    ) -> None:
        self.size = _integer("size", size, minimum=0, maximum=sys.maxsize)
        self.block_size = _integer(
            "block_size", block_size, minimum=1, maximum=sys.maxsize
        )
        self.seed = _integer("seed", seed, minimum=0, maximum=_UINT64_MASK)
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool")
        self.shuffle = shuffle
        self.epoch = _integer("epoch", epoch, minimum=0, maximum=_UINT64_MASK)
        self.start_index = _integer(
            "start_index", start_index, minimum=0, maximum=self.size
        )

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        """Select an epoch, resetting its offset unless one is explicitly given.

        Call before creating the epoch's iterator. On resume, do not subsequently
        reset this with ``set_epoch(epoch)`` unless offset zero is intended.
        """
        new_epoch = _integer("epoch", epoch, minimum=0, maximum=_UINT64_MASK)
        new_offset = _integer(
            "start_index", start_index, minimum=0, maximum=self.size
        )
        self.epoch = new_epoch
        self.start_index = new_offset

    def set_start_index(self, start_index: int) -> None:
        """Explicitly record global consumed progress; never infer it from prefetch."""
        self.start_index = _integer(
            "start_index", start_index, minimum=0, maximum=self.size
        )

    def __len__(self) -> int:
        return self.size - self.start_index

    def __iter__(self) -> Iterator[int]:
        # Snapshot the caller-owned cursor so later updates affect the next
        # iterator, not the stream currently being consumed.
        size = self.size
        start_index = self.start_index
        block_size = self.block_size
        epoch = self.epoch
        seed = self.seed
        shuffle = self.shuffle

        if not shuffle:
            yield from range(start_index, size)
            return
        if start_index == size:
            return

        num_blocks = (size + block_size - 1) // block_size
        epoch_seed = _mix64(seed ^ _mix64(epoch))
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(_mix64(epoch_seed ^ 0xB10C5EED))
        block_order = torch.randperm(
            num_blocks, generator=order_generator, device="cpu"
        )
        remaining_skip = start_index
        for block_slot in range(num_blocks):
            block_id = int(block_order[block_slot].item())
            block_begin = block_id * block_size
            block_length = min(block_size, size - block_begin)
            if remaining_skip >= block_length:
                remaining_skip -= block_length
                continue

            generator = torch.Generator(device="cpu")
            generator.manual_seed(_mix64(epoch_seed ^ _mix64(block_id)))
            local_order = torch.randperm(
                block_length, generator=generator, device="cpu"
            )
            # Materialize at most one block of Python integers, never all N.
            for local_index in local_order[remaining_skip:].tolist():
                yield block_begin + local_index
            remaining_skip = 0

    def state_dict(self) -> dict[str, Any]:
        """Return configuration plus the explicitly recorded global-stream offset."""
        return {
            "version": _STATE_VERSION,
            "size": self.size,
            "block_size": self.block_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "start_index": self.start_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore epoch/offset only after validating all immutable configuration.

        Validation is atomic: an invalid state leaves this sampler unchanged.
        Unknown or missing fields are rejected rather than silently accepting a
        checkpoint written for a different sampler-state format.
        """
        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping")
        expected_keys = set(self.state_dict())
        if set(state) != expected_keys:
            raise ValueError(
                "sampler state fields differ: "
                f"missing={sorted(expected_keys - set(state))}, "
                f"unexpected={sorted(set(state) - expected_keys, key=str)}"
            )
        version = _integer(
            "version", state["version"], minimum=0, maximum=sys.maxsize
        )
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported sampler state version: {version}")
        restored = UMIBlockShuffleSampler(
            size=state["size"],
            block_size=state["block_size"],
            seed=state["seed"],
            shuffle=state["shuffle"],
            epoch=state["epoch"],
            start_index=state["start_index"],
        )
        for field in ("size", "block_size", "seed", "shuffle"):
            if getattr(restored, field) != getattr(self, field):
                raise ValueError(
                    f"sampler {field} mismatch: current={getattr(self, field)!r}, "
                    f"checkpoint={getattr(restored, field)!r}"
                )
        self.epoch = restored.epoch
        self.start_index = restored.start_index
