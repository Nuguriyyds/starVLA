"""CPU-only properties for UMI sampling and Accelerate's one-time sharding.

No dataset, model, GPU, or training process is constructed. The Accelerate
tests deliberately wrap the same unpartitioned global sampler on every rank.
This verifies the installed BatchSamplerShard contract, not a multi-process
trainer's checkpoint coordination or numerical reproducibility.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

import torch
from torch.utils.data import BatchSampler


_SAMPLER_PATH = Path(__file__).resolve().parents[1] / "starVLA/dataloader/umi_sampler.py"
_SPEC = importlib.util.spec_from_file_location("umi_sampler_under_test", _SAMPLER_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
UMIBlockShuffleSampler = _MODULE.UMIBlockShuffleSampler


class UMIBlockShuffleSamplerTests(unittest.TestCase):
    def test_full_coverage_including_short_final_blocks(self):
        for size in (0, 1, 2, 7, 16, 17, 31, 33, 101):
            for block_size in (1, 4, 16, 128):
                with self.subTest(size=size, block_size=block_size):
                    sampler = UMIBlockShuffleSampler(size, block_size, seed=735, epoch=3)
                    values = list(sampler)
                    self.assertEqual(len(sampler), size)
                    self.assertEqual(len(values), size)
                    self.assertEqual(len(set(values)), size)
                    self.assertEqual(sorted(values), list(range(size)))

    def test_same_seed_and_epoch_ignore_global_rng(self):
        rng_state = torch.get_rng_state()
        try:
            first = list(UMIBlockShuffleSampler(103, 11, seed=45, epoch=8))
            torch.manual_seed(291)
            torch.rand(1000)
            second = list(UMIBlockShuffleSampler(103, 11, seed=45, epoch=8))
            self.assertEqual(first, second)
            before = torch.get_rng_state().clone()
            list(UMIBlockShuffleSampler(103, 11, seed=45, epoch=8))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        finally:
            torch.set_rng_state(rng_state)

    def test_different_epochs_change_order(self):
        sampler = UMIBlockShuffleSampler(103, 11, seed=45)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        self.assertEqual(sorted(first), sorted(second))

    def test_resume_at_every_offset_including_middle_of_blocks(self):
        kwargs = dict(size=53, block_size=8, seed=42, epoch=7)
        entire_stream = list(UMIBlockShuffleSampler(**kwargs))
        for offset in range(54):
            with self.subTest(offset=offset):
                resumed = UMIBlockShuffleSampler(**kwargs, start_index=offset)
                self.assertEqual(list(resumed), entire_stream[offset:])
                self.assertEqual(len(resumed), 53 - offset)

    def test_nonshuffled_resume(self):
        sampler = UMIBlockShuffleSampler(23, 4, shuffle=False, start_index=5)
        self.assertEqual(list(sampler), list(range(5, 23)))
        sampler.set_epoch(10, start_index=8)
        self.assertEqual(list(sampler), list(range(8, 23)))

    def test_iteration_does_not_advance_explicit_consumed_cursor(self):
        sampler = UMIBlockShuffleSampler(53, 8, seed=42, epoch=7, start_index=3)
        state = sampler.state_dict()
        iterator = iter(sampler)
        for _ in range(9):
            next(iterator)
        self.assertEqual(sampler.state_dict(), state)
        list(iterator)
        self.assertEqual(sampler.state_dict(), state)

    def test_valid_restore_and_returned_state_are_independent(self):
        original = UMIBlockShuffleSampler(53, 8, seed=42, epoch=7, start_index=19)
        state = original.state_dict()
        saved_state = copy.deepcopy(state)
        restored = UMIBlockShuffleSampler(53, 8, seed=42)
        restored.load_state_dict(state)
        self.assertEqual(state, saved_state)
        self.assertEqual(list(restored), list(original))
        state["epoch"] = 99
        state["start_index"] = 0
        self.assertEqual(restored.state_dict(), saved_state)
        self.assertEqual(original.state_dict(), saved_state)

    def test_invalid_restore_is_atomic(self):
        sampler = UMIBlockShuffleSampler(53, 8, seed=42, epoch=7, start_index=19)
        original = sampler.state_dict()
        invalid_states = []
        for field, value in (
            ("version", 2), ("size", 54), ("block_size", 9),
            ("seed", 99), ("shuffle", False), ("epoch", -1),
            ("epoch", True), ("start_index", 54), ("start_index", 2.5),
        ):
            state = dict(original)
            state.update(epoch=9, start_index=20)
            state[field] = value
            invalid_states.append(state)
        missing = dict(original)
        del missing["seed"]
        invalid_states.append(missing)
        extra = dict(original, unexpected=1)
        invalid_states.append(extra)
        for state in invalid_states:
            with self.subTest(state=state):
                incoming = copy.deepcopy(state)
                with self.assertRaises((ValueError, TypeError)):
                    sampler.load_state_dict(state)
                self.assertEqual(sampler.state_dict(), original)
                self.assertEqual(state, incoming)
        with self.assertRaises(TypeError):
            sampler.load_state_dict([])
        self.assertEqual(sampler.state_dict(), original)

    def test_set_epoch_resets_cursor_and_invalid_update_is_atomic(self):
        sampler = UMIBlockShuffleSampler(53, 8, epoch=3, start_index=19)
        old = sampler.state_dict()
        with self.assertRaises(ValueError):
            sampler.set_epoch(10, start_index=54)
        self.assertEqual(sampler.state_dict(), old)
        sampler.set_epoch(4)
        self.assertEqual(sampler.epoch, 4)
        self.assertEqual(sampler.start_index, 0)


class AccelerateBatchSamplerShardTests(unittest.TestCase):
    """Check exact rank streams, including duplication/truncation of tails.

    split_batches=False distributes whole batches, not elements of each batch.
    even_batches=False, drop_last=False preserves the global stream exactly,
    but ranks can receive unequal batch counts and partial final batches.
    even_batches=True repeats the global prefix to complete a rank group.
    drop_last=True truncates to full rank groups even when even_batches=False.
    """

    @classmethod
    def setUpClass(cls):
        # Fail if the production dependency is unavailable; do not silently skip.
        from accelerate.data_loader import BatchSamplerShard
        cls.Shard = BatchSamplerShard

    def check_shards(self, size, *, even_batches, drop_last, start_index=0):
        kwargs = dict(size=size, block_size=4, seed=63, epoch=3, start_index=start_index)
        global_stream = list(UMIBlockShuffleSampler(**kwargs))
        batch_size, world_size = 3, 2
        ranks = []
        for rank in range(world_size):
            sampler = UMIBlockShuffleSampler(**kwargs)
            before = sampler.state_dict()
            batch_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
            shard = self.Shard(
                batch_sampler, num_processes=world_size, process_index=rank,
                split_batches=False, even_batches=even_batches,
            )
            rank_batches = list(shard)
            self.assertEqual(len(shard), len(rank_batches))
            self.assertEqual(sampler.state_dict(), before)
            ranks.append(rank_batches)

        group_size = batch_size * world_size
        if drop_last:
            expected = global_stream[:len(global_stream) // group_size * group_size]
        elif even_batches and global_stream:
            padded_size = ((len(global_stream) + group_size - 1) // group_size) * group_size
            expected = [global_stream[i % len(global_stream)] for i in range(padded_size)]
        else:
            expected = global_stream
        expected_batches = [expected[i:i + batch_size] for i in range(0, len(expected), batch_size)]
        for rank in range(world_size):
            self.assertEqual(ranks[rank], expected_batches[rank::world_size])
        reconstructed = []
        for step in range(max(map(len, ranks), default=0)):
            for rank_batches in ranks:
                if step < len(rank_batches):
                    reconstructed.extend(rank_batches[step])
        self.assertEqual(reconstructed, expected)
        return global_stream, ranks, reconstructed

    def test_uneven_keeps_all_samples_once_without_second_partition(self):
        for size in (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 17, 53):
            with self.subTest(size=size):
                stream, ranks, combined = self.check_shards(size, even_batches=False, drop_last=False)
                self.assertEqual(combined, stream)
                rank_sets = [{i for batch in rank for i in batch} for rank in ranks]
                self.assertFalse(rank_sets[0] & rank_sets[1])
                self.assertEqual(rank_sets[0] | rank_sets[1], set(range(size)))

    def test_even_pads_with_global_prefix(self):
        for size in (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 17, 53):
            with self.subTest(size=size):
                stream, ranks, combined = self.check_shards(size, even_batches=True, drop_last=False)
                self.assertEqual(len(ranks[0]), len(ranks[1]))
                self.assertEqual(combined[:size], stream)
                if size:
                    self.assertEqual(combined[size:], [stream[i % size] for i in range(len(combined) - size)])

    def test_drop_last_discards_incomplete_rank_groups(self):
        for even_batches in (False, True):
            for size in (0, 1, 5, 6, 7, 11, 12, 13, 17, 53):
                with self.subTest(size=size, even_batches=even_batches):
                    stream, ranks, combined = self.check_shards(size, even_batches=even_batches, drop_last=True)
                    self.assertEqual(len(ranks[0]), len(ranks[1]))
                    self.assertEqual(len(combined), (size // 6) * 6)
                    self.assertEqual(combined, stream[:len(combined)])
                    self.assertEqual(len(combined), len(set(combined)))

    def test_resume_suffix_is_partitioned_once(self):
        for even_batches in (False, True):
            for drop_last in (False, True):
                for offset in (1, 5, 7, 16, 17):
                    with self.subTest(even_batches=even_batches, drop_last=drop_last, offset=offset):
                        self.check_shards(17, even_batches=even_batches, drop_last=drop_last, start_index=offset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
