import copy
import unittest

from starVLA.dataloader.umi_sampler import UMIBlockShuffleSampler
from starVLA.training.trainer_utils.umi_training_state import UMITrainingState, validate_plan


class TrainingStateTests(unittest.TestCase):
    def test_tail_accounting_and_boundaries(self):
        state = UMITrainingState([6, 6], [23, 27], 8)
        for step in range(12):
            state.commit_update()
            clone = UMITrainingState([6, 6], [23, 27], 8)
            clone.load_state_dict(state.state_dict())
            self.assertEqual(clone.state_dict(), state.state_dict())
        self.assertTrue(state.done)
        self.assertEqual(state.stages[0], dict(updates=6, epoch=3, cursor=0, used_samples=48, dropped_tail_samples=21))
        self.assertEqual(state.stages[1], dict(updates=6, epoch=2, cursor=0, used_samples=48, dropped_tail_samples=6))

    def test_invalid_counters_and_small_views(self):
        with self.assertRaises(ValueError):
            UMITrainingState([3], [3], 4)
        state = UMITrainingState([3], [7], 4)
        original = state.state_dict()
        broken = copy.deepcopy(original)
        broken["stages"][0]["cursor"] = 1
        with self.assertRaises(ValueError):
            state.load_state_dict(broken)
        self.assertEqual(state.state_dict(), original)

    def test_same_epoch_keeps_restored_position_explicit_reset_works(self):
        sampler = UMIBlockShuffleSampler(29, epoch=2, start_index=8)
        saved = sampler.state_dict()
        sampler.set_epoch(2)
        self.assertEqual(sampler.state_dict(), saved)
        sampler.set_epoch(2, start_index=0)
        self.assertEqual(sampler.start_index, 0)
        sampler.set_start_index(8)
        sampler.set_epoch(3)
        self.assertEqual(sampler.start_index, 0)

    def test_formal_rejects_synthetic(self):
        with self.assertRaisesRegex(ValueError, "formal"):
            validate_plan(dict(version="umi-training-plan-v1", purpose="formal", model_kind="tiny"))


if __name__ == "__main__":
    unittest.main()
