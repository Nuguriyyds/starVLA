"""Time-grid boundaries are distinct from nearest-frame matching tolerance."""
from fractions import Fraction as F
import unittest

from starVLA.dataloader.umi_video_time import VideoTimeWindow


class VideoTimeTests(unittest.TestCase):
    def test_original_failure_accepts_correct_pts(self):
        window = VideoTimeWindow(286.066677666, 407.73334433266666, 0, F(1, 10**9), 1/30+1e-6)
        correct = window.frame_time(286066677000, F(1, 10**9))
        self.assertTrue(window.contains(correct))
        self.assertLess(abs(correct-window.target), F(1, 10**6))
        following = window.frame_time(286100013000, F(1, 10**9))
        self.assertGreater(abs(following-window.target), window.tolerance)

    def test_shared_boundary_has_one_owner_in_different_time_bases(self):
        for base in (F(1, 10**9), F(1, 10**6), F(1, 10**7)):
            with self.subTest(base=base):
                first = VideoTimeWindow(0, 1.000000666, 0, base, .034)
                second = VideoTimeWindow(1.000000666, 2, 0, base, .034)
                self.assertFalse(first.contains(F(1)))
                self.assertTrue(second.contains(F(1)))
                for value in (F(999999,10**6), F(1), F(1000001,10**6), F(10000004,10**7)):
                    self.assertLessEqual(int(first.contains(value))+int(second.contains(value)), 1)

    def test_no_frame_wide_boundary_expansion(self):
        window = VideoTimeWindow(1.000000666, 2.000000666, 0, F(1,10**9), .034)
        self.assertFalse(window.contains(F(999999,10**6)))
        self.assertFalse(window.contains(F(2)))  # exclusive canonical endpoint
        self.assertFalse(window.contains(F(999999999,10**9)))

    def test_off_microsecond_grid_uses_strict_bounds(self):
        window = VideoTimeWindow(1.000000666, 2, 0, F(1,10**9), .034)
        self.assertFalse(window.contains(F(1000000600,10**9)))
        self.assertTrue(window.contains(F(1000000700,10**9)))
        coarse = VideoTimeWindow(.033334, 1, 0, F(1,90000), .034)
        self.assertFalse(coarse.contains(F(1,30)))

    def test_invalid_requests_and_mismatched_time_bases_fail(self):
        for offset in (-1e-9, 1):
            with self.assertRaisesRegex(ValueError, "outside"):
                VideoTimeWindow(1, 2, offset, F(1,10**9), .034)
        window = VideoTimeWindow(1, 2, 0, F(1,10**9), .034)
        with self.assertRaisesRegex(ValueError, "differs"):
            window.frame_time(1000000, F(1,10**6))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            VideoTimeWindow(1.0000001, 1.0000002, 0, F(1,10**9), .034)


if __name__ == "__main__":
    unittest.main()
