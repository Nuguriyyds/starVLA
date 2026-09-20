"""Time-grid boundaries are distinct from nearest-frame matching tolerance."""
from fractions import Fraction as F
import unittest
from types import SimpleNamespace

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


class EngineeringSlackTests(unittest.TestCase):
    def window(self, **kw):
        return VideoTimeWindow(1, 2, kw.pop("timestamp", 0), F(1, 10**9), 1/30+1e-6,
                               policy="engineering_boundary_slack", **kw)

    def frames(self, *times):
        return [SimpleNamespace(pts=int(F(str(t))*10**9), time_base=F(1, 10**9)) for t in times]

    def test_microseconds_select_earlier_not_next_frame(self):
        w = self.window(boundary_slack_seconds=.001)
        frame, _ = w.select(self.frames(.999985009, 1.033427009))
        self.assertEqual(frame.pts, 999985009)
        self.assertEqual(w.tolerance, F(str(1/30+1e-6)))

    def test_fixed_cap_and_half_open_candidate_interval(self):
        w = self.window()
        self.assertTrue(w.contains(F('0.999')))
        self.assertFalse(w.contains(F('0.998999999')))
        self.assertTrue(w.contains(F('2.000999999')))
        self.assertFalse(w.contains(F('2.001')))
        for slack in (.0011, -.001, float('nan')):
            with self.assertRaises(ValueError): self.window(boundary_slack_seconds=slack)

    def test_milliseconds_outside_cap_not_admitted(self):
        w = self.window()
        for t in (.994, .986): self.assertFalse(w.contains(F(str(t))))
        with self.assertRaisesRegex(ValueError, 'Cannot locate'):
            w.select(self.frames(.994, 1.04))

    def test_request_itself_stays_inside_logical_interval(self):
        for t in (-.00001, 1):
            with self.assertRaisesRegex(ValueError, 'outside'): self.window(timestamp=t)

    def test_ties_and_interior_nearest(self):
        w = self.window(timestamp=.5)
        self.assertEqual(w.select(self.frames(1.49, 1.51))[0].pts, 1490000000)
        self.assertEqual(w.select(self.frames(1.48, 1.501))[0].pts, 1501000000)

    def test_seek_includes_previous_frame_even_at_keyframe(self):
        w = self.window(timestamp=.5)
        self.assertLess(w.seek_pts, int(w.target / w.time_base))

    def test_missing_frames_distance_and_decoder_errors_propagate(self):
        with self.assertRaises(ValueError): self.window().select([])
        with self.assertRaises(ValueError): self.window(timestamp=.5).select(self.frames(1.54))
        def corrupt():
            yield self.frames(.99999)[0]
            raise OSError('damaged video')
        with self.assertRaisesRegex(OSError, 'damaged video'): self.window().select(corrupt())


if __name__ == "__main__":
    unittest.main()
