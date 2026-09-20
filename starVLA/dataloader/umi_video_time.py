"""Versioned logical video boundaries, separate from query matching tolerance.

Legacy callers retain the microsecond-grid policy. New engineering runs opt in
to a bounded 1 ms candidate margin. Neither policy proves physical ownership of
frames in a concatenated video. Query timestamps remain inside logical bounds.
"""
from fractions import Fraction
import math

VIDEO_TIME_POLICY = {
    "version": "umi-microsecond-grid-half-open-v1",
    "boundary_quantum_seconds": "1/1000000",
    "boundary_mapping": "floor both endpoints for microsecond-grid frames only",
    "non_grid_frames": "strict rational metadata interval",
    "request_matching": "unchanged decode_tolerance_seconds against original request",
}
MICROSECOND = Fraction(1, 1000000)
MAX_BOUNDARY_SLACK = Fraction(1, 1000)


def video_time_policy(name="legacy_microsecond", boundary_slack_seconds=None):
    if name == "legacy_microsecond":
        if boundary_slack_seconds not in (None, 0):
            raise ValueError("Legacy video policy requires zero boundary slack")
        return dict(VIDEO_TIME_POLICY), Fraction(0)
    if name != "engineering_boundary_slack":
        raise ValueError(f"Unknown video time policy: {name}")
    slack = MAX_BOUNDARY_SLACK if boundary_slack_seconds is None else seconds(boundary_slack_seconds)
    if not 0 <= slack <= MAX_BOUNDARY_SLACK:
        raise ValueError("Boundary slack must be between 0 and 0.001 seconds")
    return {"version": "umi-logical-boundary-slack-v1",
            "boundary_slack_seconds": float(slack),
            "candidate_interval": "[logical_start-slack, logical_end+slack)",
            "query_interval": "[logical_start, logical_end)",
            "request_matching": "unchanged decode_tolerance_seconds against original request",
            "tie_break": "earlier presentation timestamp",
            "physical_ownership_verified": False}, slack


def seconds(value):
    if not math.isfinite(float(value)):
        raise ValueError("Video timestamp must be finite")
    return Fraction(str(value))


class VideoTimeWindow:
    def __init__(self, start, end, timestamp, time_base, tolerance, *,
                 policy="legacy_microsecond", boundary_slack_seconds=None):
        self.start, self.end = seconds(start), seconds(end)
        self.target = self.start + seconds(timestamp)
        self.time_base = Fraction(time_base)
        self.tolerance = seconds(tolerance)
        self.policy, self.slack = video_time_policy(policy, boundary_slack_seconds)
        self.legacy = policy == "legacy_microsecond"
        if self.time_base <= 0 or self.tolerance <= 0 or not self.start < self.end:
            raise ValueError("Invalid video interval, time base or tolerance")
        if not self.start <= self.target < self.end:
            raise ValueError("Requested timestamp outside episode video span")
        self.grid_compatible = (MICROSECOND / self.time_base).denominator == 1
        self.lower = (self.start // MICROSECOND) * MICROSECOND
        self.upper = (self.end // MICROSECOND) * MICROSECOND
        if self.legacy and self.grid_compatible and self.lower >= self.upper:
            raise ValueError("Video span is ambiguous at the microsecond boundary grid")

    def frame_time(self, pts, time_base):
        if type(pts) is not int:
            raise ValueError("Frame must have integer PTS")
        frame_base = Fraction(time_base)
        if frame_base != self.time_base:
            raise ValueError("Decoded frame time base differs from selected stream")
        return pts * frame_base

    def contains(self, frame_time):
        if not self.legacy:
            return self.start - self.slack <= frame_time < self.end + self.slack
        on_grid = self.grid_compatible and (frame_time / MICROSECOND).denominator == 1
        lower, upper = (self.lower, self.upper) if on_grid else (self.start, self.end)
        return lower <= frame_time < upper

    @property
    def seek_pts(self):
        # The previous keyframe may contain a closer eligible frame. Seek back
        # by the unchanged matching tolerance before scanning in PTS order.
        seek_time = self.target if self.legacy else self.target - self.tolerance
        return seek_time // self.time_base

    def select(self, frames):
        """Nearest eligible decoded frame; equal distances choose earlier PTS.

        PyAV emits presentation-ordered frames. Retain the preceding candidate
        and compare the first candidate at/after the query before stopping.
        Decode errors propagate, including errors after an earlier candidate.
        """
        selected, selected_time, best = None, None, None
        for frame in frames:
            if frame.pts is None:
                continue
            time = self.frame_time(frame.pts, frame.time_base)
            if self.contains(time):
                key = (abs(time - self.target), time)
                if best is None or key < best:
                    selected, selected_time, best = frame, time, key
            if time >= self.target:
                break
        if selected is None or best[0] > self.tolerance:
            error = float(best[0]) if best else math.inf
            raise ValueError(f"Cannot locate image near {float(self.target):.6f}s; closest error={error}")
        return selected, selected_time
