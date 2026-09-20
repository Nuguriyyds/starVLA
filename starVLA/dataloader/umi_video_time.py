"""Rational video-time comparisons for the observed UMI microsecond PTS export.

The MP4 stream may advertise nanoseconds while actual frame PTS are quantized to
microseconds. Fractional-microsecond metadata boundaries use one monotone floor
map at BOTH endpoints, never symmetric interval dilation. Only microsecond-grid
frames on compatible stream time bases use this policy; other frames use exact
metadata bounds. Disjoint original spans remain disjoint for every candidate.
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


def seconds(value):
    if not math.isfinite(float(value)):
        raise ValueError("Video timestamp must be finite")
    return Fraction(str(value))


class VideoTimeWindow:
    def __init__(self, start, end, timestamp, time_base, tolerance):
        self.start, self.end = seconds(start), seconds(end)
        self.target = self.start + seconds(timestamp)
        self.time_base = Fraction(time_base)
        self.tolerance = seconds(tolerance)
        if self.time_base <= 0 or self.tolerance <= 0 or not self.start < self.end:
            raise ValueError("Invalid video interval, time base or tolerance")
        if not self.start <= self.target < self.end:
            raise ValueError("Requested timestamp outside episode video span")
        self.grid_compatible = (MICROSECOND / self.time_base).denominator == 1
        self.lower = (self.start // MICROSECOND) * MICROSECOND
        self.upper = (self.end // MICROSECOND) * MICROSECOND
        if self.grid_compatible and self.lower >= self.upper:
            raise ValueError("Video span is ambiguous at the microsecond boundary grid")

    def frame_time(self, pts, time_base):
        if type(pts) is not int:
            raise ValueError("Frame must have integer PTS")
        frame_base = Fraction(time_base)
        if frame_base != self.time_base:
            raise ValueError("Decoded frame time base differs from selected stream")
        return pts * frame_base

    def contains(self, frame_time):
        on_grid = self.grid_compatible and (frame_time / MICROSECOND).denominator == 1
        lower, upper = (self.lower, self.upper) if on_grid else (self.start, self.end)
        return lower <= frame_time < upper

    @property
    def seek_pts(self):
        # Floor in the actual stream unit, including negative timestamps.
        return self.target // self.time_base
