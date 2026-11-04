"""Calibrated IC Profiler profiles and geometric beam-centre trends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamio import profile


@dataclass
class ICProfilerTimeSeries:
    """One axis across all frames, with seconds, millimetres, and calibrated signals.

    ``doses`` has shape (frames, detectors), aligned with ``times`` and
    ``positions``. Interval signals/pulses are differences from the preceding
    cumulative frame (zero before the first frame). ``valid_frames`` describes
    timing and pulse validity; unusable rows and ignored detectors have NaN dose.
    For cumulative data, validity uses elapsed time and cumulative pulses instead.
    """

    axis: str
    times: np.ndarray
    durations: np.ndarray
    positions: np.ndarray
    doses: np.ndarray
    pulses: np.ndarray
    updates: np.ndarray
    statuses: np.ndarray
    valid_frames: np.ndarray
    cumulative: bool
    dose_scale: str
    detector_ids: tuple[str, ...]

    def detector(self, detector_id: str) -> np.ndarray:
        """Return one detector's calibrated signal across all frames, aligned with times.

        The returned one-dimensional array is independent of this series. Use
        ordinary NumPy slicing to select frames, such as ``detector('X33')[10:20]``.
        """
        try:
            column = self.detector_ids.index(detector_id)
        except ValueError:
            raise ValueError(
                f"Detector {detector_id!r} is not present on axis {self.axis}."
            ) from None
        return np.array(self.doses[:, column], dtype=np.float64, copy=True)

    def field_edges(
        self,
        *,
        method: str = "InflectionPoint",
        level: float = 0.5,
        min_signal: float = 0.05,
    ) -> np.ndarray:
        """Return left/right field edges in mm for each frame.

        ``InflectionPoint`` uses the maximum rising and falling spatial slopes,
        with local quadratic refinement of isolated derivative peaks. Flat
        derivative maxima use their midpoint; separated equal maxima are ambiguous.
        ``FWHM`` uses interpolated threshold crossings; ``level`` is a fraction
        of each frame's positive maximum (default 0.5) and applies to FWHM only.
        ``min_signal`` rejects peaks below this fraction of the strongest frame.
        Both methods use the actual detector positions, without bridging ignored
        edge samples or extrapolating beyond the array. Unusable edges give NaN.
        """
        if method not in ("InflectionPoint", "FWHM"):
            raise ValueError("method must be InflectionPoint or FWHM.")
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be a finite fraction between 0 and 1.")
        if not np.isfinite(min_signal) or not 0 <= min_signal <= 1:
            raise ValueError("min_signal must be a finite fraction from 0 to 1.")
        edges = np.full((len(self.times), 2), np.nan)
        peaks = np.max(
            np.where(np.isfinite(self.doses), self.doses, -np.inf), axis=1, initial=-np.inf
        )
        cutoff = min_signal * np.max(peaks[self.valid_frames], initial=0)
        for index in np.flatnonzero(self.valid_frames):
            edges[index] = profile.field_edges(
                self.positions,
                self.doses[index],
                method=method,
                level=level,
                min_signal=float(cutoff),
            )
        return edges

    def beam_center(
        self,
        *,
        method: str = "InflectionPoint",
        level: float = 0.5,
        min_signal: float = 0.05,
    ) -> np.ndarray:
        """Return the midpoint of the two selected field edges, in mm, for each frame."""
        edges = self.field_edges(method=method, level=level, min_signal=min_signal)
        return np.asarray((edges[:, 0] + edges[:, 1]) / 2, dtype=np.float64)

    def field_width(
        self,
        *,
        method: str = "InflectionPoint",
        level: float = 0.5,
        min_signal: float = 0.05,
    ) -> np.ndarray:
        """Return the separation of the two selected field edges, in mm, for each frame."""
        edges = self.field_edges(method=method, level=level, min_signal=min_signal)
        return np.asarray(edges[:, 1] - edges[:, 0], dtype=np.float64)


@dataclass
class ICProfilerBeamCenter:
    """X/Y beam-centre positions in mm at each exported elapsed time in seconds."""

    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cumulative: bool
    level: float | None
    min_signal: float
    method: str = "InflectionPoint"

    @property
    def valid_frames(self) -> np.ndarray:
        """Frames with a detectable centre on both axes."""
        return np.isfinite(self.x) & np.isfinite(self.y)
