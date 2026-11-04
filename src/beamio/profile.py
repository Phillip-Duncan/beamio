"""Vendor-independent field edges, beam centres, and widths from profile samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["field_edges", "beam_center", "field_width"]


def _prepare_samples(
    positions: ArrayLike,
    doses: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(positions, dtype=float)
    signals = np.asarray(doses, dtype=float)
    if coordinates.ndim != 1 or not np.all(np.isfinite(coordinates)):
        raise ValueError("positions must be a one-dimensional array of finite coordinates.")
    if signals.ndim != 1 or signals.size != coordinates.size:
        raise ValueError("doses must be a one-dimensional array matching positions.")
    order = np.argsort(coordinates, kind="stable")
    coordinates = coordinates[order]
    signals = signals[order]
    signals = np.where(np.isfinite(signals), signals, np.nan)
    unique, starts = np.unique(coordinates, return_index=True)
    if unique.size != coordinates.size:
        finite = np.isfinite(signals)
        totals = np.add.reduceat(np.where(finite, signals, 0.0), starts)
        counts = np.add.reduceat(finite.astype(np.int64), starts)
        signals = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
        coordinates = unique
    return coordinates, signals


def field_edges(
    positions: ArrayLike,
    doses: ArrayLike,
    *,
    method: str = "InflectionPoint",
    level: float = 0.5,
    min_signal: float = 0.0,
) -> np.ndarray:
    """Return the left and right field edges of one X/Y profile.

    ``positions`` and ``doses`` are matching one-dimensional arrays. The result
    has shape (2,).
    Edge coordinates use the same units as positions. Inputs are not modified.

    Coordinates are sorted with their signals; repeated coordinates are combined
    by averaging finite readings for analysis. Non-finite signals remain missing
    samples. Neither method bridges missing edge samples or extrapolates.

    ``InflectionPoint`` finds spatial-gradient extrema, refining isolated peaks
    quadratically between sample intervals. ``FWHM`` interpolates crossings at
    ``level`` times the profile maximum (default 0.5). ``level`` only affects FWHM.
    Unmeasurable edges, nonpositive peaks, and insufficient samples give NaN.

    ``min_signal`` is the minimum acceptable positive peak in the supplied dose
    units (default 0). It does not compare this profile with any other data.
    """
    if method not in ("InflectionPoint", "FWHM"):
        raise ValueError("method must be InflectionPoint or FWHM.")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be a finite fraction between 0 and 1.")
    if not np.isfinite(min_signal) or min_signal < 0:
        raise ValueError("min_signal must be a finite nonnegative signal value.")
    coordinates, signals = _prepare_samples(positions, doses)
    peak = np.max(np.where(np.isfinite(signals), signals, -np.inf), initial=-np.inf)
    if coordinates.size < 3 or peak <= 0 or peak < min_signal:
        return np.full(2, np.nan)
    return (
        _inflection_edges(coordinates, signals)
        if method == "InflectionPoint"
        else _fwhm_edges(coordinates, signals, level)
    )


def beam_center(
    positions: ArrayLike,
    doses: ArrayLike,
    *,
    method: str = "InflectionPoint",
    level: float = 0.5,
    min_signal: float = 0.0,
) -> float:
    """Return one profile's field-edge midpoint, in position units.

    See ``field_edges`` for methods, signal threshold, missing samples, and validation.
    """
    edges = field_edges(
        positions,
        doses,
        method=method,
        level=level,
        min_signal=min_signal,
    )
    return float((edges[0] + edges[1]) / 2)


def field_width(
    positions: ArrayLike,
    doses: ArrayLike,
    *,
    method: str = "InflectionPoint",
    level: float = 0.5,
    min_signal: float = 0.0,
) -> float:
    """Return the measured field size along the profile axis, in position units.

    Returns a float. See ``field_edges`` for methods, signal threshold,
    missing samples, and validation.
    """
    edges = field_edges(
        positions,
        doses,
        method=method,
        level=level,
        min_signal=min_signal,
    )
    return float(edges[1] - edges[0])


def _gradient_peak(positions: np.ndarray, slopes: np.ndarray, sign: int) -> float:
    """Locate an interior derivative extremum, refining isolated peaks quadratically."""
    heights = sign * slopes
    maximum = np.max(np.where(np.isfinite(heights), heights, -np.inf))
    if maximum <= 0:
        return np.nan
    strongest = np.flatnonzero(
        np.isfinite(heights) & np.isclose(heights, maximum, rtol=1e-9, atol=0)
    )
    if not strongest.size or np.any(np.diff(strongest) != 1):
        return np.nan
    first, last = int(strongest[0]), int(strongest[-1])
    if first == 0 or last == len(slopes) - 1:
        return np.nan
    if not np.all(np.isfinite(heights[first - 1 : last + 2])):
        return np.nan
    if first != last:
        # A flat maximum has no unique vertex; use its physical midpoint.
        return float((positions[first] + positions[last]) / 2)
    local_x = positions[first - 1 : first + 2] - positions[first]
    local_y = heights[first - 1 : first + 2]
    left_slope = (local_y[1] - local_y[0]) / (local_x[1] - local_x[0])
    right_slope = (local_y[2] - local_y[1]) / (local_x[2] - local_x[1])
    curvature = (right_slope - left_slope) / (local_x[2] - local_x[0])
    if curvature >= 0:
        return np.nan
    linear = left_slope + curvature * (local_x[1] - local_x[0])
    offset = -linear / (2 * curvature)
    if not local_x[0] <= offset <= local_x[2]:
        return np.nan
    return float(positions[first] + offset)


def _inflection_edges(positions: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Slopes belong at interval midpoints and use the actual sample spacing,
    # including nonuniform grids.
    slopes = np.diff(values) / np.diff(positions)
    midpoints = (positions[:-1] + positions[1:]) / 2
    left = _gradient_peak(midpoints, slopes, 1)
    right = _gradient_peak(midpoints, slopes, -1)
    peak_position = positions[int(np.nanargmax(values))]
    if not left < peak_position < right:
        return np.full(2, np.nan)
    # An exterior gap could hide a stronger penumbra than the selected internal
    # gradient. Do not report an apparent field edge when that is unobservable.
    exterior = (positions <= left) | (positions >= right)
    if not np.all(np.isfinite(values[exterior])):
        return np.full(2, np.nan)
    return np.array([left, right])


def _fwhm_edges(positions: np.ndarray, values: np.ndarray, level: float) -> np.ndarray:
    threshold = level * np.nanmax(values)
    above = np.flatnonzero(np.isfinite(values) & (values >= threshold))
    edges = np.full(2, np.nan)
    if not above.size:
        return edges
    left, right = int(above[0]), int(above[-1])
    if left == 0 or right == len(values) - 1:
        return edges
    if not np.isfinite(values[left - 1]) or not np.isfinite(values[right + 1]):
        return edges
    interior = values[left : right + 1]
    if np.any(np.isfinite(interior) & (interior < threshold)):
        return edges
    for side, i, j in ((0, left - 1, left), (1, right, right + 1)):
        edges[side] = positions[i] + (threshold - values[i]) * (positions[j] - positions[i]) / (
            values[j] - values[i]
        )
    return edges
