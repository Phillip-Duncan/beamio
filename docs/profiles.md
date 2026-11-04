# Profile analysis

`beamio.profile` is a public module for measured field edges, beam centres,
and widths. It accepts position/dose arrays independently of any reader.
The only runtime dependency is NumPy.

```python
from beamio.profile import beam_center, field_edges, field_width

edges = field_edges(positions, doses)        # [left, right]
width = field_width(positions, doses)        # measured field size along this axis
center = beam_center(positions, doses)
fwhm_width = field_width(positions, doses, method="FWHM")
```

Results use the same spatial units as `positions`. Measurements loaded by
beamio use millimetres. Signal scale is retained; the calculations do not
normalize or calibrate the input arrays.

## Use with any BeamData reader

All readers produce `Measurement` objects whose arrays can be passed explicitly:

```python
from beamio.rfa import SNCSunScan
from beamio.profile import beam_center, field_edges, field_width

beam = SNCSunScan("profiles.snctxt").parse()
scan = beam.measurements[0]  # select a crossline, inline, or diagonal profile
width_mm = field_width(scan.positions, scan.doses)
left_mm, right_mm = field_edges(scan.positions, scan.doses, method="FWHM")
center_mm = beam_center(scan.positions, scan.doses)
```

This works with SNC tank, IBA, PTW MCC, IC Profiler, and RayStation profiles.
Select the relevant crossline, inline, or diagonal profile before passing its
arrays. These functions operate on numerical samples and do not inspect a
measurement type, endpoints, or stored XYZ coordinates. Measured width is the
field size along that profile axis. `scan.field_size` continues to hold the
acquisition's nominal X/Y field size.

## Methods

All three functions default to
`method="InflectionPoint"`. It finds the strongest rising and falling spatial
derivatives. Finite differences use the actual coordinate spacing and belong
at interval midpoints. Isolated extrema are refined using a quadratic through
three neighbouring derivative samples. A contiguous flat maximum uses its
physical midpoint; separated equal maxima are ambiguous and give NaN.
Both extrema need finite neighbours inside the array and must enclose the
profile maximum. Missing samples outside the selected edges make an outer
penumbra unobservable, so InflectionPoint conservatively returns NaN in that
case. Missing interior samples remain supported. No smoothing or resampling
is applied.

`method="FWHM"` uses linear interpolation at half the measured maximum.
For this method, `level=0.25`, for example, selects quarter-maximum crossings.
`level` must be between zero and one and only affects FWHM. Disconnected
above-threshold regions give NaN. Both methods return the midpoint of the
selected edges as centre and their separation as width.

These are geometric estimates. Sampling density, noise, and beam shape affect
their precision. Missing edge samples are not bridged and edges are not
extrapolated beyond the measured range. Unmeasurable fields, insufficient
samples, and profiles with no positive signal produce NaN results.

## Arrays and repeated coordinates

`positions` and `doses` must be matching one-dimensional arrays describing
one profile. Positions must be finite and may be in any order. `field_edges()`
returns a NumPy array `(2,)`; `beam_center()` and `field_width()` return floats.

Coordinates and their signals are sorted together for analysis. Repeated
coordinates, such as stationary samples in tank exports, are combined by
averaging the finite readings at each coordinate. An entirely missing group
remains NaN. NaN and infinite dose values are treated as missing samples.
The input arrays and measurements are not modified.

Incorrect array shapes, mismatched sample counts, non-finite positions, or
invalid options raise `ValueError`. Empty or undersampled profiles give NaN.

`min_signal` is an optional minimum peak in the supplied signal units, with
default 0. A profile with no positive peak, or a peak below this threshold,
returns NaN. The profile functions do not compare signals between profiles.

## Selecting profiles from a time series

```python
import numpy as np
from beamio.profile import beam_center

series = acquisition.measurements[0]  # shared crossline TimeSeries Measurement
selected_frames = [0, 1, 2]
times = series.times[selected_frames]
centers = [
    beam_center(series.positions, series.doses[index])
    for index in selected_frames
]
```

Frame selection and iteration belong to the caller. The public functions have
no frame mask, time-series input, or acquisition state.

IC Profiler returns every frame as shared Measurement arrays, preserving the
native cumulative corrected counts. To analyze interval profiles, calculate
`np.diff(series.doses, axis=0, prepend=np.zeros_like(series.doses[:1]))` and
iterate over the resulting rows. Timestamp/pulse validity and relative signal
filtering belong to the caller. `min_signal` on the public profile functions
is always an absolute threshold in the supplied signal units. For a plane
time series, select a row or column explicitly, such as
`plane.doses[frame, row, :]` with `plane.axes[1]`.
