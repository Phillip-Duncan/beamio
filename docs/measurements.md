# Spatial and temporal measurements

Readers expose the same NumPy arrays through `beam.measurements`. Spatial
geometry and temporal acquisition are independent:

```python
from beamio.common import AcquisitionType, Measurement, MeasurementType
```

`MeasurementType` includes `DepthDose`, `CrosslineProfile`, `InlineProfile`,
`DiagonalProfile`, `DosePlane`, and `DoseVolume`. Each type exposes its
`spatial_ndim`. `AcquisitionType` is `Static` or `TimeSeries`.

| Geometry | Static doses | Time-series doses | Spatial axes |
| --- | --- | --- | --- |
| Profile | `(N,)` | `(T, N)` | `(positions,)` |
| Plane | `(Ny, Nx)` | `(T, Ny, Nx)` | `(y, x)` |
| Volume | `(Nz, Ny, Nx)` | `(T, Nz, Ny, Nx)` | `(z, y, x)` |

Time is always the leading array dimension when present. `times` is a finite
NumPy array with shape `(T,)`, in seconds, for a time series; it is `None` for
static data. A one-frame time series still retains its time dimension. A
three-dimensional dose array is therefore either a static volume or a plane
time series, distinguished explicitly by the two enums.

`axes` is a tuple of one-dimensional spatial coordinate arrays in millimetres,
in dose-array order. Its lengths match `spatial_shape`, which excludes time.
The existing `positions` property is an alias for `axes[0]` on 1D measurements.
Assigning `positions` updates that axis; there is no duplicate position array.
For planes and volumes use `axes` directly; `positions` raises an error.

Spatial geometry is fixed across time. Optional measured `coordinates` has
shape `spatial_shape + (3,)`, with the last dimension containing X, Y, Z in mm.
It has no time dimension. This permits explicit non-axial/rotated geometry
without repeating coordinates for every frame. Moving spatial geometry is
outside this model's current scope.

`reconstruct_xyz()` returns three spatial arrays without a time dimension.
Explicit coordinates take precedence. Without them, regular volume axes
generate the corresponding XYZ mesh. Regular plane axes describe an XY plane
at `startpoint[2]`, or Z=0 if no startpoint is set. Supply explicit coordinates
for other plane orientations. Existing profile endpoint and diagonal
reconstruction behavior is retained.

`validate()` checks enum values, dose dimensions, matching axes, timestamps,
and coordinate shapes without copying, sorting, filtering, or recalibrating
values. Plane/volume axes must be finite and strictly monotonic, in either
direction. Profile axes may retain duplicate or unordered positions. Missing
signals remain NaN. Timestamps preserve instrument order, including clock
anomalies; validation does not require them to be sorted.

## Construct a volume

```python
import numpy as np
from beamio.common import AcquisitionType, Measurement, MeasurementType

volume = Measurement()
volume.type = MeasurementType.DoseVolume
volume.axes = (
    np.array([0.0, 5.0, 10.0]),  # z
    np.array([-10.0, 0.0]),      # y
    np.array([-5.0, 0.0, 5.0, 10.0]),  # x
)
volume.doses = np.zeros((3, 2, 4))
volume.validate()
x, y, z = volume.reconstruct_xyz()  # each shape (3, 2, 4)

# A volume time series uses the same spatial geometry:
volume.acquisition_type = AcquisitionType.TimeSeries
volume.times = np.array([0.1, 0.2])
volume.doses = np.stack((volume.doses, volume.doses + 1))
volume.validate()                 # doses shape (2, 3, 2, 4)
last_volume = volume.doses[-1]    # (z, y, x)
trace = volume.doses[:, 1, 0, 2]  # one voxel over time
```

Both static and temporal dose volumes are supported by the shared model.
Conversion preserves them when the target explicitly declares volume support.
No current built-in format declares a volume writer; converting a volume to
one warns and omits it.

## Read array acquisitions

```python
from beamio.rfa import ICProfiler, SRSMapCheck

x, y, pd, nd = ICProfiler("measurement.prm").parse().measurements
first_profile = x.doses[0]
detector_trace = x.doses[:, 31]
selected_times = x.times[10:20]
selected_profiles = x.doses[10:20]

plane = SRSMapCheck("measurement.txt").parse().measurements[0]
dose_map = plane.doses            # final exported Dose Interpolated, (y, x)
y_mm, x_mm = plane.axes

movie = SRSMapCheck("measurement.smcm").parse().measurements[0]
image = movie.doses[10]           # interpolated cumulative raw counts, (y, x)
images = movie.doses[10:20]       # (frame, y, x)
pixel_trace = movie.doses[:, 22, 22]
integrated_raw_map = movie.doses[-1]  # already accumulated, then interpolated
```

Both readers take only a file path and expose all acquired frames. IC Profiler
retains the file's native cumulative, leakage-corrected and array-calibrated
counts. Headers, calibration descriptors, detector labels, and acquisition
counters remain in `file_meta`. SRS MapCHECK retains every source table and
movie channel in `file_meta["SRS MapCHECK"]`; IC Profiler retains auxiliary
channels and ignored raw readings needed for native writing. Neither reader
has a separate `data` attribute. Acquisition-specific conventions are
described in [formats](formats.md).

Frame selection and numerical transformations use ordinary NumPy:

```python
reader = ICProfiler("measurement.prm").parse()
x = reader.measurements[0]
final_profile = x.doses[-1]
intervals = np.diff(x.doses, axis=0, prepend=np.zeros_like(x.doses[:1]))
cumulative = np.cumsum(intervals, axis=0)  # recovers the native cumulative data
factor = float(reader.file_meta["Dose Calibration"]["Dose Per Count"])
scaled = x.doses * factor               # explicit scaling when appropriate
```

Native cumulative readings are already integrated. Summing those readings
again would double-count signal; use differences for per-update values.
No acquisition modes, frame selection, absolute scaling, or interval-validity
filtering are performed by reader options. Timestamps and acquisition counters
remain available for caller-side selection/filtering. For example,
`reader.file_meta["IC Profiler"]["Pulses"]` contains cumulative pulse counts,
aligned with every Measurement's `times`; `Updates` and `Statuses` describe
the same updates. SRS MapCHECK movies use the equivalent metadata under
`file_meta["SRS MapCHECK"]`. Common array access does not require raw-table
column indices; native source labels/data remain available in metadata.

`doses` is the common signal array; its name does not imply a dose unit or
calibration. Readers document the scale and preserve it in metadata. In
particular, SRS MapCHECK movie values are raw counts, while its measured text
maps use the vendor's final dose values. Cumulative/interval and interpolation
semantics remain separate metadata from the two enums.

SRS MapCHECK movies fill missing grid sites using contrast-adaptive spatial
interpolation, preferring the flatter direction near a steep penumbra.
Bracketed perimeter gaps use linear interpolation along the available axis.
Measured detector counts remain unchanged; no extrapolation or interpolation
across time is performed. See [format details](formats.md#raw-movies) for the
rule inferred from SNC exports and its limits.

There are no vendor-specific time-series result classes, frame-selection
methods, or analysis methods on Measurement. Use NumPy indexing and pass
selected 1D arrays to [profile analysis](profiles.md) explicitly.

## Conversion and curve export

`BeamData.convert()` checks each measurement against the target's
`supported_acquisition_types`, `supported_measurement_types` and
`supported_radiation_types`. These are public immutable class attributes:

```python
from beamio.rfa import PTWMCC

print(PTWMCC.supported_acquisition_types)
print(PTWMCC.supported_measurement_types)
print(PTWMCC.supported_radiation_types)
```

Each unsupported measurement is omitted with a `UserWarning` naming its source
index, target and unsupported types. Supported measurements retain their order
and all arrays/metadata are deep-copied. The original reader is unchanged.
If every measurement is omitted, the returned target has no measurements;
file metadata is still copied. Current formats explicitly include `None` in
their radiation capabilities so undeclared radiation remains undeclared.

| Target | Acquisition types | Measurement types | Radiation types |
| --- | --- | --- | --- |
| IBA, SNC tank text, RayStation | Static | All four 1D scan types | Photon, electron, proton, undeclared |
| PTW MCC | Static | Depth, crossline, inline | Photon, electron, undeclared |
| IC Profiler | TimeSeries | Crossline, inline, diagonal | Photon, electron, proton, undeclared |
| SRS MapCHECK | Static, TimeSeries | DosePlane | Photon, undeclared |
| ArcCHECK text and movies | Static, TimeSeries | DosePlane | Photon, undeclared |
| IBA MatriXX / OmniPro | Static, TimeSeries | DosePlane | Photon, electron, proton, undeclared |

Conversion checks declared types; it does not ensure a complete native
acquisition or correct geometric/array shapes. The target's `save()` validates
those details, including required vendor metadata. A compatible enum combination
therefore does not guarantee an unrelated curve can become a native acquisition.

IBA, SNC tank text, PTW MCC, and RayStation curve writers support only static 1D
measurements. They reject planes, volumes, and time series before writing any
output. They do not flatten arrays or choose a frame/row automatically.
MCC further requires an axial depth/crossline/inline scan. PRM writes its four
native profile time series; SRS MapCHECK writes its static or temporal plane.
Both native acquisition writers require the source-specific metadata described
in [formats](formats.md). SRS movie output stores physical detector sites;
interpolated pixels are omitted from native movie output, including any edits
to those pixels, and reconstructed on reload from the physical detector values.
Measured SRS tables are retained independently: saving an edited final
dose map does not recalculate the intermediate vendor correction tables.

Select the intended spatial/time samples and assign the curve to the source
**before** calling `convert()`; unsupported planes/time series will already
have been dropped afterward. For example:

```python
from copy import deepcopy
from beamio.common import AcquisitionType, MeasurementType
from beamio.rfa import PTWMCC, SRSMapCheck

reader = SRSMapCheck("measurement.smcm").parse()
movie = reader.measurements[0]

curve = deepcopy(movie)
curve.type = MeasurementType.CrosslineProfile
curve.acquisition_type = AcquisitionType.Static
curve.times = None
curve.positions = movie.axes[1].copy()
curve.doses = movie.doses[-1, 22].copy()
curve.coordinates = movie.coordinates[22].copy()
curve.startpoint = tuple(curve.coordinates[0])
curve.endpoint = tuple(curve.coordinates[-1])
curve.validate()

# Build an independent source with only the selected curve, then convert.
selected = deepcopy(reader)
selected.measurements = [curve]
converted = selected.convert(PTWMCC)
converted.save("selected-row.mcc")
```

SRS MapCHECK coordinates remain device-local; Z=0 denotes the detector plane,
not a water-equivalent depth. Set appropriate beam/setup geometry before
exporting a device row for use as a commissioning curve.
