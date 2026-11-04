# Formats and conversion

| Format | Reader/writer | Extension | Automatic detection |
| --- | --- | --- | --- |
| IBA myQA Accept 6 ASCII | `beamio.rfa.IBAAccept6` | `.asc` | `BeamDataFactory.from_file(path)` |
| SNC SunSCAN/SunDOSE text | `beamio.rfa.SNCSunScan` | `.snctxt` | `BeamDataFactory.from_file(path)` |
| SNC SunDOSE XML (read only) | `beamio.rfa.SNCSunScan` | `.sncxml` | `BeamDataFactory.from_file(path)` |
| PTW Mephysto MCC water-tank scans (read only) | `beamio.rfa.PTWMCC` | `.mcc` | `BeamDataFactory.from_file(path)` |
| SNC IC PROFILER acquisitions (read only, version 25 validated) | `beamio.rfa.ICProfiler` | `.prm` | `BeamDataFactory.from_file(path)` |
| RayStation 2025+ measured-curves CSV | `beamio.tps.raystation.Raystation` | `.csv` | Use the class explicitly |

`BeamDataFactory.is_supported_file(path)` checks the extension only; `from_file`
reads file contents to identify a supported reader. Extension discovery
currently returns `['asc', 'mcc', 'prm', 'snctxt', 'sncxml']`.

## IBA Accept

The reader recognizes file markers such as `:MSR`, measurement metadata lines
beginning with `%`, and samples written as `= X Y Z Dose`. It maps known
metadata into the shared fields and retains remaining values in `metadata`.

The existing parsing behavior removes sample pairs containing NaN or infinity,
sorts positions, and removes duplicates while keeping the first corresponding
dose. Axial start coordinates are rounded to 0.2 mm and values within 0.2 mm of
zero are set to zero. Diagonal scans retain measured XYZ samples and use the
same signed profile projection as SNC, including off-axis profiles. Their
coordinates and doses remain aligned through filtering and deduplication.
The writer reconstructs XYZ coordinates and rounds them to 0.1 mm. Metadata
and numerical cleanup mean that files are not byte-for-byte round trips.

## SNC SunSCAN / SunDOSE

```python
from beamio.rfa import SNCSunScan

scans = SNCSunScan("measurements.snctxt").parse()
scans = SNCSunScan("measurements.sncxml").parse()
raw = SNCSunScan("measurements.sncxml", layer="raw").parse()
earlier = SNCSunScan("measurements.sncxml", layer=2).parse()
scans.save("processed.snctxt")
```

The text reader uses `BEGIN DOSE TABLE` samples and ignores the separate raw
electrometer telemetry table. Summary and scan-header fields populate the
shared model, with remaining fields retained in metadata. Column headers
define coordinate units; native centimetres, field sizes, and SSD are converted
to millimetres. Dose percentages keep their original scale. Parsing preserves
sample order, repeated positions, non-finite dose values, and XYZ samples;
non-finite coordinates are rejected.

Text exports must include `FILE HEADER`, a nonempty `File Version`, and a
nonnegative `File Scan Count`. Both text and XML exports validate their declared
scan count; an empty dataset must explicitly declare zero scans.

Photon rate types map to `FluenceMode` (`Standard`, `FFF`, or `SRS`) for
conversion to RayStation and back to SNC's `Rate Type` when writing text.
Unsupported nonempty photon rate types raise an error rather than defaulting
to a standard beam.

Automatic XML detection respects the encoding declaration/BOM and namespaces,
including UTF-8 and UTF-16 exports. The XML reader processes one scan at a time
and selects the highest
`LayerNumber` by default. `layer="raw"` selects the `RawData` processing layer;
a positive integer selects that layer number. An unavailable layer raises an
error. Readings follow their `Sequence` values. Native XML coordinates and SSD
are in centimetres; field sizes use the exported `MeasurementUnit`.
`measurement.metadata["SNC XML"]` contains scan properties, field/energy/wedge
properties, layer descriptions, and the selected layer's metadata. This API
does not load the underlying raw electrometer channels.

Measured XYZ samples are available as `measurement.coordinates` and through
`reconstruct_xyz()`. This preserves the small off-axis movements in raw layers.
Diagonal `positions` are signed projections onto the profile axis, oriented
toward positive X (positive Y for a vertical profile), independently of scan
direction. Update or clear `coordinates` when editing scan geometry.

The writer exports a processed `.snctxt` dose table for each measurement,
updating common metadata from the shared fields. It does not recreate raw
telemetry or XML processing history. XML writing is not supported. Text output
is tested by reading it back with beamio; re-import into SunDOSE has not yet
been verified.

## PTW Mephysto MCC

```python
from beamio.rfa import PTWMCC
from beamio.tps.raystation import Raystation

scans = PTWMCC("measurements.mcc").parse()
reference = scans.measurements[0].metadata.get("PTW Reference")
scans.convert(Raystation).save("measured_curves.csv")
```

The reader accepts `BEGIN_SCAN_DATA` exports with numbered `BEGIN_SCAN` blocks,
key/value metadata, and `BEGIN_DATA` sample tables. Supported curve types are
`PDD`, `CROSSPLANE_PROFILE`, and `INPLANE_PROFILE`. Photon (`MODALITY=X`) and
electron (`MODALITY=EL`) beams are mapped to the shared model; photon `FILTER`
values `FF`, `FFF`, and `SRS` map to `FluenceMode`. Input may use UTF-8 (with or
without a BOM) or Windows-1252 encoding.

Spatial values are already in millimetres. Field size is ordered crossplane/X,
then inplane/Y. Profile geometry combines the measured positions, `SCAN_DEPTH`,
and the perpendicular `SCAN_OFFAXIS_*` offset; depth scans use both off-axis
offsets. Absent offsets default to zero. Collimator offsets are retained as
metadata and do not translate the sample coordinates. Nonzero collimator
offsets also populate `metadata["FieldsizeCorners[mm]"]` when both field widths
are available, preserving displaced jaw edges during RayStation conversion.
Update or clear that corner metadata when editing field geometry. Missing
optional beam fields remain unset; profiles require a scan depth. Axis
declarations, when present, must use crossplane X and inplane Y.

Sample order, duplicate positions, and non-finite signal values are preserved;
non-finite positions are rejected. The second data column becomes `doses` at
its original scale, including exports in arbitrary units (`MEAS_UNIT=A.U.`).
An optional third column is retained as a NumPy array in `metadata["PTW Reference"]`.
The reader does not divide by reference readings, interpolate, normalize, or
calibrate signals. Scan header values remain in `metadata`, the scan number is
available as `metadata["PTW Scan Number"]`, and file header values are stored in
`file_meta`.

Automatic detection checks the first nonblank line for `BEGIN_SCAN_DATA` before
other text signatures, so format names in scan comments do not change the reader.

MCC writing, detector-array measurements (Starcheck/Octavius), and diagonal or
angled MCC scans are not implemented. These inputs raise errors instead of
being treated as ordinary axial scans. Incomplete blocks, mismatched or
duplicate scan numbers, malformed samples, and duplicate metadata keys are
also rejected, with line and scan context where available.

The implementation was checked against the CC-Export V1.9 water-tank examples
in [pymcc](https://github.com/tbezo/pymcc). It uses beamio's own parser and adds
no runtime dependencies. Other MCC variants, including BEAMSCAN exports, have
not yet been verified.

## SNC IC Profiler

```python
from beamio.rfa import BeamDataFactory, ICProfiler

acquisition = BeamDataFactory.from_file("measurement.prm")
crossline, inline, positive_diagonal, negative_diagonal = acquisition.measurements
first = ICProfiler("measurement.prm", frame=0).parse()
absolute = ICProfiler("measurement.prm", absolute=True).parse()
```

The reader is validated with version 25 IC PROFILER acquisitions in Y Axis
orientation, using UTF-8 (with or without a BOM) or Windows-1252. Other nonempty
version identifiers are attempted using the same layout and emit a `UserWarning`
identifying the version and file and noting that results may be incorrect.
Version 25 emits no compatibility warning. Missing or empty Version headers
remain errors. The TYPE table identifies
detector columns and auxiliary channels. Each Data row contains cumulative raw
counts, so the default profiles use the final row, without summing the frames.
`frame` selects a cumulative frame by Python index, including negative indices;
it does not select an interval or subtract the preceding frame.

Relative dose follows the definitions embedded in these PRM files:

```text
corrected counts = (RawCount - TIMETIC * BIAS1) * Calibration
```

TIMETIC is the selected Data row's elapsed tick value. The BIAS1 row's own tick
value describes the background acquisition and is not subtracted from TIMETIC.
Negative signals remain negative. No normalization, reference division, or
temperature/pressure correction is applied. Ignored detectors (`IgnoreDet=1`)
keep their positions and raw counts, with NaN in `doses`; no interpolation is
performed. `absolute=True` multiplies corrected counts by the acquisition's
Dose Per Count and requires its Absolute Calibration flag to be true and the
factor positive. The reference detector's separate calibration is not used.
The PRM factor defines the absolute output scale; beamio does not assume Gy.

The four measurements are X/crossline, Y/inline, PD/positive diagonal, and
ND/negative diagonal, in that order. The exported detector IDs define a
65-position grid centred on ID 33 with 5 mm X/Y steps. X, PD, and ND omit
IDs 32 and 34; Y contains all 65. These gaps are retained, giving profile
lengths of 63, 65, 63, and 63. PD coordinates have Y=X, and ND coordinates
have Y=-X; their signed positions use distance along the diagonal, with
5*sqrt(2) mm steps away from the central gaps. Columns and samples keep their
exported order. [Sun Nuclear's specifications](https://www.sunnuclear.com/products/ic-profiler)
describe the array geometry and primary/diagonal spacing.

Coordinates use the exported axes. Orientation and gantry/collimator angles
remain metadata; no room-coordinate rotation, field-centering shift, or
projection to isocentre is applied. Z is the declared Buildup value converted
to millimetres, without adding an assumed inherent detector buildup or wedge
depth. This is a nominal setup depth; quad/double wedge acquisitions retain
their measured signals and are not converted into water-tank depth scans or
derived beam-energy metrics. SSD is converted from its declared cm/mm units.
Collimator jaws give the X/Y widths and `FieldsizeCorners[mm]` metadata for
asymmetric RayStation exports. Photon, FFF, electron, proton, and Undefined
beam metadata are accepted; Undefined energy or modality remains unset.
Active static/dynamic wedge angles and beam angles are mapped to shared fields;
a Wedge type of None gives zero wedge angle even if a stale angle is present.

Acquisition-level attributes retain the source data:

| Attribute | Contents |
| --- | --- |
| `columns` | TYPE column names, excluding the TYPE cell itself |
| `data` | All numeric frames, with columns matching `columns`; includes sensors and reference readings |
| `detector_ids` | X/Y/PD/ND detector names in exported column order |
| `leak_rates`, `calibration`, `ignored_detectors` | Arrays aligned with `detector_ids` |
| `file_meta` | Header fields with nested dictionaries for named sections; repeated calibration field names remain separate |

Each measurement retains independent header metadata under `IC Profiler Header`
and its axis, detector IDs, ignore mask, raw counts, correction factors, frame
index/count, selected TIMETIC, and dose scale under `IC Profiler`. `convert()`
copies the measurements and their metadata; the reader's full acquisition
`data` array remains on the source instance. Existing target writers determine
which metadata can be represented in their exports.

Other orientations, detector layouts/spacings, concatenated/imported
acquisitions, and PRM writing are not implemented. Missing calibration rows,
truncated Data rows, non-finite counts/factors, invalid ignore flags, duplicate
columns/headers, and non-cumulative update/pulse counters raise errors. UPDATE#
defines frame order; TIMETIC is retained as exported, including the small final
stop-frame decreases observed in three supplied acquisitions.
When present, Total Pulses must match the final pulse counter and Total Time
must match the final elapsed tick value within summary rounding precision
(10 microseconds plus 1 ppm). Totals are checked against the complete table
before selecting a frame, detecting exports missing complete final updates.
Free-text headers retain colon-terminated values; only known compound headers
split secondary labels. Automatic detection scans complete PRM header lines,
including descriptions longer than the initial sniffing buffer.
These structural and data checks also apply to unvalidated versions; a version
warning does not suppress errors or relax geometry/calibration requirements.
The implementation was exercised against all 773 acquisitions supplied from
Energy, F & S, and Wedge60, alongside synthetic regression and conversion tests.
The automated test suite does not depend on those external directories.

### Time series and beam centre

```python
acquisition = ICProfiler("measurement.prm").parse()
x = acquisition.time_series("X")
# Arrays: x.times[frame], x.positions[detector], x.doses[frame, detector]
signal = x.detector("X33")     # calibrated signal across every frame
times = x.times[10:20]        # timestamps for a selected range of frames
samples = signal[10:20]      # corresponding detector samples
edges = x.field_edges()       # left/right inflection points, mm
widths = x.field_width()      # distance between selected edges, mm
centers = acquisition.beam_center()
# centers.times is in seconds; centers.x and centers.y are in mm
fwhm = acquisition.beam_center(method="FWHM")
```

`time_series(axis)` accepts X, Y, PD, or ND. The result is an
`ICProfilerTimeSeries` containing calibrated `doses`, `positions`, `detector_ids`,
`times`, `durations`, `pulses`, `updates`, `statuses`, and `valid_frames`.
`series.detector(detector_id)` returns a copy of one detector's calibrated
signal as a one-dimensional array aligned with `series.times`. Detector IDs
must belong to the selected axis and physically exist; for example, X32 is
absent and raises an error. Slice detector signals or the full `doses` matrix
with ordinary NumPy indexing to select frames. Raw cumulative detector counts
are also available as `acquisition.data[:, acquisition.columns.index("X33")]`.
`times` are exported TIMETIC values divided by 1,000,000, measured from
acquisition start. The first time is usually the end of the first collection
interval. Original frame order and timestamps are retained.

The default is an interval series: subtract the preceding cumulative counts,
ticks, and pulses, then apply leakage correction and array calibration. The
first interval starts from zero counts/ticks/pulses. Dose is the signal added
during that interval, without division by its duration. `durations` are tick
differences in seconds and `pulses` are pulses added during each interval.
Nonpositive duration or no new pulses gives `valid_frames=False` and NaN doses,
including the anomalous stop-frame timestamps noted above. No neighbouring
frames are merged. Ignored detector doses remain NaN.

`cumulative=True` returns integrated profiles and cumulative pulse counts.
For cumulative data, validity requires positive elapsed time and cumulative
pulses; a small stop-frame timestamp reversal does not invalidate its integrated
profile. `durations` still describes the intervals between updates. Both modes
use every acquisition frame, independently of the reader's `frame=` selection,
and honour its `absolute=` calibration choice. Returned arrays can be edited
without changing the acquisition's raw data or measurements.

`beam_center()` returns an `ICProfilerBeamCenter` with `times`, `x`, `y`, and a
`valid_frames` mask indicating detectable centres on both axes. All edge,
centre, and width methods accept `method="InflectionPoint"` (the default) or
`method="FWHM"`. Centre is the midpoint of the selected pair of edges; width
is their separation. The returned centre result records the selected `method`.
These calculations delegate to the public, vendor-independent `beamio.profile`
module, which accepts a measurement's position/dose arrays from any reader. See
[profile analysis](profiles.md) for the single-profile API and caller-side loops.

InflectionPoint locates the strongest rising and falling spatial derivatives.
Finite differences are divided by the actual detector separation in mm and
located at interval midpoints. Isolated gradient extrema are refined using a
quadratic through the three neighbouring derivative samples. A contiguous flat
maximum uses its physical midpoint; separated equally strong maxima are
ambiguous and give NaN. Both extrema must have finite neighbours within the
array and enclose the profile maximum. Missing edge samples are not bridged.
Missing samples outside the candidate inflection edges are conservatively
treated as ambiguous, since they could conceal the true outer penumbra.
No smoothing or resampling is applied. This follows the
[derivative-extremum definition of inflection edges](https://pylinac.readthedocs.io/en/latest/field_analysis.html#inflection-derivative);
coarse detector spacing and noise limit the estimated edge precision.

FWHM locates crossings at 50% of each frame's measured maximum, with linear
interpolation on the actual detector positions. With `method="FWHM"`,
`level=0.25`, for example, selects quarter-maximum crossings instead. `level`
only affects FWHM; the result's `level` is None for InflectionPoint. Beam shape,
including FFF and wedge gradients, affects these geometric metrics; no
vendor-specific beam-centering correction is applied.

The default `min_signal=0.05` rejects frames whose positive peak is less than
5% of the strongest usable frame on that axis. Set `min_signal=0` to disable
this relative signal threshold. The same arguments are available on the
single-axis series' `beam_center()`, `field_edges()`, and `field_width()` methods.
Frames without two usable edges, with an ignored sample needed for an edge,
or with no positive peak give NaN. FWHM additionally rejects separated
above-threshold regions. No edge is extrapolated beyond the array.

## RayStation

The CSV format uses semicolons, leading file-header comments, metadata lines,
position/dose pairs, and `End` separators between curves. The reader accepts
decimal commas in numerical fields. The writer maps the shared fields and
uses schema defaults where applicable.

The writer generates a default file header and current export timestamp rather
than reproducing the source header. Energy and SSD have defaults when missing;
set them explicitly when constructing measurements. Supply a field size,
radiation type, measurement type, and startpoint for a meaningful export.
IBA output also needs an endpoint to reconstruct full coordinates. Position
and dose arrays should have equal lengths.

## Example

```python
from beamio.rfa import IBAAccept6
from beamio.tps.raystation import Raystation

source = IBAAccept6("measurements.asc").parse()
converted = source.convert(Raystation)
converted.save("measured_curves.csv")
```

The shared spatial fields use millimetres. Doses use each reader's documented scale.
Conversion copies the shared data; it does not translate every vendor-specific
metadata key. Automated tests cover depth, crossline, inline, and diagonal
profiles, including reversed scans, using synthetic samples. Writers may apply
their own coordinate precision or schema defaults; formats limited to a single
profile axis cannot retain all measured off-axis motion.
