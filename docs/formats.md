# Formats and conversion

| Format | Reader/writer | Extension | Automatic detection |
| --- | --- | --- | --- |
| IBA myQA Accept 6 ASCII | `beamio.rfa.IBAAccept6` | `.asc` | `BeamDataFactory.from_file(path)` |
| IBA MatriXX / OmniPro OPG, OPD and OPW | `beamio.rfa.IBAMatriXX` | `.opg`, `.opd`, `.opw` | `BeamDataFactory.from_file(path)` |
| SNC SunSCAN/SunDOSE text | `beamio.rfa.SNCSunScan` | `.snctxt` | `BeamDataFactory.from_file(path)` |
| SNC SunDOSE XML (read only) | `beamio.rfa.SNCSunScan` | `.sncxml` | `BeamDataFactory.from_file(path)` |
| PTW Mephysto MCC water-tank scans | `beamio.rfa.PTWMCC` | `.mcc` | `BeamDataFactory.from_file(path)` |
| SNC IC PROFILER acquisitions (version 25 validated) | `beamio.rfa.ICProfiler` | `.prm` | `BeamDataFactory.from_file(path)` |
| SNC SRS MapCHECK dose maps and raw movies (revision A validated) | `beamio.rfa.SRSMapCheck` | `.txt`, `.smcm` | `BeamDataFactory.from_file(path)` |
| RayStation 2025+ measured-curves CSV | `beamio.tps.raystation.Raystation` | `.csv` | Use the class explicitly |

`BeamDataFactory.is_supported_file(path)` checks the extension only; `from_file`
reads file contents to identify a supported reader. Extension discovery
currently returns `['asc', 'mcc', 'opd', 'opg', 'opw', 'prm', 'smcm', 'snctxt', 'sncxml', 'txt']`.
The `.txt` extension is shared by many formats; it does not identify an SRS
MapCHECK file by itself. Content detection requires its first-line signature.

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

## IBA MatriXX / OmniPro

```python
from beamio.rfa import BeamDataFactory, IBAMatriXX

beam = BeamDataFactory.from_file("measurement.opd")
# Equivalent: IBAMatriXX("measurement.opd").parse()
for measurement in beam.measurements:
    print(measurement.acquisition_type, measurement.doses.shape)
    print(measurement.axes, measurement.times)

beam.measurements[0].doses[..., 10, 10] *= 1.01
beam.save("edited.opd")
```

The reader recognizes OPG's `<opimrtascii>` marker, OPD's binary `opdfile`
signature, and OPW's `[FileVersion]` workspace header, independently of the
extension. OPG/OPD version 4 and OPW version 1 are validated. Other versions
emit a warning and are attempted using the same schema; incompatible record
layouts still raise an error. No additional dependencies are needed.

**OPG:** each `<asciibody>` becomes a static `DosePlane`. Dose samples are
multiplied by `Data Factor` and converted from Gy/cGy/mGy to Gy;
percentage signals retain their percentage scale. `metadata["DoseUnit"]`
records `Gy` or `%`. Coordinates, SSD and field sizes use mm. Header and body
properties are retained under `file_meta["IBA MatriXX"]["Header"]` and
`measurement.metadata["IBA MatriXX"]["Body"]`, respectively.
Grid dimensions and explicit coordinate values are read from the file;
rectangular and nonuniform grids are supported without interpolation.
XY, XZ, YX, YZ, ZX and ZY planes retain their orientation and plane position.
Array axes are `(row, column)`; non-XY planes provide explicit XYZ coordinates.

OPG writing supports generic static planes and uses `Measurement.doses`,
axes, XYZ coordinates and common beam metadata. Multiple bodies must share
beam metadata, dimensions and plane orientation. Source units, scaling,
unknown properties and text encoding are retained when possible; numerical
text/whitespace are regenerated. Dose values must be finite. Writing temporal
data requires callers to select a frame and make a static Measurement first.
Wedge-angle export is not implemented. Generic input must explicitly set
`measurement.metadata["DoseUnit"]` to `Gy` or `%`; the writer does not assume
that another reader's signal values are calibrated doses in Gy.
The source energy unit is retained when radiation is unchanged or undeclared,
including when editing the numeric energy. Explicitly changing radiation uses
MV for photons and MeV for electrons/protons. Exporting an energy with
undeclared radiation requires a retained source unit; no unit is guessed.

**OPD:** datasets are decoded as length-labelled binary records, validating
their image count, grid dimensions, record bounds and CRC-32. Native dose
images (signal type 1) become `DosePlane` measurements in Gy. Contiguous
`Snap` images with matching grid and beam setup become a `TimeSeries` with
`doses[frame, row, column]` and elapsed seconds in `times`, starting at zero.
Saved `Integral`, `Sum` and other dose images remain separate static
measurements; they are not recomputed from snapshots. Snapshot signal values
are preserved as saved, without temporal summation or differencing.
OPD rows follow native storage order: the Y axis runs from its maximum to
its minimum. Inspect `axes` rather than assuming ascending coordinates.

Native image records and unknown payloads remain in
`file_meta["IBA MatriXX"]["Images"]`. Each image has its name, signal type,
grid description and ordered `(record_type, payload_bytes)` records.
`measurement.metadata["IBA MatriXX"]["ImageIndices"]` associates its frames
with these images. There is no duplicate raw array for dose images.
The high pixel bit is retained separately as `PixelFlags` when present; doses
contain the lower 31 bits after scaling. Flags are not interpreted as NaNs,
and no pixel filtering or interpolation is applied. Non-dose images, including
uncalibrated film scans and gamma results, retain their complete native
`Pixels` arrays in metadata and are not presented as calibrated dose planes.
An OPD with only non-dose images loads with an empty measurement list and a
warning; those images can still be written back.

The binary beam setup maps energy, SSD and nominal field sizes. Radiation
code 0 maps to photons, as confirmed by the supplied paired OPD/OPG exports;
other codes remain undeclared and are retained in native metadata.
OPD writing requires the complete
native dataset. It uses current `Measurement.doses`, preserving flags and all
non-dose records, then regenerates the checksum. Values are rounded to the
nearest native integer count (the retained image factor defines precision).
Nonfinite, negative or out-of-range doses are rejected. Energy, SSD and field
sizes can be edited within native precision. Existing image geometry and
acquisition types must be retained; elapsed movie times must start at zero.
Radiation/wedge editing and constructing new OPD acquisitions are unsupported.

**OPW:** a workspace links datasets and analysis results; it is not a pixel
container. Existing absolute references are used directly; relative references
are resolved against the workspace directory, accepting either slash style.
If a stored path is unavailable, the reader tries the referenced basename beside
the workspace, then `<workspace>_1.opd`, `_2.opd`, or `_res.opd` as appropriate.
Every fallback emits a warning. Missing dependencies raise an error; references
must have an OPD binary signature and cannot recursively load workspaces.
Measurement metadata identifies its `WorkspaceDataset`; all workspace
sections and child dataset metadata are retained. Saving writes a new OPW and
its companion OPDs with updated local references. All children are validated
before writing any output, although the group is not a filesystem transaction.

Validation includes the supplied older OPG/OPD/OPW files and an independent
32×32 MatriXX OPD containing 33 snapshots and two integral images. Unedited
OPD datasets round-trip byte-for-byte, including non-dose images.

### Newer MatriXX models

[IBA documents MatriXX Resolution as a 39×39 array with 6.5 mm spacing](https://helpcenter.iba-dosimetry.com/hc/en-us/articles/20045590471324-MatriXX-Resolution-Device-Description).
Its [myQA workflow still uses OPG](https://helpcenter.iba-dosimetry.com/hc/en-us/articles/18858441464604-How-to-Import-the-Reference-Dose-Plane-for-User-Uniformity-Calibration).
The reader accepts these dimensions and compatible OPG schemas, with synthetic
39×39 coverage. This does not establish support for every newer native export:
Resolution/AiR native samples have not yet been validated.
[MatriXX AiR is a newer particle-therapy array](https://www.iba-dosimetry.com/product/myqa-matrixx-air);
its proprietary acquisition format is not claimed as supported.

[IBA's myQA 2023 release bulletin](https://helpcenter.iba-dosimetry.com/hc/en-us/article_attachments/9957147256860)
also describes in-phantom rotation in OPG. Rotation fields are retained with
a warning; coordinates remain in the exported plane frame, and beamio does
not infer an unverified transformation into global coordinates. Samples or
a specification are needed to validate those rotation conventions.

The [FRED Tools OmniPro documentation](https://jasqs.github.io/FREDtools/Documentation/ImgIO.html#omnipro-files-opg-opd)
is an independent reference for OPG scaling and OPD image types. beamio uses
its own NumPy/standard-library implementation and has no FRED Tools dependency.

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

Detector-array measurements (Starcheck/Octavius), and diagonal or
angled MCC scans are not implemented. These inputs raise errors instead of
being treated as ordinary axial scans. Incomplete blocks, mismatched or
duplicate scan numbers, malformed samples, and duplicate metadata keys are
also rejected, with line and scan context where available.

The implementation was checked against the CC-Export V1.9 water-tank examples
in [pymcc](https://github.com/tbezo/pymcc). It uses beamio's own parser and adds
no runtime dependencies. Other MCC variants, including BEAMSCAN exports, have
not yet been verified.

### Writing MCC

`scans.save("export.mcc")` writes static depth, crossline and inline scans,
including explicitly constructed Measurements or conversions from other
supported curve readers. It exports the shared positions/doses and optional
`PTW Reference` column, preserves scalar native header values and scan IDs,
and updates energy, SSD, field size, wedge, modality and axial geometry from
the shared model. Missing scan IDs use one-based measurement order; explicit
IDs must be unique. Sample order, duplicate positions and non-finite signals
are retained. Non-axial motion, temporal scans, planes and volumes raise
errors. Output uses CC-Export V1.9 headers by default, UTF-8 and CRLF.

## SNC IC Profiler

```python
from beamio.rfa import BeamDataFactory, ICProfiler

acquisition = BeamDataFactory.from_file("measurement.prm")
crossline, inline, positive_diagonal, negative_diagonal = acquisition.measurements
first = crossline.doses[0]
last = crossline.doses[-1]
```

The reader is validated with version 25 IC PROFILER acquisitions in Y Axis
orientation, using UTF-8 (with or without a BOM) or Windows-1252. Other nonempty
version identifiers are attempted using the same layout and emit a `UserWarning`
identifying the version and file and noting that results may be incorrect.
Version 25 emits no compatibility warning. Missing or empty Version headers
remain errors. The TYPE table identifies
detector columns and auxiliary channels. Each Data row contains cumulative raw
counts. By default each axis is a shared `Measurement` with
`acquisition_type=AcquisitionType.TimeSeries`, `doses[T, detector]`, and
`times[T]` in seconds. Spatial positions and coordinates have no time dimension.
The reader retains every calibrated cumulative signal without summing raw
rows. Its constructor takes only a file path. Selection and signal
transformations use the shared arrays directly.

Relative dose follows the definitions embedded in these PRM files:

```text
corrected counts = (RawCount - TIMETIC * BIAS1) * Calibration
```

TIMETIC is each Data row's elapsed tick value. The BIAS1 row's own tick
value describes the background acquisition and is not subtracted from TIMETIC.
Negative signals remain negative. No normalization, reference division, or
temperature/pressure correction is applied. Ignored detectors (`IgnoreDet=1`)
keep their positions and raw counts, with NaN in `doses`; no interpolation is
performed. Absolute calibration flags and Dose Per Count remain source
metadata; the reader does not apply that scaling. Callers can multiply the
arrays explicitly when their calibration is applicable. The reference
detector has a separate factor which is not used for the profile signals.
No Gy unit is inferred from the PRM factor.

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

`file_meta` holds the original headers, plus `file_meta["IC Profiler"]` with
`Detector IDs`, `Leak Rates`, `Calibration`, `Ignored Detectors`, `Updates`,
`Pulses` and `Statuses`. Native writing also retains `Columns`, full
`Correction Rows` (including non-detector cells), `Definitions`, `Encoding`,
`Auxiliary Channels`, `Ignored Detector IDs`, and `Ignored Raw Counts`.
Auxiliary channels are arrays keyed by their native column labels. Ignored
raw counts have shape `(T, number_of_ignored_detectors)` in the named order.
Counter vectors align with Measurement.times; pulse
counts retain their native cumulative convention. These vectors are independent
copies, so they do not keep the raw parsing table alive through NumPy views.

Each Measurement contains its signal/geometry and relevant metadata:
`Cumulative`, `Dose Scale`, and its `IC Profiler` axis and detector IDs.
There are no full header copies on each Measurement or duplicate non-ignored
raw count arrays. The full raw parsing table is released after auxiliary and
ignored readings have been retained in metadata; the reader has no separate
`data` attribute. `convert()` copies
the shared measurements and metadata.

Other orientations, detector layouts/spacings, and concatenated/imported
acquisitions are not implemented. Missing calibration rows,
truncated Data rows, non-finite counts/factors, invalid ignore flags, duplicate
columns/headers, and non-cumulative update/pulse counters raise errors. UPDATE#
defines frame order; TIMETIC is retained as exported, including the small final
stop-frame decreases observed in three supplied acquisitions.
When present, Total Pulses must match the final pulse counter and Total Time
must match the final elapsed tick value within summary rounding precision
(10 microseconds plus 1 ppm). Totals are checked against the complete table
before exposing the data, detecting exports missing complete final updates.
Free-text headers retain colon-terminated values; only known compound headers
split secondary labels. Automatic detection scans complete PRM header lines,
including descriptions longer than the initial sniffing buffer.
These structural and data checks also apply to unvalidated versions; a version
warning does not suppress errors or relax geometry/calibration requirements.
The implementation was exercised against all 773 acquisitions supplied from
Energy, F & S, and Wedge60, alongside synthetic regression and conversion tests.
The automated test suite does not depend on those external directories.

### Shared time-series arrays and analysis

```python
import numpy as np
from beamio.rfa import ICProfiler
from beamio.profile import beam_center

reader = ICProfiler("measurement.prm").parse()
x = reader.measurements[0]
samples = x.doses[10:20]         # cumulative [frame, detector]
times = x.times[10:20]
trace = x.doses[:, 31]           # X33 for the standard column order
intervals = np.diff(x.doses, axis=0, prepend=np.zeros_like(x.doses[:1]))
centers = [beam_center(x.positions, frame) for frame in intervals]
cumulative = np.cumsum(intervals, axis=0)  # native cumulative signals again
factor = float(reader.file_meta["Dose Calibration"]["Dose Per Count"])
scaled = x.doses * factor        # caller-side scaling when appropriate
```

Calibration uses each raw row's TIMETIC. All cumulative readings are retained,
including beam-off updates, without masking frames by pulse count or duration.
Ignored diode signals remain NaN without removing their positions. TIMETIC is
divided by 1,000,000 for shared `times` in seconds. Exported frame order and
stop-clock anomalies are preserved.

The source already contains cumulative counts, so `cumsum` must not be
applied directly to those readings to calculate total signal. Differences
give per-update signal; the first interval recipe above uses zero before the
first exported reading. Users decide whether to include updates using shared
timestamps or the `Pulses`/`Statuses` acquisition metadata vectors. No reader options choose a frame,
change cumulative/interval representation, or apply absolute dose scaling.

Profile analysis remains in [beamio.profile](profiles.md). Select a frame
with NumPy or loop over rows. Curve writers require an explicitly constructed
static Measurement; see [measurement arrays](measurements.md) for selection
and export examples. The former vendor-specific time-series/result classes
and frame/analysis methods are replaced by these shared arrays.

### Writing PRM

```python
acquisition = ICProfiler("measurement.prm").parse()
acquisition.measurements[0].doses[:, 0] += 1
acquisition.save("edited.prm")
```

PRM output requires the four native X/Y/PD/ND time series and their retained
acquisition metadata. The writer inverts the embedded array calibration and
leakage correction to reconstruct detector counts from Measurement.doses.
Ignored raw readings, auxiliary channels, calibration cells and definitions
are retained independently and restored. Timestamps come from Measurement.times
and must resolve to integral microseconds; all axes must have identical times.
Counter vectors and auxiliary channels must match the frame count. Native
clock anomalies are retained. Totals are updated from the final time/pulse
counter. Conversion from unrelated static curves cannot fabricate this data.

Beam/setup fields must agree across the four Measurements and with the
native acquisition headers. When changing those fields, update the headers
in file_meta too. Detector IDs/geometry retain the supported native layout;
ignored IDs identify the columns in `Ignored Raw Counts`. The complete raw
parsing table is not duplicated. Output preserves the detected UTF-8,
UTF-8 BOM or CP1252 encoding and uses CRLF. Numerical output preserves
double precision; inverse calibration may introduce floating-point rounding.

## SNC SRS MapCHECK

`SRSMapCheck(path).parse()` reads model 1179 measured text exports and raw
`.smcm` movies. Automatic detection uses the first nonblank
`SRS MapCHECK Measurement File:` line and its `Measured File` or
`Raw Movie Data` label, independently of extension. Revision A is validated;
other nonempty revisions are attempted with a `UserWarning` and the same
structural checks. UTF-8, UTF-8 BOM, and CP1252 text are supported. DICOM plans
and reference dose files are not implemented.

### Measured maps

```python
from beamio.rfa import SRSMapCheck

reader = SRSMapCheck("measurement.txt").parse()
plane = reader.measurements[0]
dose = plane.doses                # vendor's Dose Interpolated, [y, x]
y_mm, x_mm = plane.axes
```

A measured file produces one `MeasurementType.DosePlane` with
`AcquisitionType.Static` and `times=None`. Its `doses` references the exported
**Dose Interpolated** table without a duplicate copy. **Interpolated** is the
normalized table in the examples and is not selected as the measurement signal. Final dose values
are not recalibrated, normalized, masked, clipped, or interpolated again.
The reader retains the numerical dose scale without inferring a unit from
unlabelled table values. Exported zeros and Data Flags remain unchanged.

The parser checks every named grid, including available background,
calibration, raw/corrected counts, Dose Counts, Data Flags and interpolated
tables. Tables require complete rows/columns, finite values, monotonic axes,
and consistent geometry. Dose Interpolated is mandatory; missing or truncated
final data raises an error instead of falling back to an intermediate table.
Every table is retained in `file_meta["SRS MapCHECK"]["Tables"]`, including
intermediate counts, calibration and flags. The final `Dose Interpolated`
array also serves as Measurement.doses, without allocating a duplicate.
`Grid` retains X/Y axes in mm and the original `Rows`/`Columns` identifiers.

Xcm/Ycm convert to millimetres, preserving table order. Supplied grids are
45 by 45, with increasing X, decreasing Y, and 1.75 mm grid increments.
`plane.doses[i, j]` corresponds to `(plane.axes[1][j], plane.axes[0][i])`.

### Raw movies

```python
reader = SRSMapCheck("measurement.smcm").parse()
movie = reader.measurements[0]
image = movie.doses[10]           # cumulative raw counts [y, x]
images = movie.doses[10:20]       # [frame, y, x]
times = movie.times[10:20]        # seconds
trace = movie.doses[:, 22, 22]
source = reader.file_meta["SRS MapCHECK"]
pulses = source["Pulses"]       # cumulative pulse count for every movie update
updates = source["Updates"]
statuses = source["Statuses"]
```

A movie produces one `DosePlane` / `TimeSeries` Measurement. `doses` is a
three-dimensional NumPy array with a leading time dimension. Original
cumulative detector counts are mapped using exported detector coordinates.
Unmeasured sites use linear X and Y neighbour estimates at their actual physical
positions. For an interior gap, compare the absolute differences between the
two measured neighbours on each axis. If those contrasts differ by more than
10% of the current frame's largest absolute detector count, use the estimate
from the flatter direction. Otherwise average the two estimates. This avoids
equally blending a steep penumbra direction with its flatter perpendicular
direction. It uses both spatial directions locally, without a global spline
fit or temporal interpolation. All estimates stay within their contributing
neighbours' range; no overshoot or smoothing of measured samples is introduced.

The movie samples are already cumulative detector totals. Each cumulative
frame is interpolated independently, so `movie.doses[-1]` is the whole
acquisition's raw-count map with interpolation performed **after detector
accumulation**. It equals interpolating the sum of the raw per-update detector
increments. Do not sum the cumulative movie frames; this would repeatedly
count earlier signal. Interpolating separate increments before summing can
give a different result because direction selection is nonlinear; the reader
does not follow that route. An additional post-sum raw map would duplicate
the final frame already available through NumPy indexing.

Perimeter gaps with two measured neighbours on their available axis use that
linear bracket. They are inside the measured extent and require no
extrapolation. This fills all 88 formerly missing perimeter sites on the
supplied 45-by-45 movie grids. A site without a measured bracket remains NaN;
newly interpolated samples are never used as neighbours to extend coverage.
Original measured samples, including negative/zero counts, are unchanged.

The directional rule is inferred from the supplied revision-A exports, rather
than a manufacturer-published specification. Applying the 10%-of-peak rule
to Corrected Counts reproduces all 14,784 interpolated interior sites in the
16 exported relative **Interpolated** tables to floating-point precision.
The absolute **Dose Interpolated** tables instead match the same directional
rule with a threshold of 10 in their exported dose units. Raw movies use the
relative rule so interpolation is independent of count scale. Their peak and
direction selection are calculated separately for each cumulative frame.
Direction selection can change between frames, so differencing interpolated
pixels need not equal interpolating per-update counts. Use physical detector
sites when an original detector time trace is required.
For negative signals, beamio uses peak magnitude. The supplied measured
exports leave unmeasured perimeter sites as zero with Data Flags=0; the movie
reader instead interpolates bracketed perimeter gaps. Measured text maps
continue to preserve their exported final tables exactly.

Movie values remain **raw counts**. Matching the relative gap-filling rule
does not reconstruct SNC Patient's calibrated dose or its final dose table.
Calibration, temperature, angular, field-size, dose-rate and couch corrections
are not reconstructed. Metadata records `Dose Scale`, `Cumulative`, and
`Interpolation` explicitly. Measurement.doses is the retained signal array;
original readings are preserved at the physical detector sites. There is no
second raw acquisition table stored on the reader.

Interpolation estimates unmeasured values; it does not provide additional
independent detector measurements or guarantee recovery of a narrow feature.
Physical detector coordinates in `file_meta["SRS MapCHECK"]` identify the
original sample sites. Sun Nuclear discusses detector spacing and sampling
limits in [Stereotactic QA Arrays: Resolution and Detector Spacing](https://www.sunnuclear.com/about/news/stereotactic-qa-arrays-resolution-and-detector-spacing).

Movie `file_meta["SRS MapCHECK"]` contains `Detector IDs`, `Detector Coordinates`
in XY mm, `Corrections`, `Updates`, `Pulses` and `Statuses`. It also retains
`Columns` and the complete numeric native `Data` table, including every
thermistor, angle, reference, EIB and other auxiliary channel. `Coordinate Rows`
and `Correction Rows` preserve all native cells, including blanks and
non-detector correction entries. Correction metadata
retains available Background, Calibration(AP), Calibration(PA), AP Open,
PA Open and DetectorFlags vectors in detector order, without applying them.
Counter vectors reference the intentionally retained source data and align
with movie.times. Detector IDs/coordinates identify the physical sample sites
within the shared spatial grid. Full native source retention is file metadata;
the reader exposes no extra data/frame attributes.

Movie TIMETIC is milliseconds, divided by 1000 for `movie.times` in seconds.
The first update can start after zero; the original time origin and frame
order are retained. Update numbers must be consecutive; pulses and ticks must
be nonnegative integral counters that do not decrease. Available Update Count,
Total Pulses and Total Time summaries must match the complete acquisition
before exposing the data.

Every movie frame is exposed through the shared Measurement arrays. Select
frames or ranges with `movie.doses[index]` / `movie.doses[start:stop]` and the
same indices on `movie.times`. Reader-specific frame selectors and grid
accessors are not provided.

#### FFF beams and comparison with measured text maps

The interpolation rule applies to flattened and FFF acquisitions. A raw movie
can nevertheless look spotty even in a smooth high-dose region, because the
individual detector sensitivities have not been corrected. Interpolation
preserves those measured detector readings; it does not smooth them away.
The companion measured text export's **Dose Interpolated** table instead uses
corrected detector signals and dose calibration. Comparing it with the final
raw movie frame therefore compares different signal processing, even after
normalizing both images to their peaks.

The supplied paired movie/text exports have identical final raw detector
counts. The FFF high-dose pattern is largely removed by applying the embedded
background and AP array calibration factors in the supplied fixed-field
examples, but this alone does not reproduce every SNC correction or establish
a general calibrated movie pipeline. SNC describes per-update background,
array, temperature, angular, dose-rate and field-size corrections followed by
accumulation in its [SNC Patient Software Reference Guide, Dose Calculation
for SRS MapCHECK](https://www.scribd.com/document/1067588820/SNC-Patient-Software-Reference-Guide-RevC).
Load the measured `.txt` explicitly when the vendor's final calibrated dose
map is needed. A second raw-count aggregate would not supply that map.

### Geometry, metadata and conversion

Spatial coordinates have shape `(Ny, Nx, 3)`, independently of the number of
frames. They remain device-local, with Z=0 denoting the detector plane rather
than a water-equivalent depth. Angles, orientation and setup shifts remain
metadata; no patient/beam coordinate transform is inferred. Measurement beam
energy comes from Measurement Info / Beam Energy, rather than earlier array
or dose calibration energies. Declared energies are photon beams; undeclared
energy, SSD, nominal field size and wedge angle remain unset.

`file_meta` preserves header sections and labels once. Measurement metadata
retains only relevant source/coordinate conventions and common signal metadata;
frame selection belongs to NumPy indexing. Full headers are not duplicated
on measurements. Optional vendor checksum footers are retained
but not verified.

Conversion retains a plane/time series only when the target declares support
for its acquisition, measurement and radiation types. Unsupported measurements
are warned about and omitted. To convert to a curve format, explicitly select
a frame and a 1D profile before conversion and provide suitable geometry. See
[measurement arrays](measurements.md) for examples, including dose volumes.

### Writing SRS MapCHECK

```python
reader = SRSMapCheck("measurement.txt").parse()
reader.measurements[0].doses *= 1.01
reader.save("edited.txt")

movie = SRSMapCheck("measurement.smcm").parse()
movie.save("export.smcm")
```

Native output requires one plane Measurement plus the retained source metadata;
it does not invent instrument calibration, intermediate counts, flags or
auxiliary acquisition channels from an unrelated dose plane. Static planes
write measured text files and temporal planes write raw movie files, according
to the acquisition type rather than the filename extension.

For measured files, Measurement.doses supplies the final **Dose Interpolated**
table. Every other table is written from `file_meta["SRS MapCHECK"]["Tables"]`.
Tables must match the grid shape. Editing the final dose does not recalculate
the vendor's raw/corrected counts, flags or other intermediate tables; those
remain independently available for explicit editing in metadata. Grid axes
come from the Measurement, and native row/column IDs remain in `Grid`.

For movies, the writer takes physical detector readings from Measurement.doses
and timestamps from Measurement.times. `Updates`, `Pulses` and `Statuses`
supply the counters; `Corrections` supplies detector correction values.
All remaining data columns and auxiliary correction/coordinate cells are
restored from `Data`, `Columns`, `Correction Rows` and `Coordinate Rows`.
The full original Data table is retained as source data; shared signal/time
arrays and the named counter/correction vectors take precedence when writing
their corresponding columns. Sample counts and totals must be consistent,
timestamps must resolve to integral milliseconds, and detector coordinates
must coincide with the Measurement axes.

Native movies have no independent values for interpolated gaps. Physical
detector edits are supported. Interpolated pixels, including any caller edits
to them, are ignored when writing the sparse native detector channels. Their
values are reconstructed from physical detector readings on reload. Saving
preserves device-local XY geometry at Z=0 and the detected
UTF-8, UTF-8 BOM or CP1252 encoding. Original `Header Lines` and `Checksum` remain
metadata for provenance; output headers are serialized from the structured
fields and CRC-32 is regenerated over all encoded CRLF bytes before its footer.

All three native writers validate and encode their complete output before
opening the destination, so validation errors leave an existing file unchanged.
Round-trip checks cover the supplied native acquisitions and generated
regression fixtures; importing generated files in the vendor applications
remains an additional interoperability check.

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
