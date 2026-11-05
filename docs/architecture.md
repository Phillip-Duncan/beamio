# Architecture

beamio is a headless Python library with a `src` layout and one runtime
dependency, NumPy. Importing the package does not read files or start services.

| Area | Responsibility |
| --- | --- |
| `src/beamio/common/measurement.py` | Spatial/acquisition enums, NumPy arrays, shape validation, metadata, and XYZ reconstruction |
| `src/beamio/profile.py` | Public vendor-independent profile edges, centres, and widths from one X/Y profile |
| `src/beamio/common/beamdata.py` | Abstract reader/writer interface and conversion through independent copies |
| `src/beamio/rfa/iba.py` | IBA myQA Accept 6 ASCII parsing and writing |
| `src/beamio/rfa/matrixx.py` | IBA MatriXX/OmniPro OPG grids, OPD snapshots/integrals, OPW workspaces and native records |
| `src/beamio/rfa/snc.py` | SNC text parsing/writing and shared-model mapping; `_snc_xml.py` reads XML layers |
| `src/beamio/rfa/factory.py` | Content-based detection and supported-extension discovery |
| `src/beamio/rfa/ptw.py` | PTW Mephysto MCC water-tank parsing and shared-model mapping |
| `src/beamio/rfa/icprofiler.py` | IC PROFILER PRM frames, calibration, and four-axis profiles |
| `src/beamio/rfa/srsmapcheck.py` | SRS MapCHECK static/temporal plane measurements and original vendor records |
| `src/beamio/tps/raystation.py` | RayStation measured-curves CSV parsing and writing |

## Shared model

`MeasurementType` describes a 1D profile, a dose plane, or a dose volume.
`AcquisitionType` distinguishes static measurements from time series. Doses
have shape `spatial_shape` for static data or `(T, *spatial_shape)` for a time
series, with `times[T]` in seconds. Spatial axes use `(positions,)`, `(y, x)`,
or `(z, y, x)` in millimetres. `positions` aliases the single 1D axis. Optional
XYZ coordinates have shape `(*spatial_shape, 3)` without a time dimension.
`validate()` checks this contract without changing acquired values. See
[measurement arrays](measurements.md) for complete examples, including volumes.

`RadiationType` identifies photons, electrons, and protons. Optional fields
include energy, SSD, nominal field size, wedge angle, and geometry endpoints.
Source-specific values and signal/interpolation conventions remain metadata.
`reconstruct_xyz()` retains existing 1D/diagonal geometry checks, uses explicit
spatial coordinates when supplied, and reconstructs regular planes/volumes
from axis arrays otherwise.

`beamio.profile` accepts one position/dose pair and depends only on NumPy.
It never imports readers or the shared model. Callers choose frames/rows with
NumPy, filter signals as needed, and call profile functions explicitly.
Readers and Measurement have no analysis or frame-selection methods.

`BeamData` holds measurements and file-level metadata. Subclasses implement
`parse()` and `save(path)`. IC Profiler produces four profile time series by
default; SRS MapCHECK produces one static plane or one plane time series.
Both readers take only a file path and expose every acquired frame through
the same Measurement arrays. Selection and signal transformations belong to
the caller. Signals belong to Measurement.doses; source headers, detector
labels, calibration and acquisition counters remain metadata. SRS MapCHECK
retains all source tables, movie columns, correction cells and coordinate rows
in file-level metadata for native writing. Its final static dose table aliases
Measurement.doses. IC Profiler retains auxiliary channels and ignored-detector
raw counts; other raw detector counts can be reconstructed from shared corrected
signals and embedded calibration, so its full parsing table is released.
There are no vendor-specific series containers or duplicate frame accessors.
Readers expose only `file_path`, `measurements` and `file_meta`. Full headers
are not copied into each Measurement. Native column labels and acquisition
records belong in metadata, with no separate reader data container. SRS movie
counter vectors reference the intentionally retained native table. IC counter
vectors and auxiliary channels are independent copies.

Each concrete `BeamData` declares immutable `supported_acquisition_types`,
`supported_measurement_types`, and `supported_radiation_types` class attributes,
containing the shared enum members it can represent. `None` is explicitly
included where undeclared radiation metadata is supported. Empty base defaults
prevent undeclared or newly introduced types from being accepted silently.

`beam.convert(TargetClass)` constructs a new target instance, checks all three
capabilities for every measurement, and emits one `UserWarning` per omitted
measurement identifying its one-based index, target, and unsupported fields.
Remaining measurements keep their order. If none qualify, the target is empty;
an empty source converts quietly. Measurements and the complete file metadata
are deep-copied together, preserving their internal aliases without sharing
arrays with the source. Conversion does not select frames/rows, recalibrate,
normalize, or validate native acquisition completeness.
Format readers and writers translate their units
to and from the shared millimetre model. `save()` applies the target
writer's schema and defaults; format-specific metadata may not transfer. Curve
writers check their own supported measurement types and acquisition layouts
at serialization, and call Measurement.validate() for the shared array
contract. Generic model validation contains no file-format policy. Writers
buffer and encode their output before opening the destination, so rejected
measurements do not partially replace files. MCC writes generic static axial
profiles. PRM and SRS MapCHECK native writers require retained acquisition
metadata; acquisition information is never synthesized from unrelated curves.

## Adding a format

Put shared model behavior in `common`, vendor measurement readers in `rfa`, and
treatment-planning interfaces in `tps`. Shared code must not import vendor
implementations. Implement a `BeamData` subclass, declare its three capability
sets, and add corresponding tests
under the matching source area. Add complete conversions under
`tests/integration`. Extend the factory's detection rules only when a reader
has an unambiguous signature and is implemented.

## Package metadata

`pyproject.toml` owns the build configuration, dependencies, version, tool
settings, and semantic release policy. Runtime `__version__` uses installed
metadata, with a `0.0.0.dev` fallback for an uninstalled source
tree. `MANIFEST.in` includes documentation, synthetic tests, release templates,
and the GPLv2 license in the source archive. The wheel contains the package and
its distribution metadata.
