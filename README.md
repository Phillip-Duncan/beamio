
# beamio

Read, write, and convert acquired or modelled radiotherapy beam measurements
using a shared NumPy-based data model.

## Supported formats

| Format | Read | Write | Automatic detection |
| --- | --- | --- | --- |
| IBA myQA Accept 6 ASCII (`.asc`) | Yes | Yes | Yes |
| SNC SunSCAN/SunDOSE text (`.snctxt`) | Yes | Yes | Yes |
| SNC SunDOSE XML (`.sncxml`) | Yes | No | Yes |
| PTW Mephysto MCC water-tank scans (`.mcc`) | Yes | Yes | Yes |
| SNC IC PROFILER acquisitions (`.prm`, version 25 validated) | Yes | Yes; native acquisition metadata required | Yes |
| SNC SRS MapCHECK dose maps (`.txt`) and raw movies (`.smcm`, revision A validated) | Yes | Yes; native acquisition metadata required | Yes |
| RayStation 2025+ measured-curves CSV (`.csv`, semicolon-delimited) | Yes | Yes | No; use `Raystation` directly |

## Installation

Requires Python 3.12 or newer. NumPy is installed automatically.

```console
python -m pip install beamio
```

## Read and convert beam data

```python
from beamio.rfa import BeamDataFactory, IBAAccept6
from beamio.tps.raystation import Raystation

# Detect and parse an IBA ASCII, SNC, PTW MCC, IC Profiler, or SRS MapCHECK export.
beam = BeamDataFactory.from_file("measurements.asc")
# Or select the reader explicitly:
# beam = IBAAccept6("measurements.asc").parse()

for measurement in beam.measurements:
    print(measurement.type, measurement.energy, measurement.field_size)
    print(measurement.axes, measurement.doses)

# Conversion warns and drops unsupported measurements, then copies retained
# measurements and metadata. Select frames/rows first for curve-only targets.
converted = beam.convert(Raystation)
converted.save("measured_curves.csv")

# RayStation CSV files use the explicit reader.
curves = Raystation("measured_curves.csv").parse()
```

SNC exports use the same interface:

```python
from beamio.rfa import SNCSunScan

scans = SNCSunScan("measurements.sncxml").parse()  # latest processing layer
scans.save("measurements.snctxt")
scans.convert(Raystation).save("measured_curves.csv")
```

SNC positions, field sizes, and SSD are converted to millimetres. Sample order,
duplicate positions, dose values, and measured XYZ coordinates are preserved.
XML can also select an earlier layer with `layer="raw"` or a layer number.

PTW MCC water-tank exports use `PTWMCC(path).parse()` or automatic detection.
The reader supports depth-dose, crossplane, and inplane scans, preserving
exported sample order and signal scale. Optional reference readings are stored
in `measurement.metadata["PTW Reference"]`; no reference correction or
normalization is applied. MCC writing supports the same axial scan types,
including reference readings. Detector arrays and diagonal/angled MCC scans
are not supported. See [format details](https://github.com/Phillip-Duncan/beamio/blob/main/docs/formats.md).

IC Profiler acquisitions and SRS MapCHECK maps/movies use the same shared
spatial and temporal measurement model:

```python
from beamio.rfa import ICProfiler, SRSMapCheck
from beamio.profile import beam_center

acquisition = ICProfiler("measurement.prm").parse()
x, y, pd, nd = acquisition.measurements
profile = x.doses[-1]              # cumulative calibrated counts [detector]
trace = x.doses[:, 31]             # one detector across time
times_s = x.times
centers = [beam_center(x.positions, frame) for frame in x.doses]

plane = SRSMapCheck("measurement.txt").parse().measurements[0]
dose = plane.doses                # final Dose Interpolated, [y, x]
y_mm, x_mm = plane.axes

movie = SRSMapCheck("measurement.smcm").parse().measurements[0]
image = movie.doses[10]            # interpolated raw counts [y, x]
images = movie.doses[10:20]        # [frame, y, x]
pixel_trace = movie.doses[:, 22, 22]
```

`MeasurementType` describes spatial geometry: depth, crossline, inline, or
diagonal profiles, `DosePlane`, and `DoseVolume`. `AcquisitionType.Static`
uses only spatial dimensions; `AcquisitionType.TimeSeries` adds a leading time
axis with `times` in seconds. Spatial `axes` and optional XYZ `coordinates`
remain independent of time. `positions` remains the single axis for 1D data.
Static and temporal volumes are supported by the shared model; see
[measurement arrays](https://github.com/Phillip-Duncan/beamio/blob/main/docs/measurements.md)
for construction, validation, coordinate reconstruction, and NumPy indexing.

IC Profiler returns four profile time series containing the file's cumulative,
leakage-corrected and array-calibrated counts. Every frame is retained. Dose
Per Count remains metadata for explicit caller-side scaling. Frame selection,
interval differences and cumulative calculations use NumPy; the reader takes
only the file path. Version 25 is validated, with compatibility warnings for
other versions. Detector gaps and signal scale are preserved.

SRS MapCHECK returns one static plane for measured text exports or one plane
time series for movies. Measured doses retain the vendor's exported values.
Movie samples retain cumulative raw counts, with contrast-adaptive linear
interpolation of interior gaps and linear interpolation of bracketed perimeter
gaps. No extrapolation or calibrated movie dose is reconstructed. Header,
calibration, detector and acquisition-counter metadata
remain in `file_meta`; shared signal arrays are exposed through Measurements.
See [format details](https://github.com/Phillip-Duncan/beamio/blob/main/docs/formats.md).
SRS MapCHECK additionally retains every source table and movie channel in
`file_meta["SRS MapCHECK"]` for native writing. Its final dose table aliases
the static Measurement array. PRM retains auxiliary channels, calibration
rows, definitions and ignored-detector raw counts for native writing.
Both acquisition writers preserve the shared model and require their retained
instrument metadata; they do not invent missing acquisition records.

```python
PTWMCC("measurements.mcc").parse().save("export.mcc")
acquisition.save("export.prm")
SRSMapCheck("measurement.smcm").parse().save("export.smcm")
```

The public `beamio.profile` functions accept one position/dose profile at a
time, using `InflectionPoint` edges by default or optional `FWHM`. Users select
frames, rows, and columns with NumPy and loop explicitly for analysis.
`Measurement` contains geometry and acquisition data, with no profile-analysis
methods. See [profile analysis](https://github.com/Phillip-Duncan/beamio/blob/main/docs/profiles.md).

Existing IBA, SNC tank text, and RayStation writers accept only static 1D
measurements. Planes, volumes, and time series require explicit selection of
the frame/profile to export and otherwise raise an error before writing.

Each format declares `supported_acquisition_types`, `supported_measurement_types`,
and `supported_radiation_types` as immutable class attributes. Conversion checks
all three, warns for each unsupported measurement and omits it from the result.
If none are supported, the converted object has an empty measurements list.
The source is unchanged. Unknown radiation (`None`) is explicitly allowed by
the current formats; unsupported radiation types are not substituted.

Conversion transfers compatible shared measurements and copies metadata, but format-specific
writers may use defaults or omit unsupported metadata. In particular, the
RayStation writer generates its default file header (including an export time)
rather than preserving the input header. IBA parsing removes non-finite sample
pairs and sorts and deduplicates positions. Axial start coordinates are rounded
to 0.2 mm, with values within 0.2 mm of zero set to zero; diagonal scans retain
measured XYZ samples. IBA output rounds coordinates to 0.1 mm. These operations
mean that a save is not a byte-for-byte round trip.

## Development

```console
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then run:

```console
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m twine check --strict dist/*
python tools/check_dist.py
```

The tests generate synthetic measurements in temporary directories. CI tests
the installed wheel on Linux, Windows, and macOS and rebuilds the wheel from the
source distribution. Tests follow the source areas they exercise, with complete
conversions under `tests/integration`.

beamio uses Conventional Commits and Python Semantic Release. Changes
merged to `main` pass CI before automatic versioning, release-note generation,
GitHub releases, and PyPI publishing. `pyproject.toml` is the version source;
`beamio.__version__` reads the installed distribution metadata.

## Documentation

- [Spatial and temporal measurement arrays](https://github.com/Phillip-Duncan/beamio/blob/main/docs/measurements.md)
- [Architecture and shared data model](https://github.com/Phillip-Duncan/beamio/blob/main/docs/architecture.md)
- [Format support and conversion behavior](https://github.com/Phillip-Duncan/beamio/blob/main/docs/formats.md)
- [Public profile analysis](https://github.com/Phillip-Duncan/beamio/blob/main/docs/profiles.md)
- [Development and contribution workflow](https://github.com/Phillip-Duncan/beamio/blob/main/docs/development.md)
- [Testing guide](https://github.com/Phillip-Duncan/beamio/blob/main/tests/README.md)
- [Versioning and releasing](https://github.com/Phillip-Duncan/beamio/blob/main/docs/releasing.md)
- [Release notes](https://github.com/Phillip-Duncan/beamio/blob/main/RELEASE_NOTES.md)

## License

GNU General Public License version 2 only (`GPL-2.0-only`); see
[LICENSE](https://github.com/Phillip-Duncan/beamio/blob/main/LICENSE).

