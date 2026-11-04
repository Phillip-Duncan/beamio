
# beamio

Read, write, and convert acquired or modelled radiotherapy beam measurements
using a shared NumPy-based data model.

## Supported formats

| Format | Read | Write | Automatic detection |
| --- | --- | --- | --- |
| IBA myQA Accept 6 ASCII (`.asc`) | Yes | Yes | Yes |
| SNC SunSCAN/SunDOSE text (`.snctxt`) | Yes | Yes | Yes |
| SNC SunDOSE XML (`.sncxml`) | Yes | No | Yes |
| PTW Mephysto MCC water-tank scans (`.mcc`) | Yes | No | Yes |
| SNC IC PROFILER acquisitions (`.prm`, version 25 validated) | Yes | No | Yes |
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

# Detect and parse an IBA ASCII, SNC, PTW MCC, or IC Profiler export.
beam = BeamDataFactory.from_file("measurements.asc")
# Or select the reader explicitly:
# beam = IBAAccept6("measurements.asc").parse()

for measurement in beam.measurements:
    print(measurement.type, measurement.energy, measurement.field_size)
    print(measurement.positions, measurement.doses)

# Conversion copies measurements and metadata; save writes the target format.
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
normalization is applied. MCC writing, detector arrays, and diagonal/angled
MCC scans are not supported. See [format details](https://github.com/Phillip-Duncan/beamio/blob/main/docs/formats.md).

IC Profiler acquisitions use `ICProfiler(path).parse()` or automatic detection:

```python
from beamio.rfa import ICProfiler

acquisition = ICProfiler("measurement.prm").parse()
crossline, inline, positive_diagonal, negative_diagonal = acquisition.measurements
frames = acquisition.data  # all raw frames; column names are in acquisition.columns
```

The four profiles use the final cumulative frame, corrected for leakage and
array calibration using factors embedded in the PRM. Dose remains in corrected
counts without normalization. Use `frame=0` for the first cumulative frame,
or `absolute=True` to apply a valid enabled Dose Per Count calibration.
Positions preserve the detector gaps and diagonal spacing. The reader is
validated with version 25, Y Axis orientation acquisitions. Other versions
are attempted with a `UserWarning` that results may be incorrect. Structural
and data validation still apply. PRM writing is not implemented.
See [IC Profiler details](https://github.com/Phillip-Duncan/beamio/blob/main/docs/formats.md#snc-ic-profiler)
for geometry and calibration conventions.

Time-series access calculates the signal added during each update interval,
with timestamps in seconds and detector positions in millimetres:

```python
x = acquisition.time_series("X")  # x.doses[frame, detector], x.times, x.positions
signal = x.detector("X33")         # one detector across every frame
times, samples = x.times[10:20], signal[10:20]  # select a range of frames
centers = acquisition.beam_center()  # centers.times, centers.x, centers.y
fwhm = acquisition.beam_center(method="FWHM")  # optional threshold-based edges
```

Use `cumulative=True` on `time_series()` or `beam_center()` to
inspect integrated profiles. Edge, centre, and width calculations default to
`method="InflectionPoint"`, using spatial-gradient extrema. `method="FWHM"`
selects half-maximum crossings. Beam centre is the midpoint of the chosen
edges; low-signal or undetectable frames give NaN.

Profile analysis is also available for any reader or independent arrays:

```python
from beamio.profile import field_edges, field_width

scan = scans.measurements[0]  # choose a crossline, inline, or diagonal profile
edges_mm = field_edges(scan.positions, scan.doses, method="FWHM")
width_mm = field_width(scan.positions, scan.doses)
```

These helpers share the same InflectionPoint/FWHM implementation used by IC
Profiler. Measured width is the field size along the profile axis; `field_size`
holds the nominal X/Y field size. See [profile analysis](https://github.com/Phillip-Duncan/beamio/blob/main/docs/profiles.md)
for selecting profiles, repeated coordinates, and input handling.

`BeamData` provides `parse()`, `save()`, and `convert()`. Each `Measurement`
contains NumPy `positions` and `doses` arrays, a `MeasurementType`, a
`RadiationType`, energy, SSD, field size, scan endpoints, and a metadata
dictionary. The shared spatial fields use millimetres. Dose values use each
reader's documented scale; conversion does not normalize or calibrate them.

Conversion transfers the shared model and copies metadata, but format-specific
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

