
# beamio

Read, write, and convert acquired or modelled radiotherapy beam measurements
using a shared NumPy-based data model.

## Supported formats

| Format | Read | Write | Automatic detection |
| --- | --- | --- | --- |
| IBA myQA Accept 6 ASCII (`.asc`) | Yes | Yes | Yes |
| SNC SunSCAN/SunDOSE text (`.snctxt`) | Yes | Yes | Yes |
| SNC SunDOSE XML (`.sncxml`) | Yes | No | Yes |
| RayStation 2025+ measured-curves CSV (`.csv`, semicolon-delimited) | Yes | Yes | No; use `Raystation` directly |

PTW BEAMSCAN/Mephysto (`.mcc`) support is planned.

## Installation

Requires Python 3.12 or newer. NumPy is installed automatically.

```console
python -m pip install beamio
```

## Read and convert beam data

```python
from beamio.rfa import BeamDataFactory, IBAAccept6
from beamio.tps.raystation import Raystation

# Detect and parse an IBA ASCII or SNC export.
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

`BeamData` provides `parse()`, `save()`, and `convert()`. Each `Measurement`
contains NumPy `positions` and `doses` arrays, a `MeasurementType`, a
`RadiationType`, energy, SSD, field size, scan endpoints, and a metadata
dictionary. The shared spatial fields use millimetres. Dose values retain the
input scale; conversion does not normalize or calibrate them.

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
- [Development and contribution workflow](https://github.com/Phillip-Duncan/beamio/blob/main/docs/development.md)
- [Testing guide](https://github.com/Phillip-Duncan/beamio/blob/main/tests/README.md)
- [Versioning and releasing](https://github.com/Phillip-Duncan/beamio/blob/main/docs/releasing.md)
- [Release notes](https://github.com/Phillip-Duncan/beamio/blob/main/RELEASE_NOTES.md)

## License

GNU General Public License version 2 only (`GPL-2.0-only`); see
[LICENSE](https://github.com/Phillip-Duncan/beamio/blob/main/LICENSE).

