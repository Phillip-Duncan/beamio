# beamio tests

Tests follow the production area they exercise:

| Path | Coverage |
| --- | --- |
| `rfa/test_factory.py` | Text/XML detection, Unicode encodings, namespaces, and extension discovery |
| `rfa/test_iba.py` | IBA parsing, numerical cleanup, writing, and round trips |
| `rfa/test_ptw.py` | PTW MCC parsing, geometry, signals/reference data, encodings, and rejection of unsupported inputs |
| `rfa/test_icprofiler.py` | IC Profiler PRM frames, leakage/calibration, detector gaps, diagonal geometry, metadata, encodings, and validation |
| `rfa/test_icprofiler_series.py` | Native cumulative Measurements, caller-side NumPy differencing/scaling, timestamp/counter preservation, raw-table release, and profile analysis |
| `rfa/test_snc.py` | SNC text/XML parsing, layer selection, units, metadata, and text writing |
| `rfa/test_srsmapcheck.py` | Shared static/temporal dose planes, raw movies, penumbra/perimeter interpolation, measured-sample preservation, full source-record retention in metadata, headers, revision warnings, and malformed acquisitions |
| `rfa/test_native_writers.py` | MCC axial/reference exports, PRM acquisition reconstruction, complete SRS tables/channels/correction cells, CRC-32, edited signals, encoding preservation, and rejection before touching destinations |
| `common/test_measurement.py` | Static/temporal shape contracts for profiles, planes and volumes; axes, timestamps, XYZ geometry, and validation |
| `common/test_beamdata.py` | Declared format capabilities, conversion filtering/warnings, source isolation, empty results, volume-capable custom targets, and native metadata aliases |
| `common/test_profile.py` | Public single-profile helpers, signal thresholds, repeated coordinates, missing values, and validation |
| `tps/test_raystation.py` | RayStation writing, parsing, metadata, and round trips |
| `integration/test_conversion.py` | Static curve conversions, explicit array selection, volume/time-series copy isolation, and writer-owned layout checks, shared validation, and rejection without partial writes |
| `integration/test_profile.py` | Profile analysis on SNC tank, IBA, PTW MCC, and RayStation measurements |
| `test_package.py` | Installed version metadata |
| `support/measurements.py` | Shared scan cases and assertions |
| `support/snc.py` | Synthetic SNC text and XML exports |
| `support/ptw.py` | Synthetic PTW MCC exports |
| `support/icprofiler.py` | Synthetic IC Profiler PRM acquisitions |
| `support/srsmapcheck.py` | Asymmetric synthetic SRS MapCHECK dose maps and raw movies |

## Running tests

Install with `python -m pip install -e ".[dev]"` and run from the project root:

```console
python -m pytest
python -m pytest tests/rfa
python -m pytest tests/tps/test_raystation.py
python -m pytest -m integration
python -m pytest --collect-only -m ""
```

Use `python -m pytest` so named `tests.support` helpers remain importable.
Pytest uses `importlib` collection, verbose output, and short tracebacks.
The default suite excludes `slow` and `performance` cases. These markers are
reserved for future tests and can be selected explicitly with `-m slow` or
`-m performance`. Tests under `integration` and `e2e` are marked automatically
by the root conftest; they are independent of test duration.

## Fixtures and data

The shared `measurement` and `iba_text` fixtures contain synthetic samples.
Readers and writers use `tmp_path`; tests do not write generated beam files
into the checkout. No external measurements, clinical datasets, network
services, or credentials are required.

Place fixtures in the closest relevant `conftest.py`. Request fixtures through
pytest arguments. Put reusable ordinary helpers in named `tests.support`
modules rather than importing from another test or conftest module.

CI runs the suite against the installed wheel. It also rebuilds the package
from the source archive, which includes the tests and their helpers. The wheel
itself contains only the library and package metadata.
