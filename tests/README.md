# beamio tests

Tests follow the production area they exercise:

| Path | Coverage |
| --- | --- |
| `rfa/test_factory.py` | Text/XML detection, Unicode encodings, namespaces, and extension discovery |
| `rfa/test_iba.py` | IBA parsing, numerical cleanup, writing, and round trips |
| `rfa/test_ptw.py` | PTW MCC parsing, geometry, signals/reference data, encodings, and rejection of unsupported inputs |
| `rfa/test_icprofiler.py` | IC Profiler PRM frames, leakage/calibration, detector gaps, diagonal geometry, metadata, encodings, and validation |
| `rfa/test_icprofiler_series.py` | Interval/cumulative series, detector signals, timing/pulse validity, inflection/FWHM selection, known moving fields, and missing edges |
| `rfa/test_snc.py` | SNC text/XML parsing, layer selection, units, metadata, and text writing |
| `common/test_measurement.py` | Diagonal reconstruction and measured XYZ sample preservation |
| `common/test_profile.py` | Public single-profile helpers, signal thresholds, repeated coordinates, missing values, and validation |
| `tps/test_raystation.py` | RayStation writing, parsing, metadata, and round trips |
| `integration/test_conversion.py` | IBA/SNC/PTW/ICProfiler/RayStation conversions and copy isolation |
| `integration/test_profile.py` | Profile analysis on SNC tank, IBA, PTW MCC, and RayStation measurements |
| `test_package.py` | Installed version metadata |
| `support/measurements.py` | Shared scan cases and assertions |
| `support/snc.py` | Synthetic SNC text and XML exports |
| `support/ptw.py` | Synthetic PTW MCC exports |
| `support/icprofiler.py` | Synthetic IC Profiler PRM acquisitions |

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
