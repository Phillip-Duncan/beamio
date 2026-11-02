# beamio tests

Tests follow the production area they exercise:

| Path | Coverage |
| --- | --- |
| `rfa/test_factory.py` | Format detection and extension discovery |
| `rfa/test_iba.py` | IBA parsing, numerical cleanup, writing, and round trips |
| `tps/test_raystation.py` | RayStation writing, parsing, metadata, and round trips |
| `integration/test_conversion.py` | Complete IBA-to-RayStation conversion and copy isolation |
| `test_package.py` | Installed version metadata |
| `support/measurements.py` | Shared scan cases and assertions |

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
