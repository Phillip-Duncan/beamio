# beamio repository guidance

beamio is a Python 3.12+ library for radiotherapy beam data IO and conversion.
NumPy is the only runtime dependency. Source lives under `src/beamio`:
`common` owns shared models, `rfa` owns measurement-vendor formats, and `tps`
owns treatment-planning formats. Shared models must not import vendor code.

Use absolute imports, keep imports free of file IO and service startup, and
preserve existing numerical/metadata behavior unless a change explicitly
requires it. Black uses 100-character lines. The testing guide is
`tests/README.md`; tests mirror source areas, and complete conversions live in
`tests/integration`. Use synthetic data and pytest temporary directories.

```console
python -m pip install -e ".[dev]"
python -m pytest tests/rfa
python -m pytest tests/tps
python -m pytest -m integration
python -m build
python -m twine check --strict dist/*
python tools/check_dist.py
```

Use Conventional Commits for commits and PR titles. `main` uses Python
Semantic Release to update `pyproject.toml`, generate `RELEASE_NOTES.md`, tag,
and publish after CI. Do not maintain a second runtime version constant.
See `docs/development.md` and `docs/releasing.md` for the contribution and
release workflows. SNC text supports reading/writing; SNC XML supports reading
and processing-layer selection. PTW support is planned, not implemented.
