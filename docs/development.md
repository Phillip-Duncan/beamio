# Development

beamio uses Conventional Commits, semantic releases, Markdown documentation,
and tests organized by production area. CI checks the library and its PyPI
distribution archives.

## Environment

Use Python 3.12 or newer. Create and activate a virtual environment:

```console
python -m venv .venv
```

In PowerShell, run `.venv\Scripts\Activate.ps1`. On macOS/Linux, run
`source .venv/bin/activate`. Then:

```console
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

The `test` extra installs pytest without the release and formatting tools.
The `dev` extra includes tests, coverage, Black, Flake8, mypy, build, Twine, and
Python Semantic Release. Tool settings live in `pyproject.toml`; Black uses
a 100-character line length.

## Contributions and commits

Use a feature branch and a pull request targeting `main`. Title the PR using
Conventional Commits, for example:

```text
feat(iba): support an additional measurement field
fix(raystation): preserve an existing curve value
docs: clarify conversion behavior
```

The **Conventional Commits** workflow validates PR titles. Squash merging with
the PR title preserves that message for semantic release. `feat` produces a
minor release; `fix` and `perf` produce patch releases. Breaking changes use
`!` or a `BREAKING CHANGE:` footer. While versions are `0.x`, breaking changes
produce minor releases under the `major_on_zero = false` policy.
Routine `docs`, `test`, `build`, `ci`, `chore`, `style`, and `refactor` commits
do not create a release on their own.

Update relevant guides when changing behavior. Add numerical and format tests
beside the source area and complete conversion tests under `tests/integration`.
See [the testing guide](../tests/README.md) for fixtures and markers.

## Continuous integration

**CI** runs for branch pushes, pull requests, and manual dispatches; release
tags are excluded. The same workflow is reused by **Release** before
versioning and after checking out the resulting release tag.

The build job byte-compiles the source, builds the source archive and wheel,
runs strict Twine checks, checks archive contents and metadata, and rebuilds
and tests a wheel from the source archive. Installed-wheel tests run on
Python 3.12, 3.13, and 3.14 on Linux and Python 3.12 on Windows and macOS. An
additional Python 3.12 job tests NumPy 2.1.3, the minimum declared dependency.

```console
python -m compileall -q src
python -m pytest
python -m build
python -m twine check --strict dist/*
python tools/check_dist.py
```

Use a clean `dist/` containing exactly this version's two archives.
The optional `--tag` argument also verifies a release tag against the
project version.

Only the release job may write version commits and tags. Only publishing jobs
may request PyPI OIDC tokens. CI jobs use read-only repository permissions.
See [releasing](releasing.md) for versioning and the release workflow.
