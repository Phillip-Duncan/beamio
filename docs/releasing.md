# Versioning and releases

beamio uses Python Semantic Release with Conventional Commits on
`main`, version updates in `pyproject.toml`, tags named `v{version}`, generated
`RELEASE_NOTES.md`, and GitHub releases. The templates in
`dev/templates/release_notes` include commit bodies and links. Checked
distribution archives are published to PyPI after the release tag passes CI.

## Version policy

`pyproject.toml` is the only stored package version. Installed
`beamio.__version__` reads distribution metadata. `feat` increments the minor
version, `fix` and `perf` increment the patch version, and breaking changes
increment the minor version while the package is below 1.0. Routine maintenance
commits do not release independently. Release commits use
`chore(release): {version} [skip ci]`.

The first release, when no version tags exist, is forced to a minor bump from
0.0.0 to **0.1.0**. This avoids the older fix-only history selecting 0.0.1.
Afterward, commit history since the latest release determines the next version.
Do not manually edit versions or generated release notes for normal releases.

## One-time GitHub setup

1. Keep `main` as the default release branch. Enable Actions for this repository
   and permit the release workflow's `contents: write` permission to create
   version commits, tags, and GitHub releases.
2. Use squash merging with the PR title as the commit message so the
   **Conventional Commits** check controls the release input. Configure branch
   protection/status checks to suit the repository. If protection blocks the
   automated version commit, configure an appropriate release-bot/GitHub App
   policy before enabling releases.
3. Create environments named `pypi` and `testpypi` under
   **Settings > Environments**. Configure reviewers and deployment restrictions
   as appropriate. Configure PyPI before pushing the initial release setup to
   `main`, since eligible pushes automatically publish.

These are account/repository settings; adding workflow files does not apply
them automatically.

## One-time PyPI setup

Create accounts on [PyPI](https://pypi.org/) and
[TestPyPI](https://test.pypi.org/) and enable two-factor authentication.
On each index, add a GitHub Trusted Publisher. For a new project, use
**Your account > Publishing > Add a new pending publisher**. For an existing
project, use **Manage > Publishing**.

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project name | `beamio` | `beamio` |
| Owner | `Phillip-Duncan` | `Phillip-Duncan` |
| Repository | `beamio` | `beamio` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `pypi` | `testpypi` |

The indexes need separate publisher registrations. Pending publishers do not
reserve a name; the first successful upload creates the project. Publishing
uses GitHub OIDC, so no API token or repository secret is needed for PyPI.

See the [PyPI pending publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and the [PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

## Automatic release flow

1. Merge a conventional feature/fix PR into `main`, or manually run **Release**
   on `main` with target **release**.
2. The complete CI workflow checks the candidate source, archives, and installed
   wheels. A stale run stops if `main` has advanced beyond its tested commit.
3. Python Semantic Release selects the version, updates `pyproject.toml` and
   `RELEASE_NOTES.md`, commits, tags, and creates the GitHub release. If no
   release is needed, the publishing jobs are skipped.
4. The same CI workflow runs against that exact tag, including its new metadata.
5. The checked wheel and source archive are attached to the GitHub release and
   published to PyPI through the `pypi` environment. Approve the deployment if
   environment reviewers are configured.

A pushed tag or manually created GitHub release does not trigger publication;
the semantic release workflow owns that sequence. All jobs run in the same
workflow, so release commits created with `GITHUB_TOKEN` do not need to trigger
another workflow. If publishing fails after the tag exists, use **Re-run failed
jobs** on the original run after fixing the account/environment issue.

## TestPyPI rehearsal

In Actions, select **Release > Run workflow**, choose the candidate branch or
tag, and choose **testpypi** (the manual default). The complete CI checks run,
then the current version is uploaded to TestPyPI. This path does not update the
version, create commits/tags, or publish to PyPI. Each TestPyPI upload also needs
a unique version; rerunning an already uploaded version fails.

In a fresh virtual environment outside the checkout, install dependencies from
PyPI first, then install the rehearsal package only from TestPyPI:

```console
python -m pip install "numpy>=2.1.3"
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps "beamio==0.1.0"
python -c "import beamio; from beamio.rfa import IBAAccept6; from beamio.tps.raystation import Raystation; print(beamio.__version__)"
python -m pip check
```

Use the actual candidate version in these commands. Review a representative
conversion before releasing.

## Local checks and preview

```console
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m twine check --strict dist/*
python tools/check_dist.py --tag v0.1.0
python -m semantic_release --noop version --print
```

Use a clean `dist/` and update the example tag for later versions. The last
command previews the ordinary commit-driven version without writing or
publishing. Before the first release, include `--minor` after `version` to
preview the bootstrap policy. Do not run an actual local release merely to
check configuration.

After publication, verify `python -m pip install "beamio==<version>"` in a
fresh environment and run `python -m pip check`.
