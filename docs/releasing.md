# Versioning and releases

beamio uses Python Semantic Release to version and publish changes merged to
`main`. The workflow updates `pyproject.toml`, generates `RELEASE_NOTES.md`,
creates a GitHub release, and publishes checked distributions to PyPI.

## Version policy

Use Conventional Commits for PR titles and squash-merge messages:

| Change | Version bump |
| --- | --- |
| `fix:` or `perf:` | Patch |
| `feat:` | Minor |
| Breaking change (`!` or `BREAKING CHANGE:`) | Minor while below 1.0; major afterward |
| Routine documentation, tests, or maintenance | No release on their own |

`pyproject.toml` is the version source; `beamio.__version__` reads the installed
metadata. Versions, tags, and release notes are maintained by the workflow.
Release commits use `chore(release): {version} [skip ci]`.

## Releasing

1. Merge the change into `main` with a Conventional Commit message.
2. The **Release** workflow runs CI, selects the next version, and creates the
   version commit, `v{version}` tag, release notes, and GitHub release.
3. CI checks the exact release tag before publishing the wheel and source
   archive to PyPI and attaching them to the GitHub release.
4. Pull the generated release commit into your checkout with
   `git pull --ff-only origin main`.

To run a release manually, select **Actions > Release > Run workflow**, choose
`main`, and set the target to **release**. If no release is needed, publishing
is skipped.

If publication fails after the tag exists, fix the cause and use **Re-run
failed jobs** on the original workflow run.

## Local preview

```console
python -m semantic_release --noop version --print
```

This previews the next version without changing files or publishing. See the
[development guide](development.md#continuous-integration) for local checks.
