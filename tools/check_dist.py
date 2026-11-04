"""Check release archives without importing the package or its dependencies."""

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metadata(raw, project):
    metadata = BytesParser().parsebytes(raw)
    for field, expected in (
        ("Name", project["name"]),
        ("Version", project["version"]),
        ("Requires-Python", project["requires-python"]),
        ("License-Expression", project["license"]),
        ("Description-Content-Type", "text/markdown"),
    ):
        require(metadata[field] == expected, f"Incorrect {field}: {metadata[field]!r}")
    require(metadata.get_all("License-File") == ["LICENSE"], "Missing license metadata")
    require(
        "numpy>=2.1.3" in metadata.get_all("Requires-Dist", []),
        "Missing NumPy runtime dependency",
    )


def check_archives(directory, tag):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if tag:
        require(tag == f"v{version}", f"Release tag {tag!r} must be v{version}")
    wheel = directory / f"beamio-{version}-py3-none-any.whl"
    sdist = directory / f"beamio-{version}.tar.gz"
    require(
        {p.name for p in directory.iterdir()} == {wheel.name, sdist.name},
        "dist must contain exactly this version's wheel and source archive",
    )

    source_files = sorted((ROOT / "src" / "beamio").rglob("*.py"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        expected_sources = {p.relative_to(ROOT / "src").as_posix() for p in source_files}
        require(
            {name for name in names if name.startswith("beamio/")} == expected_sources,
            "Wheel has missing or unexpected package files",
        )
        for path in source_files:
            name = path.relative_to(ROOT / "src").as_posix()
            require(archive.read(name) == path.read_bytes(), f"Stale wheel source: {name}")
        info = f"beamio-{version}.dist-info"
        check_metadata(archive.read(f"{info}/METADATA"), project)
        require(
            archive.read(f"{info}/licenses/LICENSE") == (ROOT / "LICENSE").read_bytes(),
            "Wheel license does not match LICENSE",
        )

    with tarfile.open(sdist) as archive:
        prefix = f"beamio-{version}/"
        names = {member.name for member in archive.getmembers() if member.isfile()}
        required_files = source_files + [
            ROOT / name
            for name in (
                "pyproject.toml",
                "LICENSE",
                "README.md",
                "RELEASE_NOTES.md",
            )
        ]
        required_files += sorted((ROOT / "tests").rglob("*.py"))
        required_files += sorted((ROOT / "tools").rglob("*.py"))
        required_files += sorted((ROOT / "docs").rglob("*.md"))
        required_files += sorted((ROOT / "dev" / "templates").rglob("*.j2"))
        required_files.append(ROOT / "tests" / "README.md")
        for path in required_files:
            name = prefix + path.relative_to(ROOT).as_posix()
            require(name in names, f"Source archive missing {name}")
            require(
                archive.extractfile(name).read() == path.read_bytes(),
                f"Stale source archive: {name}",
            )
        check_metadata(archive.extractfile(prefix + "PKG-INFO").read(), project)
        for name in names:
            path = PurePosixPath(name)
            require(
                path.suffix.lower()
                not in {
                    ".asc",
                    ".csv",
                    ".mcc",
                    ".prm",
                    ".smcm",
                    ".dcm",
                    ".snctxt",
                    ".sncxml",
                    ".pyc",
                }
                and not {".venv", "__pycache__", ".git", ".env"}.intersection(path.parts),
                f"Unexpected local/data artifact in source archive: {name}",
            )

    print(f"Validated {wheel.name} and {sdist.name} (license: {project['license']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag", default="", help="Expected release tag, e.g. v0.1.0")
    args = parser.parse_args()
    try:
        check_archives(args.dist_dir, args.tag)
    except (
        ValueError,
        OSError,
        KeyError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as error:
        parser.exit(1, f"Release check failed: {error}\n")


if __name__ == "__main__":
    main()
