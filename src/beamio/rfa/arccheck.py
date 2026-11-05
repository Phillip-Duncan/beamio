"""ArcCHECK SNC Patient measured/merged text exports on their unwrapped grid."""

from __future__ import annotations

import copy
from pathlib import Path
import re
import warnings
import zlib

import numpy as np

from beamio.common.beamdata import BeamData
from beamio.common.measurement import AcquisitionType, Measurement, MeasurementType, RadiationType
from beamio.rfa._snc_patient_grid import read_grids, write_grids
from beamio.rfa._arccheck_movie import is_acm, is_acml, read_acm, read_acml, movie_output

_KEY = "ArcCHECK"
_MODEL = re.compile(r"(?:\d+[A-Za-z0-9-]*\s+)?ArcCHECK(?:-[A-Za-z0-9]+)?(?:\s+.*)?", re.I)
_ENERGY = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:MV|X)?\s*(FFF|FF)?", re.I)
_CORRECTIONS = "Measurement Corrections"
# Revision E exports use 68 at interpolated sites and 69 at physical detectors.
_INTERPOLATED_FLAG = 68


def _cells(line):
    return [cell.strip() for cell in line.rstrip("\t ").split("\t")]


def _preamble(lines):
    nonblank = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]
    if nonblank and nonblank[0][1].startswith("** WARNING!"):
        nonblank = nonblank[1:]
    if len(nonblank) < 5:
        raise ValueError("Incomplete ArcCHECK preamble.")
    entries = nonblank[:5]
    version, revision, kind, manufacturer, model = (line for _, line in entries)
    revision_cells = _cells(revision)
    if (
        not re.fullmatch(r"\d+(?:\.\d+){1,3}", version)
        or len(revision_cells) != 2
        or revision_cells[0] != "File Revision:"
        or not revision_cells[1]
        or kind not in ("Measured File", "Merged File")
        or manufacturer != "Sun Nuclear Corporation"
        or not _MODEL.fullmatch(model)
    ):
        raise ValueError("Expected an ArcCHECK measured or merged text-file preamble.")
    metadata = {
        "Software Version": version,
        "File Revision": revision_cells[1],
        "File Type": kind,
        "Manufacturer": manufacturer,
        "Model": model,
    }
    return metadata, tuple(i for i, _ in entries)


def is_arccheck_header(text):
    """Identify a complete preamble, including files whose extension is not .txt."""
    try:
        _preamble(text.splitlines())
    except ValueError:
        return False
    return True


def _fields(line):
    cells = _cells(line)
    if not cells[0].endswith(":"):
        return []
    # Date's second Time label is part of the timestamp, distinct from the
    # acquisition duration's Time field. Keep that timestamp row intact.
    if cells[0] == "Applied Heterogeneity:" and "Applied Out Of Field:" in cells:
        index = cells.index("Applied Out Of Field:")
        return [
            (cells[0][:-1], "\t".join(cells[1:index]).rstrip("\t")),
            (cells[index][:-1], "\t".join(cells[index + 1 :])),
        ]
    return [(cells[0][:-1], "\t".join(cells[1:]))]


def _header(lines):
    metadata, positions = _preamble(lines)
    heading = next(
        (i for i in range(positions[-1] + 1, len(lines)) if _cells(lines[i]) == ["Ycm", "ROW"]),
        None,
    )
    if heading is None:
        raise ValueError("ArcCHECK text file has no dose tables.")
    index = heading - 1
    while index > positions[-1] and not lines[index].strip():
        index -= 1
    if index <= positions[-1] or _fields(lines[index]):
        raise ValueError("ArcCHECK first dose table requires a name.")
    metadata["Header"] = {}
    target = metadata["Header"]
    for line in lines[positions[-1] + 1 : index]:
        if line.strip() == _CORRECTIONS:
            if _CORRECTIONS in metadata:
                raise ValueError("Duplicate ArcCHECK measurement-corrections section.")
            target = metadata[_CORRECTIONS] = {}
        else:
            for key, value in _fields(line):
                if not key or key in target:
                    raise ValueError(f"Duplicate or empty ArcCHECK header field: {key!r}.")
                target[key] = value
    return metadata, index


def _integer(value, label, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid ArcCHECK {label}.") from None
    if (
        not np.isfinite(number)
        or number % 1
        or number < 1
        or (maximum is not None and number > maximum)
    ):
        raise ValueError(f"ArcCHECK {label} must be a positive integer within the grid.")
    return int(number)


def _dimensions(metadata):
    header = metadata["Header"]
    nrows = _integer(header.get("Rows"), "Rows")
    ncols = _integer(header.get("Cols"), "Cols")
    _integer(header.get("CAX X"), "CAX X", ncols)
    _integer(header.get("CAX Y"), "CAX Y", nrows)
    return nrows, ncols


def _energy(value):
    match = _ENERGY.fullmatch(value)
    if match is None:
        return None
    energy = float(match[1])
    if not np.isfinite(energy) or energy <= 0:
        return None
    return energy, (match[2] or "").upper()


def _fill_grid_gaps(values, x, y, missing):
    """Fill row/column-bracketed gaps without altering supplied samples."""
    result = np.asarray(values, dtype=float).copy()
    known = ~missing.copy()
    # The row pass followed by the column pass gives bilinear interpolation
    # on the rectangular detector lattice of the native merged exports.
    for samples, valid, positions in ((result, known, x), (result.T, known.T, y)):
        for row, available in zip(samples, valid):
            indices = np.flatnonzero(available)
            if len(indices) < 2:
                continue
            gaps = np.flatnonzero(~available)
            gaps = gaps[(gaps > indices[0]) & (gaps < indices[-1])]
            if not len(gaps):
                continue
            coords, signal = positions[indices], row[indices]
            if coords[0] > coords[-1]:
                coords, signal = coords[::-1], signal[::-1]
            row[gaps] = np.interp(positions[gaps], coords, signal)
            available[gaps] = True
    return result, missing & known


def _header_output(metadata, native):
    lines = list(native["Header Lines"])
    _, positions = _preamble(lines)
    replacements = (
        metadata["Software Version"],
        f"File Revision:\t{metadata['File Revision']}",
        metadata["File Type"],
        metadata["Manufacturer"],
        metadata["Model"],
    )
    for index, value in zip(positions, replacements):
        if not isinstance(value, str) or any(c in value for c in "\r\n"):
            raise ValueError("ArcCHECK preamble values must be single-line strings.")
        lines[index] = value
    output = lines[: positions[-1] + 1]
    target = metadata["Header"]
    seen = set()

    def render(fields):
        cells = []
        for key, value in fields:
            if (
                not isinstance(key, str)
                or not key
                or any(c in key for c in ":\r\n\t")
                or any(c in str(value) for c in "\r\n")
            ):
                raise ValueError("ArcCHECK header fields must have single-line keys and values.")
            cells.extend((f"{key}:", str(value)))
        return "\t".join(cells)

    def extras():
        return [render([(key, value)]) for key, value in target.items() if key not in seen]

    for line in lines[positions[-1] + 1 :]:
        if line.strip() == _CORRECTIONS:
            output.extend(extras())
            target = metadata.get(_CORRECTIONS, {})
            seen = set()
            output.append(line)
            continue
        fields = _fields(line)
        if fields:
            seen.update(key for key, _ in fields)
            fields = [(key, target[key]) for key, _ in fields if key in target]
            if fields:
                output.append(render(fields))
        else:
            output.append(line)
    output.extend(extras())
    return output


class ArcCHECK(BeamData):
    """ArcCHECK dose maps and native helical detector movies.

    Doses retain the exported cGy scale, with axes=(axial y, unwrapped x) in
    mm. The grid is a cylindrical surface projection, not a Cartesian volume.
    Measured files use Dose Interpolated; merged files use native Dose Counts.
    By default, flag-zero empty sites are filled by bracketed linear spatial
    interpolation. Original tables and interpolation masks remain in metadata.
    parse(interpolate=False) retains the exact native arrays. Writing uses
    current Measurement.doses; generated merged sites are omitted on export.

    ACM movies expose cumulative raw counts; ACML exposes vendor-processed
    float32 cGy frames and requires a matching ACM for physical geometry.
    Movie arrays are (time, detector turn, detector column), with times in
    seconds and XYZ coordinates in mm. Native movie output retains its format.
    """

    supported_acquisition_types = frozenset({AcquisitionType.Static, AcquisitionType.TimeSeries})
    supported_measurement_types = frozenset({MeasurementType.DosePlane})
    supported_radiation_types = frozenset({None, RadiationType.PHOTON})

    def parse(self, *, interpolate: bool = True) -> ArcCHECK:
        if not isinstance(interpolate, (bool, np.bool_)):
            raise TypeError("ArcCHECK interpolate must be a boolean.")
        raw = Path(self.file_path).read_bytes()
        if is_acml(raw):
            self.file_meta, scan = read_acml(raw, self.file_path)
            self.measurements = [scan]
            return self
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            encoding = "cp1252"
            text = raw.decode(encoding)
        if is_acm(text):
            self.file_meta, scan = read_acm(raw)
            self.measurements = [scan]
            return self
        lines = text.splitlines()
        metadata, index = _header(lines)
        if metadata["File Revision"] != "E":
            warnings.warn(
                f"ArcCHECK file revision {metadata['File Revision']!r} is unvalidated "
                "(tested revision: E). Attempting to read it; results may be incorrect.",
                UserWarning,
                stacklevel=2,
            )
        if metadata["Model"] != "1220 ArcCHECK":
            warnings.warn(
                f"ArcCHECK model {metadata['Model']!r} has not been validated with "
                "native samples; attempting its exported grid schema.",
                UserWarning,
                stacklevel=2,
            )
        nrows, ncols = _dimensions(metadata)
        source = "Dose Counts" if metadata["File Type"] == "Merged File" else "Dose Interpolated"
        geometry, tables, checksum = read_grids(lines[index:], nrows, ncols, _KEY, source)
        if checksum is not None:
            raw_lines = raw.splitlines(keepends=True)
            footer = next(i for i in range(len(raw_lines) - 1, -1, -1) if raw_lines[i].strip())
            if zlib.crc32(b"".join(raw_lines[:footer])) != int(checksum, 16):
                raise ValueError("ArcCHECK checksum does not match file contents.")
        x, y, rows, columns = geometry
        native = {
            "Header Lines": tuple(lines[:index]),
            "Encoding": encoding,
            "Tables": tables,
            "Grid": {"X": x.copy(), "Y": y.copy(), "Rows": rows, "Columns": columns},
            "Checksum": checksum,
        }
        metadata[_KEY] = native
        scan = Measurement()
        scan.type = MeasurementType.DosePlane
        scan.doses = tables[source]
        scan.axes = (y, x)
        scan.startpoint = (float(x[0]), float(y[0]), 0.0)
        scan.endpoint = (float(x[-1]), float(y[-1]), 0.0)
        scan.radiation_type = RadiationType.PHOTON
        scan.metadata = {
            "DoseUnit": "cGy",
            "Interpolation": (
                "none (native sparse grid)" if source == "Dose Counts" else "vendor-exported"
            ),
            _KEY: {
                "Source Table": source,
                "Coordinate System": "unwrapped cylindrical surface (mm)",
            },
        }
        flags = tables.get("Data Flags")
        if flags is not None and np.any(flags == 0):
            # Zero dose alone does not mean missing: measured zero readings
            # and vendor-interpolated zeros have nonzero native flags.
            missing = (flags == 0) & (tables[source] == 0)
            # Keep validity independently of the mutable signal, including
            # supplied nonzero values whose original flags were zero.
            native["Missing Mask"] = missing
            if interpolate and missing.any():
                processed, filled = _fill_grid_gaps(tables[source], x, y, missing)
                if filled.any():
                    scan.doses = processed
                    native["Interpolation Mask"] = filled
                    scan.metadata["Interpolation"] = (
                        "vendor-exported; " if source == "Dose Interpolated" else ""
                    ) + "linear row/column gap fill; bracketed sites only; no extrapolation"
        label = metadata.get(_CORRECTIONS, {}).get("Energy", "")
        energy = _energy(label)
        if energy is not None:
            scan.energy = energy[0]
            if energy[1]:
                scan.metadata["FluenceMode"] = energy[1]
        elif label:
            warnings.warn(
                f"ArcCHECK energy label {label!r} is unrecognized; retained in metadata.",
                UserWarning,
                stacklevel=2,
            )
        scan.validate()
        self.file_meta, self.measurements = metadata, [scan]
        return self

    def save(self, output_path: str) -> None:
        """Export an edited native acquisition after validating the complete output."""
        if self.file_meta.get(_KEY, {}).get("Format") in ("ACM", "ACML"):
            if len(self.measurements) != 1:
                raise ValueError("ArcCHECK movie output requires one time-series measurement.")
            outputs = movie_output(self.measurements[0], self.file_meta, Path(output_path))
            for path, data in outputs.items():
                path.write_bytes(data)
            return
        if len(self.measurements) != 1:
            raise ValueError("ArcCHECK output requires one static dose-plane measurement.")
        scan = self.measurements[0]
        scan.validate()
        if (
            scan.type not in self.supported_measurement_types
            or scan.acquisition_type is not AcquisitionType.Static
            or scan.radiation_type not in self.supported_radiation_types
        ):
            raise ValueError("ArcCHECK measured writing requires a static photon dose plane.")
        if scan.metadata.get("DoseUnit") != "cGy":
            raise ValueError("ArcCHECK native output doses must be in cGy.")
        if any(value is not None for value in (scan.SSD, scan.field_size, scan.wedge_angle)):
            raise ValueError(
                "ArcCHECK measured files cannot encode SSD, field size or wedge edits."
            )
        native = self.file_meta.get(_KEY, {})
        if not {"Header Lines", "Grid", "Tables"}.issubset(native):
            raise ValueError("ArcCHECK writing requires retained native acquisition metadata.")
        y, x = scan.axes
        grid = native["Grid"]
        if (
            not np.array_equal(x, grid["X"])
            or not np.array_equal(y, grid["Y"])
            or any(p is not None and p[2] != 0 for p in (scan.startpoint, scan.endpoint))
        ):
            raise ValueError("ArcCHECK writing requires unchanged native unwrapped grid geometry.")
        if scan.coordinates is not None:
            yy, xx = np.meshgrid(y, x, indexing="ij")
            expected = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
            if not np.allclose(scan.coordinates, expected, rtol=0, atol=1e-9):
                raise ValueError("ArcCHECK coordinates must describe the native unwrapped layout.")
        metadata = {
            key: copy.deepcopy(value) for key, value in self.file_meta.items() if key != _KEY
        }
        corrections = metadata.setdefault(_CORRECTIONS, {})
        old_label = corrections.get("Energy", "")
        old_energy = _energy(old_label)
        if scan.energy is None:
            corrections["Energy"] = old_label if old_energy is None else ""
        else:
            if not np.isfinite(scan.energy) or scan.energy <= 0:
                raise ValueError("ArcCHECK energy must be finite and positive.")
            mode = scan.metadata.get("FluenceMode", "")
            if mode not in ("", "FF", "FFF"):
                raise ValueError("ArcCHECK FluenceMode must be FF, FFF or undeclared.")
            if old_energy != (scan.energy, mode):
                corrections["Energy"] = f"{scan.energy:.17g}" + ("FFF" if mode == "FFF" else "MV")
        source = "Dose Counts" if metadata["File Type"] == "Merged File" else "Dose Interpolated"
        if scan.metadata.get(_KEY, {}).get("Source Table") != source:
            raise ValueError("ArcCHECK primary source table must match the native file type.")
        signal = np.asarray(scan.doses)
        tables = {**native["Tables"], source: signal}
        if "Missing Mask" in native:
            missing = np.asarray(native["Missing Mask"])
            original = np.asarray(native["Tables"][source])
            if (
                missing.dtype.kind != "b"
                or missing.shape != scan.spatial_shape
                or original.shape != scan.spatial_shape
            ):
                raise ValueError("ArcCHECK missing-site mask must match the native dose grid.")
            if source == "Dose Counts":
                signal = signal.copy()
                # Merged files store detector readings, not a complete dose image.
                # Keep generated/unsupported sites in their original native form.
                signal[missing] = original[missing]
                tables[source] = signal
            else:
                flags = np.asarray(tables.get("Data Flags"))
                filled = np.asarray(native.get("Interpolation Mask", np.zeros_like(missing)))
                if flags.shape != scan.spatial_shape:
                    raise ValueError("ArcCHECK data flags must match the native dose grid.")
                if filled.dtype.kind != "b" or filled.shape != scan.spatial_shape:
                    raise ValueError("ArcCHECK interpolation mask must match the native dose grid.")
                # Mark supplied/completed grid sites using the native interpolated
                # flag, so intentional zero edits are not mistaken for padding.
                # Preserve genuine gaps and leave the retained source flags intact.
                supplied = ~missing | filled | (signal != 0)
                flags = flags.copy()
                flags[(flags == 0) & supplied] = _INTERPOLATED_FLAG
                tables["Data Flags"] = flags
        body = write_grids(tables, (x, y, grid["Rows"], grid["Columns"]), _KEY, source)
        header = _header_output(metadata, native)
        parsed, _ = _header(header + body)
        if _dimensions(parsed) != scan.spatial_shape:
            raise ValueError("ArcCHECK declared dimensions must match its native dose grid.")
        encoding = native.get("Encoding", "utf-8")
        if encoding not in ("utf-8", "utf-8-sig", "cp1252"):
            raise ValueError("Unsupported ArcCHECK output encoding.")
        payload = ("\r\n".join(header + body) + "\r\n").encode(encoding)
        Path(output_path).write_bytes(payload + f"{zlib.crc32(payload):08X}\r\n".encode("ascii"))
