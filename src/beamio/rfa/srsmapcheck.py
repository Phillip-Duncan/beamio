"""SRS MapCHECK measured dose maps and raw acquisition movies (file revision A)."""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import copy
import zlib

import numpy as np

from beamio.rfa._snc_patient_grid import read_grids, write_grids
from beamio.common.beamdata import BeamData
from beamio.common.measurement import AcquisitionType, Measurement, MeasurementType, RadiationType

_SIGNATURE = "SRS MapCHECK Measurement File:"
_SECTIONS = {
    "Header:",
    "Device Config",
    "Device Setup",
    "Dose Calibration",
    "Data Corrections Applied",
    "Measurement Info",
}
_MOVIE_CORRECTIONS = {
    "Background",
    "Calibration(AP)",
    "Calibration(PA)",
    "AP Open",
    "PA Open",
    "DetectorFlags",
}
_DETECTOR = re.compile(r"D[1-9][0-9]*")


def _number(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"Invalid SRS MapCHECK {name}: {value!r}") from None
    if not np.isfinite(result):
        raise ValueError(f"SRS MapCHECK {name} must be finite.")
    return result


def _integer(value: str, name: str, *, minimum: int = 1) -> int:
    result = _number(value, name)
    if result % 1 or result < minimum:
        raise ValueError(f"SRS MapCHECK {name} must be an integer >= {minimum}.")
    return int(result)


def _cells(line: str) -> list[str]:
    # Trailing tabs are padding; leading blanks align movie/table columns.
    return [cell.strip() for cell in line.rstrip("\t ").split("\t")]


def _header(lines: list[str]) -> tuple[dict, int]:
    cells = _cells(lines[0])
    if len(cells) != 2 or cells[0] != _SIGNATURE:
        raise ValueError("Expected an SRS MapCHECK Measurement File signature.")
    if cells[1] not in ("Measured File", "Raw Movie Data"):
        raise ValueError(f"Unsupported SRS MapCHECK file kind: {cells[1]!r}")
    metadata: dict = {"File Type": cells[1]}
    target = None
    for index, line in enumerate(lines[1:], 1):
        if not line.strip():
            continue
        if line.strip() in _SECTIONS:
            section = line.strip().rstrip(":")
            if section in metadata:
                raise ValueError(f"Duplicate SRS MapCHECK section: {section}")
            target = metadata[section] = {}
            continue
        cells = _cells(line)
        if not cells[0].endswith(":"):
            return metadata, index
        if target is None:
            raise ValueError("SRS MapCHECK header fields require a section.")
        key = cells[0][:-1]
        if key in target:
            raise ValueError(f"Duplicate SRS MapCHECK header field: {key}")
        if key == "Date" and "Time:" in cells[1:]:
            if "Time" in target:
                raise ValueError("Duplicate SRS MapCHECK header field: Time")
            split = cells.index("Time:")
            target[key] = "\t".join(cells[1:split])
            target["Time"] = "\t".join(cells[split + 1 :])
        else:
            target[key] = "\t".join(cells[1:])
    raise ValueError("SRS MapCHECK file contains no measurement data.")


def _axis(values: np.ndarray, name: str) -> None:
    differences = np.diff(values)
    if not np.all(np.isfinite(values)) or not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"SRS MapCHECK {name} coordinates must be finite and strictly monotonic.")


def _grids(lines: list[str], nrows: int, ncols: int):
    return read_grids(lines, nrows, ncols, "SRS MapCHECK")


def _movie(lines: list[str], info: dict, nrows: int, ncols: int):
    content = [line for line in lines if line.strip()]
    if len(content) < 4:
        raise ValueError("SRS MapCHECK movie requires coordinates, TYPE columns, and data.")
    x_cells, y_cells, labels = map(_cells, content[:3])
    if x_cells[0] != "X (mm)" or y_cells[0] != "Y (mm)" or labels[0] != "TYPE":
        raise ValueError("SRS MapCHECK movie requires X (mm), Y (mm), and TYPE headings.")
    columns = tuple(labels[1:])
    if any(not label for label in columns) or len(set(columns)) != len(columns):
        raise ValueError("SRS MapCHECK movie has empty or duplicate TYPE columns.")
    required = ("UPDATE#", "TIMETIC", "PULSES", "STATUS")
    if any(label not in columns for label in required):
        raise ValueError("SRS MapCHECK movie requires UPDATE#, TIMETIC, PULSES, and STATUS.")
    indices = [i for i, label in enumerate(columns) if _DETECTOR.fullmatch(label)]
    if not indices or len(x_cells) > len(labels) or len(y_cells) > len(labels):
        raise ValueError("SRS MapCHECK movie has missing detectors or inconsistent coordinates.")
    expected_detectors = (nrows * ncols + 1) // 2
    if len(indices) != expected_detectors or {columns[i] for i in indices} != {
        f"D{i}" for i in range(1, expected_detectors + 1)
    }:
        raise ValueError("SRS MapCHECK movie has missing or inconsistent detector identifiers.")
    try:
        coordinates = np.array(
            [
                [_number(cells[i + 1], "detector coordinate") for cells in (x_cells, y_cells)]
                for i in indices
            ]
        )
    except IndexError:
        raise ValueError("SRS MapCHECK movie has missing detector coordinates.") from None
    x = np.unique(coordinates[:, 0])
    y = np.unique(coordinates[:, 1])[::-1]
    if len(x) != ncols or len(y) != nrows or len(np.unique(coordinates, axis=0)) != len(indices):
        raise ValueError(
            "SRS MapCHECK movie detector geometry differs from declared grid dimensions."
        )
    corrections: dict[str, np.ndarray] = {}
    correction_rows: dict[str, tuple[str, ...]] = {}
    rows: list[list[float]] = []
    checksum = None
    for index, line in enumerate(content[3:]):
        cells = _cells(line)
        marker = cells[0]
        if len(cells) == 1 and re.fullmatch(r"[0-9A-Fa-f]{1,8}", marker):
            if index != len(content) - 4:
                raise ValueError("Unexpected data after SRS MapCHECK movie checksum.")
            checksum = marker
        elif marker in _MOVIE_CORRECTIONS:
            if rows or marker in corrections or len(cells) != len(labels):
                raise ValueError(f"Unexpected, duplicate, or incomplete SRS MapCHECK {marker} row.")
            corrections[marker] = np.array([_number(cells[i + 1], marker) for i in indices])
            correction_rows[marker] = tuple(cells[1:])
        elif marker == "Data:":
            if len(cells) != len(labels):
                raise ValueError("SRS MapCHECK movie Data row width differs from TYPE columns.")
            try:
                rows.append([float(value) for value in cells[1:]])
            except ValueError:
                raise ValueError("SRS MapCHECK movie Data row must be numeric.") from None
        else:
            raise ValueError(f"Unexpected SRS MapCHECK movie row: {marker!r}")
    if not rows:
        raise ValueError("SRS MapCHECK movie requires at least one Data row.")
    data = np.array(rows)
    if not np.all(np.isfinite(data)):
        raise ValueError("SRS MapCHECK movie Data values must be finite.")
    for label in required[:3]:
        values = data[:, columns.index(label)]
        if np.any(values < 0) or np.any(values % 1) or np.any(np.diff(values) < 0):
            raise ValueError(
                f"SRS MapCHECK movie {label} must be nonnegative, integral, and increasing."
            )
        if label == "UPDATE#" and np.any(np.diff(values) != 1):
            raise ValueError("SRS MapCHECK movie contains missing or duplicate updates.")
    for key, actual in (
        ("Update Count", len(data)),
        ("Total Pulses", data[-1, columns.index("PULSES")]),
    ):
        if key in info and _integer(info[key], key, minimum=0) != actual:
            raise ValueError(f"SRS MapCHECK movie {key} differs from the complete acquisition.")
    if "Total Time" in info:
        parts = info["Total Time"].split()
        if len(parts) != 2 or parts[1] != "ms":
            raise ValueError("SRS MapCHECK movie Total Time requires ms units.")
        if _number(parts[0], "Total Time") != data[-1, columns.index("TIMETIC")]:
            raise ValueError("SRS MapCHECK movie Total Time differs from final TIMETIC.")
    records = {
        "Coordinate Rows": (tuple(x_cells[1:]), tuple(y_cells[1:])),
        "Correction Rows": correction_rows,
    }
    return x, y, columns, data, indices, coordinates, corrections, checksum, records


def _interpolate_movie(frames: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Fill bracketed gaps in place, preserving every measured detector sample.

    The directional rule matches the relative Interpolated tables in the supplied
    revision-A exports: prefer the smaller neighbour contrast when the contrasts
    differ by more than 10% of the frame's peak magnitude; otherwise average the
    linear X/Y estimates. This is an empirical reconstruction, not an SNC dose
    calculation. Perimeter gaps use their single bracketed direction. No samples
    outside available brackets are extrapolated, and time is never interpolated.
    """
    wx = (x[2:] - x[1:-1]) / (x[2:] - x[:-2])
    wy = (y[2:] - y[1:-1]) / (y[2:] - y[:-2])
    if len(y) > 2 and len(x) > 2:
        threshold = 0.1 * np.nanmax(np.abs(frames), axis=(1, 2))[:, None, None]
        left, right = frames[:, 1:-1, :-2], frames[:, 1:-1, 2:]
        above, below = frames[:, :-2, 1:-1], frames[:, 2:, 1:-1]
        along_x = left * wx + right * (1 - wx)
        along_y = above * wy[:, None] + below * (1 - wy[:, None])
        contrast = np.abs(right - left) - np.abs(below - above)
        estimates = np.where(
            contrast > threshold,
            along_y,
            np.where(contrast < -threshold, along_x, 0.5 * along_x + 0.5 * along_y),
        )
        # A single available bracket is still interpolation, even when its
        # perpendicular pair is unavailable. Never use newly filled neighbours.
        np.copyto(estimates, along_x, where=np.isnan(along_y))
        np.copyto(estimates, along_y, where=np.isnan(along_x))
        interior = frames[:, 1:-1, 1:-1]
        np.copyto(interior, estimates, where=np.isnan(interior))
    for row in {0, len(y) - 1}:
        edge = frames[:, row, :]
        estimate = edge[:, :-2] * wx + edge[:, 2:] * (1 - wx)
        np.copyto(edge[:, 1:-1], estimate, where=np.isnan(edge[:, 1:-1]))
    for column in {0, len(x) - 1}:
        edge = frames[:, :, column]
        estimate = edge[:, :-2] * wy + edge[:, 2:] * (1 - wy)
        np.copyto(edge[:, 1:-1], estimate, where=np.isnan(edge[:, 1:-1]))


class SRSMapCheck(BeamData):
    """Read one shared dose-plane Measurement from an SRS MapCHECK export.

    Measured files retain the vendor's Dose Interpolated table in doses[y, x].
    Movies expose interpolated cumulative raw counts in doses[time, y, x], with
    times in seconds and axes=(y, x) in device-local mm. Interior gaps use
    contrast-adaptive linear X/Y interpolation; bracketed perimeter gaps use
    their available direction. No extrapolation or movie dose calibration is
    inferred. All movie frames are exposed without selection.

    Acquisition counters, detector geometry and correction vectors belong
    to ``file_meta["SRS MapCHECK"]`` for movies. Original headers remain in
    file_meta once. All source grids, movie channels and correction cells are
    retained in file_meta["SRS MapCHECK"] for native writing.
    File revision A is
    validated; other revisions are attempted with a UserWarning.
    """

    supported_acquisition_types = frozenset({AcquisitionType.Static, AcquisitionType.TimeSeries})
    supported_measurement_types = frozenset({MeasurementType.DosePlane})
    supported_radiation_types = frozenset({None, RadiationType.PHOTON})

    def parse(self) -> SRSMapCheck:
        path = Path(self.file_path)
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
            encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
            encoding = "cp1252"
        lines = text.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            raise ValueError("SRS MapCHECK file is empty.")
        metadata, index = _header(lines)
        native = {"Header Lines": tuple(lines[:index]), "Encoding": encoding}
        if metadata.get("Device Config", {}).get("Model") != "1179 SRS MapCHECK":
            raise ValueError("Expected SRS MapCHECK Model: 1179 SRS MapCHECK.")
        revision = metadata.get("Header", {}).get("File Revision")
        if not revision:
            raise ValueError("SRS MapCHECK requires a nonempty File Revision.")
        if revision != "A":
            warnings.warn(
                f"SRS MapCHECK file revision {revision!r} is unvalidated (tested revision: A). "
                f"Attempting to parse {str(self.file_path)!r}; results may be incorrect.",
                UserWarning,
                stacklevel=2,
            )
        info = metadata.get("Measurement Info", {})
        nrows = _integer(info.get("Row Counts in 2D", ""), "Row Counts in 2D")
        ncols = _integer(info.get("Col Counts in 2D", ""), "Col Counts in 2D")
        scan = Measurement()
        scan.type = MeasurementType.DosePlane
        if metadata["File Type"] == "Measured File":
            geometry, tables, checksum = _grids(lines[index:], nrows, ncols)
            x, y, _, _ = geometry
            native.update(
                {
                    "Tables": tables,
                    "Grid": {"X": x, "Y": y, "Rows": geometry[2], "Columns": geometry[3]},
                }
            )
            scan.doses = tables["Dose Interpolated"]
            source, scale, interpolation = "Dose Interpolated", "exported dose", "vendor-exported"
        else:
            (
                x,
                y,
                columns,
                data,
                indices,
                detector_coordinates,
                corrections,
                checksum,
                records,
            ) = _movie(lines[index:], info, nrows, ncols)
            detector_indices = np.asarray(indices, dtype=int)
            native.update(
                {
                    "Detector IDs": tuple(columns[i] for i in indices),
                    "Detector Coordinates": detector_coordinates,
                    "Corrections": corrections,
                    "Columns": columns,
                    "Data": data,
                    "Updates": data[:, columns.index("UPDATE#")],
                    "Pulses": data[:, columns.index("PULSES")],
                    "Statuses": data[:, columns.index("STATUS")],
                    **records,
                }
            )
            frames = np.full((len(data), nrows, ncols), np.nan)
            rows = np.searchsorted(-y, -detector_coordinates[:, 1])
            grid_columns = np.searchsorted(x, detector_coordinates[:, 0])
            frames[:, rows, grid_columns] = data[:, detector_indices]
            _interpolate_movie(frames, x, y)
            scan.doses = frames
            scan.acquisition_type = AcquisitionType.TimeSeries
            scan.times = data[:, columns.index("TIMETIC")] * 1e-3
            source, scale = "Raw Counts", "raw counts"
            interpolation = (
                "contrast-adaptive linear X/Y neighbours (10% of per-frame peak magnitude); "
                "linear perimeter brackets; no extrapolation"
            )
        if checksum is not None:
            metadata["Checksum"] = checksum  # Retained vendor footer, not verified.
        metadata["SRS MapCHECK"] = native
        scan.axes = (y, x)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        scan.coordinates = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
        scan.startpoint = (float(x[0]), float(y[0]), 0.0)
        scan.endpoint = (float(x[-1]), float(y[-1]), 0.0)
        scan.metadata = {
            "Dose Scale": scale,
            "Cumulative": True,
            "Interpolation": interpolation,
            "SRS MapCHECK": {
                "Source Table": source,
                "Coordinate System": "device-local detector plane (mm)",
            },
        }
        energy = info.get("Beam Energy", "").split()
        if energy:
            scan.energy = _number(energy[0], "Beam Energy")
            if scan.energy <= 0:
                raise ValueError("SRS MapCHECK Beam Energy must be positive.")
            scan.radiation_type = RadiationType.PHOTON
            if len(energy) > 1:
                scan.metadata["FluenceMode"] = " ".join(energy[1:])
        scan.validate()
        self.file_meta, self.measurements = metadata, [scan]
        return self

    def save(self, output_path: str) -> None:
        if len(self.measurements) != 1:
            raise ValueError("SRS MapCHECK output requires one dose-plane measurement.")
        scan = self.measurements[0]
        if scan.type not in self.supported_measurement_types:
            raise ValueError("SRS MapCHECK output requires a dose plane.")
        scan.validate()
        native = self.file_meta.get("SRS MapCHECK", {})
        if not native:
            raise ValueError("SRS MapCHECK writing requires retained native acquisition metadata.")
        metadata = {
            key: copy.deepcopy(value)
            for key, value in self.file_meta.items()
            if key != "SRS MapCHECK"
        }
        y, x = scan.axes
        yy, xx = np.meshgrid(y, x, indexing="ij")
        xyz = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
        if scan.coordinates is not None and not np.allclose(
            scan.coordinates, xyz, rtol=0, atol=1e-9
        ):
            raise ValueError(
                "SRS MapCHECK coordinates must describe the device-local XY grid at Z=0."
            )
        info = metadata.setdefault("Measurement Info", {})
        info["Row Counts in 2D"], info["Col Counts in 2D"] = str(len(y)), str(len(x))
        if scan.radiation_type not in self.supported_radiation_types:
            raise ValueError("SRS MapCHECK output supports photon measurements.")
        if scan.energy is None:
            info.pop("Beam Energy", None)
        else:
            if not np.isfinite(scan.energy) or scan.energy <= 0:
                raise ValueError("SRS MapCHECK Beam Energy must be finite and positive.")
            mode = scan.metadata.get("FluenceMode", "")
            info["Beam Energy"] = f"{scan.energy:.17g}" + (f"\t{mode}" if mode else "")
        if scan.acquisition_type is AcquisitionType.Static:
            if "Tables" not in native or "Grid" not in native:
                raise ValueError(
                    "SRS MapCHECK measured writing requires retained tables and grid identifiers."
                )
            metadata["File Type"] = "Measured File"
            tables = dict(native["Tables"])
            tables["Dose Interpolated"] = np.asarray(scan.doses)
            grid = native["Grid"]
            rows, columns = np.asarray(grid["Rows"]), np.asarray(grid["Columns"])
            body = write_grids(tables, (x, y, rows, columns), "SRS MapCHECK")
        else:
            metadata["File Type"] = "Raw Movie Data"
            body = _movie_output(scan, native, info)
            _movie(body, info, len(y), len(x))
        header = _header_output(metadata)
        # Validate the emitted header as well as the source tables before writing.
        parsed, _ = _header(header + body)
        if parsed.get("Device Config", {}).get("Model") != "1179 SRS MapCHECK" or not parsed.get(
            "Header", {}
        ).get("File Revision"):
            raise ValueError("SRS MapCHECK output requires its model and file revision headers.")
        encoding = native.get("Encoding", "utf-8")
        if encoding not in ("utf-8", "utf-8-sig", "cp1252"):
            raise ValueError("SRS MapCHECK output encoding must be UTF-8, UTF-8 BOM or CP1252.")
        payload = ("\r\n".join(header + body) + "\r\n").encode(encoding)
        checksum = f"{zlib.crc32(payload):08X}\r\n".encode("ascii")
        Path(output_path).write_bytes(payload + checksum)


def _header_output(metadata: dict) -> list[str]:
    lines = [f"{_SIGNATURE}\t{metadata['File Type']}", ""]
    for section, fields in metadata.items():
        label = "Header:" if section == "Header" else section
        if label not in _SECTIONS:
            continue
        lines.append(label if section == "Header" else "\t" + label)
        for key, value in fields.items():
            if (
                not isinstance(key, str)
                or not key
                or any(c in key for c in ":\r\n\t")
                or any(c in str(value) for c in "\r\n")
            ):
                raise ValueError(
                    "SRS MapCHECK header fields must have single-line keys and values."
                )
            if key == "Time" and "Date" in fields:
                continue
            suffix = f"\tTime:\t{fields['Time']}" if key == "Date" and "Time" in fields else ""
            lines.append(f"{key}:\t{value}{suffix}")
        lines.append("")
    return lines


def _movie_output(scan: Measurement, native: dict, info: dict) -> list[str]:
    required = {
        "Columns",
        "Data",
        "Detector IDs",
        "Detector Coordinates",
        "Coordinate Rows",
        "Correction Rows",
        "Corrections",
    }
    if not required.issubset(native):
        raise ValueError(
            "SRS MapCHECK movie writing requires retained columns, data and correction rows."
        )
    columns = tuple(native["Columns"])
    original = np.asarray(native["Data"])
    if (
        original.dtype.kind not in "biuf"
        or original.shape != (len(scan.times), len(columns))
        or not np.isfinite(original).all()
    ):
        raise ValueError("Retained SRS MapCHECK movie data must match times and columns.")
    # Measurement edits must not inherit an integer or lower-precision source
    # dtype. Keep the retained native metadata unchanged.
    data = original.astype(np.float64, copy=True)
    ids = tuple(native["Detector IDs"])
    if (
        len(set(columns)) != len(columns)
        or len(set(ids)) != len(ids)
        or any(label not in columns for label in ids)
    ):
        raise ValueError("SRS MapCHECK detector IDs must identify unique movie columns.")
    xy = np.asarray(native["Detector Coordinates"])
    if xy.shape != (len(ids), 2) or xy.dtype.kind not in "biuf" or not np.isfinite(xy).all():
        raise ValueError("SRS MapCHECK detector coordinates must match detector IDs.")
    y, x = scan.axes
    rows = np.searchsorted(-y, -xy[:, 1])
    cols = np.searchsorted(x, xy[:, 0])
    if (
        np.any(rows >= len(y))
        or np.any(cols >= len(x))
        or not np.allclose(x[cols], xy[:, 0], rtol=0, atol=1e-9)
        or not np.allclose(y[rows], xy[:, 1], rtol=0, atol=1e-9)
    ):
        raise ValueError(
            "Movie detector coordinates must coincide with the measurement axes (X increasing, Y decreasing)."
        )
    indices = [columns.index(label) for label in ids]
    values = np.asarray(scan.doses)[:, rows, cols]
    if not np.isfinite(values).all():
        raise ValueError("SRS MapCHECK physical detector counts must be finite.")
    data[:, indices] = values
    # Interpolated pixels are not native detector channels. Their values,
    # including caller edits, are intentionally omitted from movie output.
    ticks = np.asarray(scan.times) * 1e3
    if not np.allclose(ticks, np.rint(ticks), rtol=0, atol=1e-6):
        raise ValueError("SRS MapCHECK timestamps must resolve to integral milliseconds.")
    data[:, columns.index("TIMETIC")] = np.rint(ticks)
    for label, key in (("UPDATE#", "Updates"), ("PULSES", "Pulses"), ("STATUS", "Statuses")):
        vector = np.asarray(native.get(key))
        if vector.dtype.kind not in "biuf" or vector.shape != (len(data),):
            raise ValueError(f"SRS MapCHECK {key} must match movie frames.")
        data[:, columns.index(label)] = vector
    info.update(
        {
            "Update Count": str(len(data)),
            "Total Pulses": f"{data[-1, columns.index('PULSES')]:.17g}",
            "Total Time": f"{data[-1, columns.index('TIMETIC')]:.17g}\tms",
        }
    )
    if any(
        not isinstance(cell, str) or any(c in cell for c in "\r\n\t")
        for cells in native["Coordinate Rows"]
        for cell in cells
    ):
        raise ValueError("SRS MapCHECK coordinate cells must be single-line values.")
    coordinate_rows = [list(row) for row in native["Coordinate Rows"]]
    if len(coordinate_rows) != 2:
        raise ValueError("SRS MapCHECK movie requires two coordinate rows.")
    for axis, cells in enumerate(coordinate_rows):
        cells.extend([""] * (len(columns) - len(cells)))
        if len(cells) != len(columns):
            raise ValueError("SRS MapCHECK coordinate row width differs from columns.")
        for index, value in zip(indices, xy[:, axis]):
            cells[index] = f"{value:.17g}"
    lines = [
        "\t".join([label, *cells]) for label, cells in zip(("X (mm)", "Y (mm)"), coordinate_rows)
    ]
    lines.append("\t".join(["TYPE", *columns]))
    for label, original_cells in native["Correction Rows"].items():
        cells = list(original_cells)
        if any(not isinstance(cell, str) or any(c in cell for c in "\r\n\t") for cell in cells):
            raise ValueError("SRS MapCHECK correction cells must be single-line values.")
        factors = np.asarray(native["Corrections"].get(label))
        if (
            len(cells) != len(columns)
            or factors.dtype.kind not in "biuf"
            or factors.shape != (len(ids),)
            or not np.isfinite(factors).all()
        ):
            raise ValueError(f"SRS MapCHECK {label} correction row is inconsistent.")
        for index, value in zip(indices, factors):
            cells[index] = f"{value:.17g}"
        lines.append("\t".join([label, *cells]))
    lines.extend("\t".join(["Data:", *(f"{value:.17g}" for value in row)]) for row in data)
    return lines
