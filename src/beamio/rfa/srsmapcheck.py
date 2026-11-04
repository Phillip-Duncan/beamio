"""SRS MapCHECK measured dose maps and raw acquisition movies (file revision A)."""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np

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
    tables: dict[str, np.ndarray] = {}
    reference = None
    index = 0
    checksum = None
    while index < len(lines):
        name = lines[index].strip()
        index += 1
        if not name:
            continue
        if re.fullmatch(r"[0-9A-Fa-f]{1,8}", name) and not any(
            line.strip() for line in lines[index:]
        ):
            checksum = name
            break
        if name in tables:
            raise ValueError(f"Duplicate SRS MapCHECK table: {name}")
        block = lines[index : index + nrows + 3]
        if len(block) != nrows + 3 or _cells(block[0]) != ["Ycm", "ROW"]:
            raise ValueError(f"SRS MapCHECK {name}: incomplete table or missing Ycm/ROW heading.")
        data = []
        ys, row_ids = [], []
        for row in block[1 : nrows + 1]:
            cells = _cells(row)
            if len(cells) != ncols + 2:
                raise ValueError(f"SRS MapCHECK {name}: row width differs from Col Counts in 2D.")
            ys.append(_number(cells[0], f"{name} Ycm") * 10)
            row_ids.append(_integer(cells[1], f"{name} ROW"))
            data.append([_number(value, name) for value in cells[2:]])
        columns, xs = _cells(block[-2]), _cells(block[-1])
        if (
            len(columns) != ncols + 2
            or columns[:2] != ["", "COL"]
            or len(xs) != ncols + 2
            or xs[:2] != ["", "Xcm"]
        ):
            raise ValueError(f"SRS MapCHECK {name}: missing or incomplete COL/Xcm coordinates.")
        column_ids = [_integer(value, f"{name} COL") for value in columns[2:]]
        if sorted(row_ids) != list(range(1, nrows + 1)) or sorted(column_ids) != list(
            range(1, ncols + 1)
        ):
            raise ValueError(
                f"SRS MapCHECK {name}: duplicate or inconsistent row/column identifiers."
            )
        x = np.array([_number(value, f"{name} Xcm") * 10 for value in xs[2:]])
        y = np.array(ys)
        _axis(x, "X")
        _axis(y, "Y")
        geometry = (x, y, np.array(row_ids), np.array(column_ids))
        if reference is not None and any(
            not np.array_equal(a, b) for a, b in zip(reference, geometry)
        ):
            raise ValueError(f"SRS MapCHECK {name}: grid geometry differs from the other tables.")
        reference = geometry
        tables[name] = np.array(data)
        index += nrows + 3
    if "Dose Interpolated" not in tables:
        raise ValueError("SRS MapCHECK measured files require a Dose Interpolated table.")
    return reference, tables, checksum


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
    return x, y, columns, data, indices, coordinates, corrections, checksum


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
    file_meta once. Raw tables and unused intermediate grids are not retained.
    File revision A is
    validated; other revisions are attempted with a UserWarning.
    """

    def parse(self) -> SRSMapCheck:
        path = Path(self.file_path)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp1252")
        lines = text.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            raise ValueError("SRS MapCHECK file is empty.")
        metadata, index = _header(lines)
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
            ) = _movie(lines[index:], info, nrows, ncols)
            detector_indices = np.asarray(indices, dtype=int)
            metadata["SRS MapCHECK"] = {
                "Detector IDs": tuple(columns[i] for i in indices),
                "Detector Coordinates": detector_coordinates,
                "Corrections": corrections,
                "Updates": data[:, columns.index("UPDATE#")].copy(),
                "Pulses": data[:, columns.index("PULSES")].copy(),
                "Statuses": data[:, columns.index("STATUS")].copy(),
            }
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
        raise NotImplementedError("SRS MapCHECK writing is not implemented.")
