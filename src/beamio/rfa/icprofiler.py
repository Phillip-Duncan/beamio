"""Sun Nuclear IC PROFILER acquisitions, validated with PRM version 25."""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import copy

import numpy as np

from beamio.common.beamdata import BeamData
from beamio.common.measurement import (
    AcquisitionType,
    Measurement,
    MeasurementType,
    RadiationType,
    diagonal_direction,
)

_SECTIONS = {
    "Profiler Setup",
    "Collector Config",
    "Machine Data",
    "Hardware Data",
    "Dose Calibration",
    "Reference Detector Dose Calibration",
    "UltraSonic Calibration",
}
_DETECTOR = re.compile(r"(X|Y|PD|ND)([1-9][0-9]*)")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_SECONDARY_FIELDS = {
    "Date": {"Time"},
    "Collector Serial": {"Revision"},
    "Beam Type": {"Energy"},
    "Rate": {"Dose"},
    "Gantry Angle": {"Collimator Angle"},
    "Bp": {"Mp", "Bt"},
    "PulseCountsDuringIdle": {
        "PulseCountsDuringMmt",
        "BeamDurationDuringIdle",
        "BeamDurationDuringMmt",
    },
}


def _number(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"Invalid IC Profiler {name}: {value!r}") from None
    if not np.isfinite(result):
        raise ValueError(f"IC Profiler {name} must be finite.")
    return result


def _length(value: str, name: str) -> float:
    parts = value.split()
    if len(parts) < 2 or parts[1] not in ("cm", "mm"):
        raise ValueError(f"IC Profiler {name} requires cm or mm units.")
    return _number(parts[0], name) * (10 if parts[1] == "cm" else 1)


def _header_line(target: dict, cells: list[str]) -> None:
    """Split known compound headers; preserve free-text values literally."""
    key = cells[0].rstrip(":")
    secondary = _SECONDARY_FIELDS.get(key, set())
    values: list[str] = []
    for cell in cells[1:]:
        if cell.endswith(":") and cell[:-1] in secondary:
            if key in target:
                raise ValueError(f"Duplicate IC Profiler header: {key}")
            target[key] = "\t".join(values)
            key, values = cell[:-1], []
        else:
            values.append(cell)
    if key in target:
        raise ValueError(f"Duplicate IC Profiler header: {key}")
    target[key] = "\t".join(values)


def _validate_totals(metadata: dict, columns: tuple[str, ...], data: np.ndarray) -> None:
    """Check available export totals against the complete acquisition table."""
    if "Total Pulses" in metadata:
        total = _number(metadata["Total Pulses"], "Total Pulses")
        if total < 0 or total % 1:
            raise ValueError("IC Profiler Total Pulses must be a nonnegative integer.")
        actual = data[-1, columns.index("PULSES")]
        if total != actual:
            raise ValueError(
                f"IC Profiler Total Pulses ({total:g}) differs from final PULSES ({actual:g}); "
                "the acquisition is incomplete or inconsistent."
            )
    if "Total Time" in metadata:
        total = _number(metadata["Total Time"], "Total Time")
        if total < 0:
            raise ValueError("IC Profiler Total Time must be nonnegative.")
        actual = data[-1, columns.index("TIMETIC")] * 1e-6
        # Native summaries have limited floating-point precision, while the
        # TIMETIC counter is integral. Allow rounding, not missing updates.
        if not np.isclose(total, actual, rtol=1e-6, atol=1e-5):
            raise ValueError(
                f"IC Profiler Total Time ({total:g}) differs from final TIMETIC ({actual:g} s); "
                "the acquisition is incomplete or inconsistent."
            )


def _read(path: Path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
        encoding = "cp1252"
    metadata: dict = {}
    corrections: dict[str, list[str]] = {}
    rows: list[list[float]] = []
    target = metadata
    columns = None
    definitions = False
    definition_lines = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        cells = [cell.strip() for cell in raw.split("\t")]
        if not any(cells):
            continue
        marker = cells[0]
        try:
            if marker == "TYPE":
                if columns is not None:
                    raise ValueError("Duplicate TYPE table.")
                columns = tuple(cells[1:])
                if len(set(columns)) != len(columns) or any(not c for c in columns):
                    raise ValueError("Empty or duplicate TYPE columns.")
            elif columns is not None:
                if marker in ("BIAS1", "Calibration", "IgnoreDet"):
                    if rows or marker in corrections:
                        raise ValueError(f"Unexpected or duplicate {marker} row.")
                    corrections[marker] = cells[1:]
                elif marker == "Data:":
                    if len(cells) != len(columns) + 1:
                        raise ValueError("Data row does not match TYPE columns.")
                    rows.append([float(cell) for cell in cells[1:]])
                else:
                    raise ValueError(f"Unexpected table row: {marker!r}")
            elif marker == "Definitions:":
                definitions = True
            elif definitions:
                definition_lines.append(raw)
            elif not definitions:
                section = raw.strip()
                if section in _SECTIONS:
                    if section in metadata:
                        raise ValueError(f"Duplicate section: {section}")
                    target = metadata[section] = {}
                else:
                    if marker == "Multi Frame:":
                        target = metadata
                    _header_line(target, cells)
        except ValueError as exc:
            raise ValueError(f"IC Profiler line {line_number}: {exc}") from exc
    if columns is None or not rows:
        raise ValueError("IC Profiler requires a TYPE table and at least one Data row.")
    metadata["IC Profiler"] = {"Definitions": tuple(definition_lines), "Encoding": encoding}
    return metadata, columns, corrections, np.asarray(rows, dtype=float)


class ICProfiler(BeamData):
    """Read four profiles from an IC PROFILER acquisition.

    Each axis is a shared Measurement with doses[time, detector] and times in
    seconds. Exported cumulative signals are leakage-corrected and array-calibrated
    in corrected counts. Every frame is retained. Dose Per Count remains metadata
    for caller-side scaling; frame selection and differencing use NumPy directly.

    Version 25 is validated. Other nonempty versions emit UserWarning and are
    attempted using the same layout and data checks.

    Acquisition counters and calibration vectors are in ``file_meta["IC Profiler"]``.
    Measurement metadata identifies its axis and detector IDs. Ignored
    detectors retain their geometry with NaN signals; no interpolation is applied.
    """

    supported_acquisition_types = frozenset({AcquisitionType.TimeSeries})
    supported_measurement_types = frozenset(
        {
            MeasurementType.CrosslineProfile,
            MeasurementType.InlineProfile,
            MeasurementType.DiagonalProfile,
        }
    )
    supported_radiation_types = frozenset(
        {None, RadiationType.PHOTON, RadiationType.ELECTRON, RadiationType.PROTON}
    )

    def parse(self) -> ICProfiler:
        metadata, columns, corrections, data = _read(Path(self.file_path))
        version = metadata.get("Version")
        if not version:
            raise ValueError("IC Profiler requires a nonempty Version header.")
        if metadata.get("Collector Config", {}).get("Collector Model") != "IC PROFILER":
            raise ValueError("Expected Collector Model: IC PROFILER.")
        if version != "25":
            warnings.warn(
                f"IC Profiler PRM version {version!r} is unvalidated (tested version: 25). "
                f"Attempting to parse {str(self.file_path)!r} using the version 25 layout; "
                "results may be incorrect.",
                UserWarning,
                stacklevel=2,
            )
        setup = metadata.get("Profiler Setup", {})
        if setup.get("Orientation") != "Y Axis":
            raise ValueError("Only IC Profiler Y Axis orientation is supported.")
        for key in ("Concatenation", "Imported Data"):
            if metadata.get(key, "false").lower() != "false":
                raise ValueError(f"IC Profiler {key} acquisitions are not supported.")
        required = ("UPDATE#", "TIMETIC", "PULSES", "STATUS")
        if any(key not in columns for key in required):
            raise ValueError("IC Profiler TYPE requires UPDATE#, TIMETIC, PULSES, and STATUS.")
        indices = np.array([i for i, label in enumerate(columns) if _DETECTOR.fullmatch(label)])
        detector_ids = tuple(columns[i] for i in indices)
        layout = {
            label: (match[1], int(match[2]))
            for label in detector_ids
            if (match := _DETECTOR.fullmatch(label)) is not None
        }
        expected = {
            "X": set(range(1, 66)) - {32, 34},
            "Y": set(range(1, 66)),
            "PD": set(range(1, 66)) - {32, 34},
            "ND": set(range(1, 66)) - {32, 34},
        }
        for axis, labels in expected.items():
            actual = {number for name, number in layout.values() if name == axis}
            if actual != labels:
                raise ValueError(f"Unsupported IC Profiler {axis} detector layout.")
        if not np.all(np.isfinite(data[:, indices])):
            raise ValueError("IC Profiler detector counts must be finite.")
        for key in required:
            values = data[:, columns.index(key)]
            if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values % 1):
                raise ValueError(f"IC Profiler {key} must contain nonnegative integers.")
            # Some version 25 stop frames have a slightly earlier TIMETIC than
            # the preceding update, while counts and pulses still accumulate.
            # UPDATE# defines frame order; preserve and use each exported tick.
            if key in ("UPDATE#", "PULSES") and np.any(
                np.diff(values) < (1 if key == "UPDATE#" else 0)
            ):
                raise ValueError(f"IC Profiler {key} must increase cumulatively.")
        _validate_totals(metadata, columns, data)
        factors = {}
        for name in ("BIAS1", "Calibration", "IgnoreDet"):
            cells = corrections.get(name, [])
            if len(cells) > len(columns) or any(i >= len(cells) or not cells[i] for i in indices):
                raise ValueError(f"IC Profiler requires {name} for every detector.")
            factors[name] = np.array([_number(cells[i], name) for i in indices])
        if np.any(factors["Calibration"] <= 0):
            raise ValueError("IC Profiler Calibration factors must be positive.")
        if not np.all(np.isin(factors["IgnoreDet"], (0, 1))):
            raise ValueError("IC Profiler IgnoreDet flags must be 0 or 1.")
        ignored = factors["IgnoreDet"].astype(bool)
        ticks = data[:, columns.index("TIMETIC")]
        times = ticks * 1e-6
        doses = (data[:, indices] - ticks[:, None] * factors["BIAS1"]) * factors["Calibration"]
        doses[:, ignored] = np.nan
        spacing = _number(metadata.get("Detector Spacing", ""), "Detector Spacing") * 10
        if spacing != 5:
            raise ValueError("Only IC Profiler 0.5 cm detector spacing is supported.")
        depth = _length(setup.get("Buildup", ""), "Buildup")
        if depth < 0:
            raise ValueError("IC Profiler Buildup must be nonnegative.")
        measurements = []
        for axis in expected:
            selected = [i for i, label in enumerate(detector_ids) if layout[label][0] == axis]
            # IDs encode the 65-position grid; 32 and 34 are physically absent on
            # three axes. Do not compact these gaps or infer spacing from counts.
            distance = (
                np.asarray(
                    [layout[detector_ids[i]][1] - 33 for i in selected],
                    dtype=np.float64,
                )
                * spacing
            )
            xyz = np.zeros((len(selected), 3))
            xyz[:, 2] = depth
            if axis == "X":
                xyz[:, 0] = distance
                kind = MeasurementType.CrosslineProfile
            elif axis == "Y":
                xyz[:, 1] = distance
                kind = MeasurementType.InlineProfile
            else:
                xyz[:, 0] = distance
                xyz[:, 1] = distance * (1 if axis == "PD" else -1)
                kind = MeasurementType.DiagonalProfile
            scan = _measurement(metadata)
            scan.type = kind
            scan.coordinates = xyz
            scan.startpoint, scan.endpoint = tuple(xyz[0]), tuple(xyz[-1])
            scan.positions = (
                xyz[:, :2] @ diagonal_direction(scan.startpoint, scan.endpoint)
                if axis in ("PD", "ND")
                else distance
            )
            scan.acquisition_type = AcquisitionType.TimeSeries
            scan.times = times
            scan.doses = doses[:, selected]
            scan.metadata["Cumulative"] = True
            scan.metadata["Dose Scale"] = "corrected counts"
            scan.metadata["IC Profiler"] = {
                "Axis": axis,
                "Detector IDs": tuple(detector_ids[i] for i in selected),
            }
            scan.validate()
            measurements.append(scan)
        metadata["IC Profiler"].update(
            {
                "Detector IDs": detector_ids,
                "Leak Rates": factors["BIAS1"],
                "Calibration": factors["Calibration"],
                "Ignored Detectors": ignored,
                # Copies avoid retaining the complete raw table through NumPy views.
                "Updates": data[:, columns.index("UPDATE#")].copy(),
                "Pulses": data[:, columns.index("PULSES")].copy(),
                "Statuses": data[:, columns.index("STATUS")].copy(),
                "Columns": columns,
                "Correction Rows": corrections,
                "Auxiliary Channels": {
                    label: data[:, i].copy()
                    for i, label in enumerate(columns)
                    if i not in indices and label not in required
                },
                "Ignored Raw Counts": data[:, indices[ignored]].copy(),
                "Ignored Detector IDs": tuple(
                    label for label, flag in zip(detector_ids, ignored) if flag
                ),
            }
        )
        self.file_meta, self.measurements = metadata, measurements
        return self

    def save(self, output_path: str) -> None:
        native = self.file_meta.get("IC Profiler", {})
        required = {
            "Columns",
            "Correction Rows",
            "Auxiliary Channels",
            "Ignored Raw Counts",
            "Ignored Detector IDs",
            "Detector IDs",
            "Leak Rates",
            "Calibration",
            "Ignored Detectors",
        }
        if not required.issubset(native) or len(self.measurements) != 4:
            raise ValueError(
                "PRM output requires four IC Profiler axes and retained acquisition metadata."
            )
        metadata = {
            key: copy.deepcopy(value)
            for key, value in self.file_meta.items()
            if key != "IC Profiler"
        }
        if (
            not metadata.get("Version")
            or metadata.get("Collector Config", {}).get("Collector Model") != "IC PROFILER"
            or metadata.get("Profiler Setup", {}).get("Orientation") != "Y Axis"
            or _number(metadata.get("Detector Spacing", ""), "Detector Spacing") != 0.5
            or any(
                metadata.get(key, "false").lower() != "false"
                for key in ("Concatenation", "Imported Data")
            )
        ):
            raise ValueError(
                "PRM writing requires a supported IC Profiler acquisition header and detector geometry."
            )
        columns = tuple(native["Columns"])
        detector_ids = tuple(native["Detector IDs"])
        expected_ids = {
            f"{axis}{number}"
            for axis in ("X", "Y", "PD", "ND")
            for number in range(1, 66)
            if axis == "Y" or number not in (32, 34)
        }
        if set(detector_ids) != expected_ids:
            raise ValueError("PRM output requires the complete native IC Profiler detector layout.")
        if (
            len(set(columns)) != len(columns)
            or len(set(detector_ids)) != len(detector_ids)
            or any(label not in columns for label in detector_ids)
        ):
            raise ValueError("PRM detector IDs and columns must be unique and consistent.")
        scans = {}
        for scan in self.measurements:
            scan.validate()
            axis = scan.metadata.get("IC Profiler", {}).get("Axis")
            if (
                scan.acquisition_type not in self.supported_acquisition_types
                or axis not in ("X", "Y", "PD", "ND")
                or axis in scans
            ):
                raise ValueError(
                    "PRM output requires one time series for each X, Y, PD and ND axis."
                )
            if (
                scan.metadata.get("Dose Scale") != "corrected counts"
                or scan.metadata.get("Cumulative") is not True
            ):
                raise ValueError("PRM output requires cumulative corrected counts.")
            scans[axis] = scan
        first = scans["X"]
        nframes = len(first.times)
        ticks = np.asarray(first.times) * 1e6
        if not np.allclose(ticks, np.rint(ticks), rtol=0, atol=1e-6) or np.any(ticks < 0):
            raise ValueError("PRM timestamps must resolve to nonnegative integral microseconds.")
        ticks = np.rint(ticks)
        data = np.empty((nframes, len(columns)))
        for label, key in (("UPDATE#", "Updates"), ("PULSES", "Pulses"), ("STATUS", "Statuses")):
            values = np.asarray(native.get(key))
            if (
                values.dtype.kind not in "biuf"
                or values.shape != (nframes,)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values % 1)
            ):
                raise ValueError(f"PRM {key} must contain nonnegative integers matching frames.")
            data[:, columns.index(label)] = values
        if np.any(np.diff(data[:, columns.index("UPDATE#")]) < 1) or np.any(
            np.diff(data[:, columns.index("PULSES")]) < 0
        ):
            raise ValueError("PRM update and pulse counters must increase cumulatively.")
        data[:, columns.index("TIMETIC")] = ticks
        for label in columns:
            if label in detector_ids or label in ("UPDATE#", "TIMETIC", "PULSES", "STATUS"):
                continue
            values = np.asarray(native["Auxiliary Channels"].get(label))
            if values.dtype.kind not in "biuf" or values.shape != (nframes,):
                raise ValueError(f"PRM auxiliary channel {label} must match frames.")
            data[:, columns.index(label)] = values
        leaks = np.asarray(native["Leak Rates"])
        calibration = np.asarray(native["Calibration"])
        ignored = np.asarray(native["Ignored Detectors"])
        for values in (leaks, calibration, ignored):
            if (
                values.dtype.kind not in "biuf"
                or values.shape != (len(detector_ids),)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "PRM calibration arrays must match detector IDs and contain finite real values."
                )
        if np.any(calibration <= 0) or not np.isin(ignored, (0, 1)).all():
            raise ValueError("PRM calibration must be positive and ignore flags must be 0 or 1.")
        ignored = ignored.astype(bool)
        if tuple(label for label, flag in zip(detector_ids, ignored) if flag) != tuple(
            native["Ignored Detector IDs"]
        ):
            raise ValueError(
                "PRM ignored detector IDs must match the retained ignored raw count columns."
            )
        ignored_raw = np.asarray(native["Ignored Raw Counts"])
        if (
            ignored_raw.dtype.kind not in "biuf"
            or ignored_raw.shape != (nframes, int(ignored.sum()))
            or not np.isfinite(ignored_raw).all()
        ):
            raise ValueError("PRM ignored raw counts must match frames and ignored detectors.")
        expected_beam = _measurement(metadata)
        depth = _length(metadata.get("Profiler Setup", {}).get("Buildup", ""), "Buildup")
        if depth < 0:
            raise ValueError("IC Profiler Buildup must be nonnegative.")
        seen = set()
        for axis, scan in scans.items():
            expected_type = {
                "X": MeasurementType.CrosslineProfile,
                "Y": MeasurementType.InlineProfile,
                "PD": MeasurementType.DiagonalProfile,
                "ND": MeasurementType.DiagonalProfile,
            }[axis]
            if scan.type is not expected_type or not np.array_equal(scan.times, first.times):
                raise ValueError(
                    "PRM axes must have their native profile types and identical timestamps."
                )
            for field in ("radiation_type", "energy", "SSD", "field_size", "wedge_angle"):
                if getattr(scan, field) != getattr(expected_beam, field):
                    raise ValueError(
                        f"PRM {field} must agree with the acquisition header; update file_meta too."
                    )
            for key in (
                "FluenceMode",
                "FieldsizeCorners[mm]",
                "GantryAngle[deg]",
                "CollimatorAngle[deg]",
            ):
                if scan.metadata.get(key) != expected_beam.metadata.get(key):
                    raise ValueError(f"PRM {key} must agree with the acquisition header.")
            ids = tuple(scan.metadata["IC Profiler"].get("Detector IDs", ()))
            expected_ids = {label for label in detector_ids if label.startswith(axis)}
            if (
                len(ids) != scan.doses.shape[1]
                or set(ids) != expected_ids
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(
                    "PRM measurement detector IDs must match their native axis layout."
                )
            distance = np.array([int(label[len(axis) :]) - 33 for label in ids]) * 5.0
            xyz = np.zeros((len(ids), 3))
            xyz[:, 2] = depth
            xyz[:, 0 if axis != "Y" else 1] = distance
            if axis in ("PD", "ND"):
                xyz[:, 1] = distance * (1 if axis == "PD" else -1)
            if not np.allclose(np.column_stack(scan.reconstruct_xyz()), xyz, rtol=0, atol=1e-9):
                raise ValueError(
                    "PRM geometry must match the native detector positions and buildup."
                )
            for j, label in enumerate(ids):
                i = detector_ids.index(label)
                if ignored[i]:
                    if not np.isnan(scan.doses[:, j]).all():
                        raise ValueError("PRM ignored detectors must retain NaN corrected signals.")
                    raw = ignored_raw[:, np.count_nonzero(ignored[:i])]
                else:
                    raw = scan.doses[:, j] / calibration[i] + ticks * leaks[i]
                    if not np.isfinite(raw).all():
                        raise ValueError("PRM detector counts must remain finite.")
                data[:, columns.index(label)] = raw
                seen.add(label)
        if seen != set(detector_ids):
            raise ValueError("PRM output requires every native detector.")
        metadata["Total Pulses"] = f"{data[-1, columns.index('PULSES')]:.17g}"
        metadata["Total Time"] = f"{ticks[-1] * 1e-6:.17g}"
        lines = _header_output(metadata)
        lines.extend(
            ("Definitions:", *native.get("Definitions", ()), "\t".join(["TYPE", *columns]))
        )
        factors = {"BIAS1": leaks, "Calibration": calibration, "IgnoreDet": ignored.astype(int)}
        for label, original_cells in native["Correction Rows"].items():
            cells = list(original_cells)
            if label not in factors or len(cells) > len(columns):
                raise ValueError("PRM correction rows are inconsistent with columns.")
            cells.extend([""] * (len(columns) - len(cells)))
            for name, value in zip(detector_ids, factors[label]):
                cells[columns.index(name)] = f"{value:.17g}"
            lines.append("\t".join([label, *cells]))
        if set(native["Correction Rows"]) != set(factors):
            raise ValueError("PRM output requires all calibration rows.")
        lines.extend("\t".join(["Data:", *(f"{v:.17g}" for v in row)]) for row in data)
        if any("\r" in line or "\n" in line for line in lines):
            raise ValueError("PRM records must be single-line values.")
        encoding = native.get("Encoding", "utf-8")
        if encoding not in ("utf-8", "utf-8-sig", "cp1252"):
            raise ValueError("PRM output encoding must be UTF-8, UTF-8 BOM or CP1252.")
        payload = ("\r\n".join(lines) + "\r\n").encode(encoding)
        Path(output_path).write_bytes(payload)


def _header_output(metadata: dict) -> list[str]:
    def fields(values):
        result = []
        secondary = {
            key
            for primary in _SECONDARY_FIELDS
            if primary in values
            for key in _SECONDARY_FIELDS[primary]
        }
        for key, value in values.items():
            if key in secondary or isinstance(value, dict):
                continue
            if not isinstance(key, str) or not key or any(c in key for c in ":\t\r\n"):
                raise ValueError("PRM header keys must be single-line labels.")
            line = f"{key}:\t{value}"
            for name in values:
                if name in _SECONDARY_FIELDS.get(key, ()):
                    line += f"\t{name}:\t{values[name]}"
            result.append(line)
        return result

    # Multi Frame resets the parser's section context for acquisition totals.
    roots = {key: value for key, value in metadata.items() if key not in _SECTIONS}
    trailing = {
        key: roots.pop(key)
        for key in tuple(roots)
        if key
        in {
            "Multi Frame",
            "Total Pulses",
            "Total Time",
            "Detectors",
            "Detector Spacing",
            "Concatenation",
            "Imported Data",
        }
    }
    result = [f"Version:\t{roots.pop('Version')}", *fields(roots)]
    for section, value in metadata.items():
        if section in _SECTIONS:
            result.append("\t" + section)
            result.extend(fields(value))
    multi = trailing.pop("Multi Frame", "true")
    result.append(f"Multi Frame:\t{multi}")
    result.extend(fields(trailing))
    return result


def _measurement(metadata: dict) -> Measurement:
    scan = Measurement()
    machine, setup = metadata.get("Machine Data", {}), metadata.get("Profiler Setup", {})
    beam = machine.get("Beam Type", "Undefined")
    modes = {
        "X-Ray": RadiationType.PHOTON,
        "X-Ray FFF": RadiationType.PHOTON,
        "Electron": RadiationType.ELECTRON,
        "Proton": RadiationType.PROTON,
    }
    if beam != "Undefined":
        if beam not in modes:
            raise ValueError(f"Unsupported IC Profiler Beam Type: {beam!r}")
        scan.radiation_type = modes[beam]
    energy = machine.get("Energy", "Undefined")
    if energy != "Undefined":
        match = re.fullmatch(f"({_NUMBER}) (MV|MeV)", energy)
        if not match:
            raise ValueError(f"Invalid IC Profiler Energy: {energy!r}")
        scan.energy = _number(match[1], "Energy")
    scan.SSD = _length(setup["SSD"], "SSD") if setup.get("SSD") else None
    if machine.get("Collimator"):
        match = re.fullmatch(
            f"Left: ({_NUMBER}) Right: ({_NUMBER}) Top: ({_NUMBER}) Bottom: ({_NUMBER}) (cm|mm)",
            machine["Collimator"],
        )
        if not match:
            raise ValueError("Invalid IC Profiler Collimator jaw positions.")
        left, right, top, bottom = [_number(v, "Collimator") for v in match.groups()[:4]]
        scale = 10 if match[5] == "cm" else 1
        scan.field_size = ((left + right) * scale, (top + bottom) * scale)
        scan.metadata["FieldsizeCorners[mm]"] = "; ".join(
            f"{v * scale:.17g}" for v in (-left, -bottom, right, top)
        )
    wedge = machine.get("Wedge")
    if wedge:
        parts = wedge.split()
        if len(parts) != 3 or parts[1] != "at" or parts[0] not in ("None", "Static", "Dynamic"):
            raise ValueError(f"Unsupported IC Profiler Wedge: {wedge!r}")
        scan.wedge_angle = _number(parts[2], "Wedge") if parts[0] != "None" else 0.0
    if scan.radiation_type == RadiationType.PHOTON:
        scan.metadata["FluenceMode"] = "FFF" if beam == "X-Ray FFF" else "Standard"
    for key in ("Gantry Angle", "Collimator Angle"):
        if machine.get(key):
            parts = machine[key].split()
            if len(parts) != 2 or parts[1] != "deg":
                raise ValueError(f"Invalid IC Profiler {key}.")
            scan.metadata[key.replace(" ", "") + "[deg]"] = _number(parts[0], key)
    return scan
