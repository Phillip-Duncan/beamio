"""Sun Nuclear IC PROFILER acquisitions, validated with PRM version 25."""

from __future__ import annotations

from pathlib import Path
import re
import warnings

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
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")
    metadata: dict = {}
    corrections: dict[str, list[str]] = {}
    rows: list[list[float]] = []
    target = metadata
    columns = None
    definitions = False
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
        metadata["IC Profiler"] = {
            "Detector IDs": detector_ids,
            "Leak Rates": factors["BIAS1"],
            "Calibration": factors["Calibration"],
            "Ignored Detectors": ignored,
            # Copies avoid retaining the complete raw table through NumPy views.
            "Updates": data[:, columns.index("UPDATE#")].copy(),
            "Pulses": data[:, columns.index("PULSES")].copy(),
            "Statuses": data[:, columns.index("STATUS")].copy(),
        }
        self.file_meta, self.measurements = metadata, measurements
        return self

    def save(self, output_path: str) -> None:
        raise NotImplementedError(
            "IC Profiler PRM writing is not supported; convert to another format."
        )


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
