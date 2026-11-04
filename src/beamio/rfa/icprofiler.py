"""Sun Nuclear IC PROFILER acquisitions, validated with PRM version 25."""

from __future__ import annotations

import copy
from pathlib import Path
import re
import warnings

import numpy as np

from beamio.common.beamdata import BeamData
from beamio.common.measurement import (
    Measurement,
    MeasurementType,
    RadiationType,
    diagonal_direction,
)
from beamio.rfa._icprofiler_series import ICProfilerBeamCenter, ICProfilerTimeSeries

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
    """Check available export totals against the complete table, before frame selection."""
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

    By default the final cumulative frame is leakage-corrected and array-calibrated,
    in relative counts. ``frame`` selects a cumulative frame by Python index.
    ``absolute=True`` additionally applies the acquisition's Dose Per Count factor;
    a valid enabled absolute calibration is required. No normalization is performed.

    Version 25 is validated. Other nonempty versions emit UserWarning and are
    attempted using the same layout and data checks.

    ``columns`` and ``data`` retain all numeric acquisition frames, including auxiliary
    channels. ``detector_ids``, ``leak_rates``, ``calibration``, and
    ``ignored_detectors`` describe the detector channels in exported column order.
    Ignored detectors remain in profiles with NaN dose; no interpolation is applied.
    """

    def __init__(self, file_path: str, *, frame: int = -1, absolute: bool = False):
        super().__init__(file_path)
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError("IC Profiler frame must be an integer index.")
        self.frame = frame
        self.absolute = absolute
        self.columns: tuple[str, ...] = ()
        self.data = np.empty((0, 0))
        self.detector_ids: tuple[str, ...] = ()
        self.leak_rates = np.array([])
        self.calibration = np.array([])
        self.ignored_detectors = np.array([], dtype=bool)
        self._dose_factor = 1.0
        self._dose_scale = "corrected counts"

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
        if not -len(data) <= self.frame < len(data):
            raise ValueError(
                f"IC Profiler frame {self.frame} is out of range for {len(data)} frames."
            )
        frame_index = self.frame % len(data)

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
        ticks = data[frame_index, columns.index("TIMETIC")]
        doses = (data[frame_index, indices] - ticks * factors["BIAS1"]) * factors["Calibration"]
        dose_factor = 1.0
        if self.absolute:
            dose_cal = metadata.get("Dose Calibration", {})
            dose_per_count = _number(dose_cal.get("Dose Per Count", ""), "Dose Per Count")
            if dose_cal.get("Absolute Calibration", "").lower() != "true" or dose_per_count <= 0:
                raise ValueError(
                    "IC Profiler absolute dose requires an enabled positive calibration."
                )
            dose_factor = dose_per_count
            doses *= dose_factor
        doses[ignored] = np.nan
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
            scan.doses = doses[selected].copy()
            scan.metadata["IC Profiler"] = {
                "Axis": axis,
                "Detector IDs": tuple(detector_ids[i] for i in selected),
                "Ignored Detectors": ignored[selected].copy(),
                "Raw Counts": data[frame_index, indices[selected]].copy(),
                "Leak Rates": factors["BIAS1"][selected].copy(),
                "Calibration": factors["Calibration"][selected].copy(),
                "Frame Index": frame_index,
                "Frame Count": len(data),
                "TimeTic": ticks,
                "Dose Scale": "absolute" if self.absolute else "corrected counts",
            }
            measurements.append(scan)
        self.file_meta, self.measurements = metadata, measurements
        self.columns, self.data, self.detector_ids = columns, data, detector_ids
        self.leak_rates, self.calibration = factors["BIAS1"], factors["Calibration"]
        self.ignored_detectors = ignored
        self._dose_factor = dose_factor
        self._dose_scale = "absolute" if self.absolute else "corrected counts"
        return self

    def time_series(self, axis: str = "X", *, cumulative: bool = False) -> ICProfilerTimeSeries:
        """Return calibrated profiles for all updates on X, Y, PD, or ND.

        By default each row is the signal added since the previous update, with
        leakage correction using that interval's tick difference. The first row
        starts at acquisition time zero. ``cumulative=True`` returns integrated
        profiles. This method always uses all frames, regardless of ``frame``.

        Times are exported TIMETIC values converted from microseconds to seconds.
        In interval mode, nonpositive durations or no new pulses give NaN rows;
        frame alignment and the original timestamps are retained. No smoothing,
        resampling, normalization, or division by interval duration is applied.
        """
        if axis not in ("X", "Y", "PD", "ND"):
            raise ValueError("IC Profiler time-series axis must be X, Y, PD, or ND.")
        if not self.columns or not self.data.size:
            raise ValueError("Parse an IC Profiler acquisition before requesting time series.")
        selected = [
            i
            for i, label in enumerate(self.detector_ids)
            if label.startswith(axis) and label[len(axis) :].isdigit()
        ]
        detector_ids = tuple(self.detector_ids[i] for i in selected)
        columns = [self.columns.index(label) for label in detector_ids]
        ticks = self.data[:, self.columns.index("TIMETIC")].copy()
        delta_ticks = np.diff(ticks, prepend=0)
        pulses = self.data[:, self.columns.index("PULSES")].copy()
        counts = self.data[:, columns].copy()
        if cumulative:
            elapsed = ticks
            valid = (ticks > 0) & (pulses > 0)
        else:
            elapsed = delta_ticks
            pulses = np.diff(pulses, prepend=0)
            counts = np.diff(counts, axis=0, prepend=np.zeros((1, len(columns))))
            valid = (delta_ticks > 0) & (pulses > 0)
        doses = (counts - elapsed[:, None] * self.leak_rates[selected]) * self.calibration[selected]
        doses *= self._dose_factor
        doses[:, self.ignored_detectors[selected]] = np.nan
        doses[~valid] = np.nan
        positions = (
            np.array([int(label[len(axis) :]) - 33 for label in detector_ids], dtype=float) * 5
        )
        if axis in ("PD", "ND"):
            positions *= np.sqrt(2)
        return ICProfilerTimeSeries(
            axis=axis,
            times=ticks * 1e-6,
            durations=delta_ticks * 1e-6,
            positions=positions,
            doses=doses,
            pulses=pulses,
            updates=self.data[:, self.columns.index("UPDATE#")].astype(np.int64),
            statuses=self.data[:, self.columns.index("STATUS")].astype(np.int64),
            valid_frames=valid,
            cumulative=cumulative,
            dose_scale=self._dose_scale,
            detector_ids=detector_ids,
        )

    def beam_center(
        self,
        *,
        cumulative: bool = False,
        method: str = "InflectionPoint",
        level: float = 0.5,
        min_signal: float = 0.05,
    ) -> ICProfilerBeamCenter:
        """Return X/Y field-edge-midpoint beam centres over the complete acquisition.

        The default uses interval profiles and InflectionPoint edges. Set
        ``method='FWHM'`` for half-maximum edges (or ``level`` for another fraction).
        ``min_signal``
        is the minimum peak as a fraction of the strongest frame on each axis.
        Undetectable centres are NaN. This geometric metric does not apply a
        vendor-specific FFF, wedge, or beam-quality correction.
        """
        x = self.time_series("X", cumulative=cumulative)
        y = self.time_series("Y", cumulative=cumulative)
        return ICProfilerBeamCenter(
            times=x.times,
            x=x.beam_center(method=method, level=level, min_signal=min_signal),
            y=y.beam_center(method=method, level=level, min_signal=min_signal),
            cumulative=cumulative,
            level=level if method == "FWHM" else None,
            min_signal=min_signal,
            method=method,
        )

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
    # Header sections are retained independently for each measurement and conversions.
    scan.metadata["IC Profiler Header"] = copy.deepcopy(metadata)
    return scan
