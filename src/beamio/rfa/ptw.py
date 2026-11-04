"""Read and write PTW MCC water-tank scans without resampling or normalization."""

from __future__ import annotations

from pathlib import Path
import copy
import re

import numpy as np

from beamio.common.beamdata import BeamData
from beamio.common.measurement import AcquisitionType, Measurement, MeasurementType, RadiationType

_CURVE_TYPES = {
    "PDD": MeasurementType.DepthDose,
    "CROSSPLANE_PROFILE": MeasurementType.CrosslineProfile,
    "INPLANE_PROFILE": MeasurementType.InlineProfile,
}
_MODALITIES = {"X": RadiationType.PHOTON, "EL": RadiationType.ELECTRON}
_FILTERS = {"FF": "Standard", "FFF": "FFF", "SRS": "SRS"}


def _number(metadata: dict[str, str], key: str, default=None) -> float | None:
    value = metadata.get(key, "")
    if not value:
        return default
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"Invalid MCC {key}: {value!r}") from None
    if not np.isfinite(result):
        raise ValueError(f"MCC {key} must be finite.")
    return result


def _measurement(metadata: dict[str, str], rows: list[list[float]]) -> Measurement:
    detector = metadata.get("DETECTOR", "").upper()
    device = metadata.get("SCAN_DEVICE", "").upper()
    if any(name.startswith(("STARCHECK", "OCTAVIUS")) for name in (detector, device)):
        raise ValueError("MCC detector-array measurements are not supported.")
    if metadata.get("SCAN_DIAGONAL", "NOT_DIAGONAL") != "NOT_DIAGONAL":
        raise ValueError("MCC diagonal scans are not supported.")
    if _number(metadata, "SCAN_ANGLE", 0.0) != 0:
        raise ValueError("MCC angled scans are not supported.")
    for key, expected in (("CROSSPLANE_AXIS", "X"), ("INPLANE_AXIS", "Y")):
        if metadata.get(key, expected) != expected:
            raise ValueError(f"Unsupported MCC axis mapping: {key}={metadata[key]!r}")
    kind = metadata.get("SCAN_CURVETYPE", "")
    if kind not in _CURVE_TYPES:
        raise ValueError(f"Unsupported MCC SCAN_CURVETYPE: {kind!r}")
    if not rows:
        raise ValueError("MCC scan must contain at least one data sample.")

    samples = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(samples[:, 0])):
        raise ValueError("MCC sample positions must be finite.")
    result = Measurement()
    result.type = _CURVE_TYPES[kind]
    result.positions = samples[:, 0].copy()
    result.doses = samples[:, 1].copy()
    result.metadata = dict(metadata)
    if samples.shape[1] == 3:
        result.metadata["PTW Reference"] = samples[:, 2].copy()

    modality = metadata.get("MODALITY", "")
    if modality:
        if modality not in _MODALITIES:
            raise ValueError(f"Unsupported MCC MODALITY: {modality!r}")
        result.radiation_type = _MODALITIES[modality]
    beam_filter = metadata.get("FILTER", "")
    if result.radiation_type == RadiationType.PHOTON and beam_filter:
        if beam_filter not in _FILTERS:
            raise ValueError(f"Unsupported MCC photon FILTER: {beam_filter!r}")
        result.metadata["FluenceMode"] = _FILTERS[beam_filter]
    result.energy = _number(metadata, "ENERGY")
    result.SSD = _number(metadata, "SSD")
    result.wedge_angle = _number(metadata, "WEDGE_ANGLE")
    center_x = _number(metadata, "COLL_OFFSET_CROSSPLANE", 0.0)
    center_y = _number(metadata, "COLL_OFFSET_INPLANE", 0.0)
    width, height = _number(metadata, "FIELD_CROSSPLANE"), _number(metadata, "FIELD_INPLANE")
    if width is not None and height is not None:
        result.field_size = (width, height)
        if center_x != 0 or center_y != 0:
            # Preserve displaced jaw edges through the existing field-corner metadata
            # used by the RayStation writer; widths alone cannot describe this field.
            corners = (
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            )
            result.metadata["FieldsizeCorners[mm]"] = "; ".join(
                f"{value:.17g}" for value in corners
            )

    # Samples already include displacement along the moving axis. Collimator offsets
    # describe the field, so they must not be added to the measured positions.
    xyz = np.empty((len(samples), 3), dtype=float)
    xyz[:, 0] = _number(metadata, "SCAN_OFFAXIS_CROSSPLANE", 0.0)
    xyz[:, 1] = _number(metadata, "SCAN_OFFAXIS_INPLANE", 0.0)
    axis = {
        MeasurementType.CrosslineProfile: 0,
        MeasurementType.InlineProfile: 1,
        MeasurementType.DepthDose: 2,
    }[result.type]
    if axis != 2:
        depth = _number(metadata, "SCAN_DEPTH")
        if depth is None:
            raise ValueError("MCC profiles require SCAN_DEPTH.")
        xyz[:, 2] = depth
    xyz[:, axis] = result.positions
    result.coordinates = xyz
    result.startpoint = tuple(float(value) for value in xyz[0])
    result.endpoint = tuple(float(value) for value in xyz[-1])
    return result


class PTWMCC(BeamData):
    """Read MCC depth, crossplane, and inplane water-tank scans (spatial units: mm).

    Reference readings, when present, are stored in ``metadata['PTW Reference']``.
    Samples retain their export order and scale; no reference correction is applied.
    Writing supports the same static axial scan types, including reference readings.
    """

    supported_acquisition_types = frozenset({AcquisitionType.Static})
    supported_measurement_types = frozenset(_CURVE_TYPES.values())
    supported_radiation_types = frozenset({None, *_MODALITIES.values()})

    def parse(self) -> PTWMCC:
        path = Path(self.file_path)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp1252")

        state = "outside"
        file_meta: dict[str, str] = {}
        measurements: list[Measurement] = []
        identifiers: set[int] = set()
        scan_id = None
        metadata: dict[str, str] = {}
        rows: list[list[float]] = []
        data_seen = False
        for line_number, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            try:
                if state == "outside":
                    if line != "BEGIN_SCAN_DATA":
                        raise ValueError("Expected BEGIN_SCAN_DATA.")
                    state = "header"
                    continue
                if state == "closed":
                    raise ValueError("Unexpected content after END_SCAN_DATA.")
                if state == "data":
                    if line == "END_DATA":
                        state = "scan"
                        continue
                    if line.startswith(("BEGIN_", "END_")):
                        raise ValueError("Expected END_DATA before another MCC block marker.")
                    columns = line.split()
                    if any(token.startswith("#") for token in columns):
                        raise ValueError("MCC detector-array channel data are not supported.")
                    if len(columns) not in (2, 3) or (rows and len(columns) != len(rows[0])):
                        raise ValueError("MCC data require consistently two or three columns.")
                    try:
                        rows.append([float(value) for value in columns])
                    except ValueError:
                        raise ValueError("Invalid numeric MCC data sample or missing END_DATA.")
                    continue
                begin = re.fullmatch(r"BEGIN_SCAN\s+(\d+)", line)
                end = re.fullmatch(r"END_SCAN\s+(\d+)", line)
                if begin:
                    if state != "header":
                        raise ValueError("Nested MCC scans are not allowed.")
                    scan_id = int(begin[1])
                    if scan_id in identifiers:
                        raise ValueError(f"Duplicate MCC scan number: {scan_id}")
                    identifiers.add(scan_id)
                    metadata, rows, data_seen = {}, [], False
                    state = "scan"
                elif end:
                    if state != "scan" or int(end[1]) != scan_id:
                        raise ValueError("Mismatched MCC END_SCAN.")
                    if not data_seen:
                        raise ValueError("MCC scan is missing BEGIN_DATA.")
                    measurement = _measurement(metadata, rows)
                    measurement.metadata["PTW Scan Number"] = scan_id
                    measurements.append(measurement)
                    state, scan_id = "header", None
                elif line == "BEGIN_DATA":
                    if state != "scan" or data_seen:
                        raise ValueError("Unexpected or duplicate MCC BEGIN_DATA.")
                    data_seen, state = True, "data"
                elif line == "END_SCAN_DATA":
                    if state != "header":
                        raise ValueError("MCC scan is missing END_SCAN.")
                    state = "closed"
                elif "=" in line:
                    key, value = (part.strip() for part in line.split("=", 1))
                    target = metadata if state == "scan" else file_meta
                    if not key or key in target:
                        raise ValueError(f"Empty or duplicate MCC metadata key: {key!r}")
                    target[key] = value
                else:
                    raise ValueError(f"Unexpected MCC content: {line!r}")
            except ValueError as exc:
                context = f", scan {scan_id}" if scan_id is not None else ""
                raise ValueError(f"MCC line {line_number}{context}: {exc}") from exc
        if state != "closed":
            raise ValueError("Incomplete MCC file: expected closed data, scan, and END_SCAN_DATA.")
        if not measurements:
            raise ValueError("MCC file contains no scans.")
        self.file_meta, self.measurements = file_meta, measurements
        return self

    def save(self, output_path: str) -> None:
        if not self.measurements:
            raise ValueError("MCC output requires at least one measurement.")
        lines = ["BEGIN_SCAN_DATA"]
        lines.extend(_metadata_lines({"FORMAT": "CC-Export V1.9", **self.file_meta}, "\t"))
        used = set()
        for index, measurement in enumerate(self.measurements, 1):
            scan = copy.copy(measurement)
            if (
                scan.acquisition_type not in self.supported_acquisition_types
                or scan.type not in self.supported_measurement_types
            ):
                raise ValueError("MCC output supports only static axial 1D measurements.")
            scan.validate()
            axis = {
                MeasurementType.CrosslineProfile: 0,
                MeasurementType.InlineProfile: 1,
                MeasurementType.DepthDose: 2,
            }[scan.type]
            if scan.coordinates is not None:
                if scan.startpoint is None:
                    scan.startpoint = tuple(scan.coordinates[0])
                if scan.endpoint is None:
                    scan.endpoint = tuple(scan.coordinates[-1])
            if scan.startpoint is None:
                raise ValueError("MCC output requires spatial coordinates or a startpoint.")
            if scan.endpoint is None:
                end = list(scan.startpoint)
                end[axis] = float(scan.positions[-1])
                scan.endpoint = tuple(end)
            xyz = np.column_stack(scan.reconstruct_xyz())
            for fixed in set(range(3)) - {axis}:
                if not np.allclose(xyz[:, fixed], xyz[0, fixed], rtol=0, atol=1e-9):
                    raise ValueError("MCC output cannot represent non-axial acquisition motion.")
            fields = {
                key: value
                for key, value in scan.metadata.items()
                if key
                not in {"PTW Reference", "PTW Scan Number", "FluenceMode", "FieldsizeCorners[mm]"}
                and isinstance(value, (str, int, float))
            }
            fields.update(
                {
                    "SCAN_CURVETYPE": next(
                        key for key, value in _CURVE_TYPES.items() if value == scan.type
                    ),
                    "SCAN_DIAGONAL": "NOT_DIAGONAL",
                    "SCAN_ANGLE": "0",
                    "CROSSPLANE_AXIS": "X",
                    "INPLANE_AXIS": "Y",
                }
            )
            if axis != 2:
                fields["SCAN_DEPTH"] = f"{xyz[0, 2]:.17g}"
            for coordinate, key in ((0, "SCAN_OFFAXIS_CROSSPLANE"), (1, "SCAN_OFFAXIS_INPLANE")):
                fields[key] = (
                    fields.get(key, "0") if coordinate == axis else f"{xyz[0, coordinate]:.17g}"
                )
            for key, value in (
                ("ENERGY", scan.energy),
                ("SSD", scan.SSD),
                ("WEDGE_ANGLE", scan.wedge_angle),
            ):
                if value is None:
                    fields.pop(key, None)
                else:
                    fields[key] = f"{value:.17g}"
            if scan.radiation_type is None:
                fields.pop("MODALITY", None)
            elif scan.radiation_type not in _MODALITIES.values():
                raise ValueError("MCC output supports photon and electron modalities.")
            else:
                fields["MODALITY"] = next(
                    key for key, value in _MODALITIES.items() if value == scan.radiation_type
                )
            mode = scan.metadata.get("FluenceMode")
            if scan.radiation_type is RadiationType.PHOTON and mode is not None:
                if mode not in _FILTERS.values():
                    raise ValueError("Unsupported MCC photon fluence mode.")
                fields["FILTER"] = next(key for key, value in _FILTERS.items() if value == mode)
            if scan.field_size is None:
                fields.pop("FIELD_CROSSPLANE", None)
                fields.pop("FIELD_INPLANE", None)
            else:
                fields["FIELD_CROSSPLANE"], fields["FIELD_INPLANE"] = (
                    f"{v:.17g}" for v in scan.field_size
                )
                corners = scan.metadata.get("FieldsizeCorners[mm]")
                if corners is not None:
                    edges = np.array([float(v) for v in str(corners).split(";")])
                    if (
                        edges.shape != (4,)
                        or not np.isfinite(edges).all()
                        or not np.allclose(
                            edges[2:] - edges[:2], scan.field_size, rtol=0, atol=1e-9
                        )
                    ):
                        raise ValueError(
                            "MCC field corners must match field_size; update or clear them."
                        )
                    fields["COLL_OFFSET_CROSSPLANE"], fields["COLL_OFFSET_INPLANE"] = (
                        f"{v:.17g}" for v in (edges[2:] + edges[:2]) / 2
                    )
            arrays = [scan.positions, scan.doses]
            if "PTW Reference" in scan.metadata:
                reference = np.asarray(scan.metadata["PTW Reference"])
                if reference.dtype.kind not in "biuf" or reference.shape != scan.positions.shape:
                    raise ValueError("PTW Reference must be a real array matching positions.")
                arrays.append(reference)
            rows = np.column_stack(arrays).tolist()
            _measurement(fields, rows)  # Apply the native schema before touching the destination.
            number = scan.metadata.get("PTW Scan Number", index)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, np.integer))
                or number < 0
                or number in used
            ):
                raise ValueError("MCC scan numbers must be unique nonnegative integers.")
            used.add(number)
            lines.append(f"\tBEGIN_SCAN {number}")
            lines.extend(_metadata_lines(fields, "\t\t"))
            lines.append("\t\tBEGIN_DATA")
            lines.extend("\t\t\t" + "\t".join(f"{value:.17g}" for value in row) for row in rows)
            lines.extend(("\t\tEND_DATA", f"\tEND_SCAN {number}"))
        lines.append("END_SCAN_DATA")
        payload = ("\r\n".join(lines) + "\r\n").encode("utf-8")
        Path(output_path).write_bytes(payload)


def _metadata_lines(metadata: dict, indent: str) -> list[str]:
    result = []
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float)):
            continue
        if (
            not isinstance(key, str)
            or not key
            or any(c in key for c in "=\r\n\t")
            or any(c in str(value) for c in "\r\n")
        ):
            raise ValueError("MCC metadata requires single-line keys and values.")
        result.append(f"{indent}{key}={value}")
    return result
