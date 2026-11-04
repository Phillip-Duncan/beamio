"""Sun Nuclear SunDOSE/SunSCAN processed measurement input and text output."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import re

import numpy as np

from beamio import __version__
from beamio.common.beamdata import BeamData
from beamio.common.measurement import (
    AcquisitionType,
    Measurement,
    MeasurementType,
    RadiationType,
    diagonal_direction,
)
from beamio.rfa._snc_xml import number, read_xml

SCAN_TYPES = {
    "depthscan": MeasurementType.DepthDose,
    "percentdepthdose": MeasurementType.DepthDose,
    "crossline": MeasurementType.CrosslineProfile,
    "inline": MeasurementType.InlineProfile,
    "profileatangle": MeasurementType.DiagonalProfile,
    "diagonal": MeasurementType.DiagonalProfile,
}
SCAN_NAMES = {
    MeasurementType.DepthDose: "Depth Scan",
    MeasurementType.CrosslineProfile: "Crossline",
    MeasurementType.InlineProfile: "Inline",
    MeasurementType.DiagonalProfile: "Profile at Angle",
}
HEADER_SECTIONS = (
    "FACILITY INFORMATION",
    "SNC EQUIPMENT",
    "DELIVERY SYSTEM",
    "SETUP PARAMETERS",
    "MEASUREMENT DETAILS",
)
PHOTON_MODES = {
    "ff": "Standard",
    "standard": "Standard",
    "flattened": "Standard",
    "fff": "FFF",
    "flatteningfilterfree": "FFF",
    "unflattened": "FFF",
    "srs": "SRS",
}


def photon_fluence_mode(value: str) -> str:
    try:
        return PHOTON_MODES[re.sub(r"[\s_-]+", "", value).casefold()]
    except KeyError:
        raise ValueError(f"Unsupported SNC photon rate type or fluence mode: {value!r}") from None


def first(metadata: dict, *keys: str) -> str:
    return next(
        (str(metadata[key]).strip() for key in keys if metadata.get(key) not in (None, "")), ""
    )


def optional_number(metadata: dict, *keys: str, scale: float = 1.0) -> float | None:
    value = first(metadata, *keys)
    return number(value) * scale if value else None


def measurement_from_table(metadata: dict, xyz: np.ndarray, doses: np.ndarray) -> Measurement:
    """Map processed coordinates in millimetres without sorting or filtering samples."""
    if xyz.ndim != 2 or xyz.shape != (len(doses), 3) or not len(doses):
        raise ValueError("SNC dose table must contain matching XYZ coordinates and doses.")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("SNC sample coordinates must be finite.")
    scan_type = first(metadata, "Summary Scan Type", "Scan Type")
    try:
        kind = SCAN_TYPES[scan_type.replace(" ", "").casefold()]
    except KeyError:
        raise ValueError(f"Unsupported SNC scan type: {scan_type!r}") from None

    measurement = Measurement()
    measurement.type = kind
    measurement.metadata = metadata
    measurement.startpoint = tuple(float(value) for value in xyz[0])
    measurement.endpoint = tuple(float(value) for value in xyz[-1])
    if kind == MeasurementType.DiagonalProfile:
        direction = diagonal_direction(measurement.startpoint, measurement.endpoint)
        measurement.positions = xyz[:, :2] @ direction
    else:
        axis = {
            MeasurementType.CrosslineProfile: 0,
            MeasurementType.InlineProfile: 1,
            MeasurementType.DepthDose: 2,
        }[kind]
        measurement.positions = xyz[:, axis].copy()
    measurement.coordinates = xyz.copy()
    measurement.doses = doses.copy()
    radiation = first(metadata, "Summary Beam Type", "Beam Type")
    if radiation:
        try:
            measurement.radiation_type = RadiationType(radiation.title())
        except ValueError:
            raise ValueError(f"Unsupported SNC beam type: {radiation!r}") from None
    rate_type = first(metadata, "Rate Type", "FluenceMode")
    if measurement.radiation_type == RadiationType.PHOTON and rate_type:
        metadata["FluenceMode"] = photon_fluence_mode(rate_type)
    measurement.energy = optional_number(metadata, "Summary Energy (MV/MeV)", "Energy (MV / MeV)")
    measurement.SSD = optional_number(metadata, "Source to Surface Distance (cm)", scale=10)
    width = optional_number(metadata, "Summary FieldSize X (cm)", "Field Size X (cm)", scale=10)
    height = optional_number(metadata, "Summary FieldSize Y (cm)", "Field Size Y (cm)", scale=10)
    if width is not None and height is not None:
        measurement.field_size = (width, height)
    measurement.wedge_angle = optional_number(
        metadata, "Summary Wedge Angle (degrees)", "Wedge Angle (degrees)"
    )
    for source, target in (
        ("Gantry Angle (degrees)", "GantryAngle[deg]"),
        ("Collimator Angle (degrees)", "CollimatorAngle[deg]"),
    ):
        if first(metadata, source):
            metadata[target] = metadata[source]
    return measurement


def column_layout(cells: list[str]):
    axes = {}
    dose = None
    for index, label in enumerate(cells):
        match = re.fullmatch(r"([XYZ])\s*\((cm|mm)\)", label, flags=re.IGNORECASE)
        if match:
            axes[match[1].upper()] = (index, 10.0 if match[2].lower() == "cm" else 1.0)
        elif label.casefold() == "relative dose (%)":
            dose = index
    if len(cells) != 4 or len(axes) != 3 or dose is None:
        raise ValueError(
            "SNC dose table requires X, Y, Z (cm or mm) and Relative Dose (%) columns."
        )
    return axes, dose


def read_text(path: Path):
    file_meta, scans, scan_meta, rows = {}, [], None, None
    in_table = in_raw = False
    layout = None
    signature = False
    file_header = False
    section = None
    with path.open(encoding="utf-8-sig") as source:
        for line_number, raw in enumerate(source, start=1):
            line = raw.rstrip("\r\n")
            marker = line.strip()
            if not marker:
                continue
            try:
                if not signature:
                    if marker != "Tab-Delimited Scan Output":
                        raise ValueError("Not an SNC tab-delimited scan export.")
                    signature = True
                elif in_raw:
                    if marker == "END RAW DATA":
                        in_raw = False
                elif marker == "FILE HEADER":
                    if file_header or scan_meta is not None or scans:
                        raise ValueError("Unexpected FILE HEADER.")
                    file_header = True
                elif marker == "BEGIN SCAN":
                    if scan_meta is not None:
                        raise ValueError("Nested scan; missing END SCAN.")
                    scan_meta, rows, layout, section = {"SNC Sections": {}}, None, None, None
                elif marker == "END SCAN":
                    if scan_meta is None or in_table or not rows:
                        raise ValueError("Incomplete scan or missing dose table.")
                    data = np.asarray(rows, dtype=float)
                    scans.append((scan_meta, data[:, :3], data[:, 3]))
                    scan_meta = None
                elif marker == "BEGIN DOSE TABLE":
                    if scan_meta is None or rows is not None:
                        raise ValueError(
                            "Dose table outside a scan or multiple dose tables in one scan."
                        )
                    rows, in_table = [], True
                elif marker == "END DOSE TABLE":
                    if not in_table or layout is None or not rows:
                        raise ValueError("Incomplete dose table.")
                    in_table = False
                elif marker == "BEGIN RAW DATA":
                    if scan_meta is None or in_table:
                        raise ValueError("Raw data outside a scan or inside a dose table.")
                    in_raw = True
                elif in_table:
                    if line.startswith("Action\t") and layout is None:
                        scan_meta["Action"] = line.split("\t", 1)[1].rstrip("\t")
                        continue
                    cells = line.strip("\t").split("\t")
                    if layout is None:
                        layout = column_layout([cell.strip() for cell in cells])
                    else:
                        if len(cells) != 4:
                            raise ValueError("SNC dose row must have four columns.")
                        axes, dose = layout
                        rows.append(
                            [
                                number(cells[axes[axis][0]]) * axes[axis][1]
                                for axis in ("X", "Y", "Z")
                            ]
                            + [number(cells[dose])]
                        )
                elif "\t" in line:
                    key, value = line.split("\t", 1)
                    (file_meta if scan_meta is None else scan_meta)[key.strip()] = value.rstrip(
                        "\t"
                    )
                    if scan_meta is not None and section is not None:
                        scan_meta["SNC Sections"].setdefault(section, []).append(key.strip())
                elif scan_meta is not None and marker in HEADER_SECTIONS:
                    section = marker
            except ValueError as error:
                raise ValueError(
                    f"Invalid SNC text file {path.name}, line {line_number}: {error}"
                ) from error
    if (
        not signature
        or not file_header
        or not first(file_meta, "File Version")
        or not first(file_meta, "File Scan Count")
        or scan_meta is not None
        or in_table
        or in_raw
    ):
        raise ValueError(f"Incomplete SNC text file: {path.name}")
    return file_meta, scans


class SNCSunScan(BeamData):
    """Read .snctxt/.sncxml dose tables and write .snctxt processed scans.

    XML ``layer`` may be ``"latest"`` (default), ``"raw"``, or a positive layer
    number. Text exports contain one processed table and support only ``"latest"``.
    """

    supported_acquisition_types = frozenset({AcquisitionType.Static})
    supported_measurement_types = frozenset(SCAN_NAMES)
    supported_radiation_types = frozenset(
        {None, RadiationType.PHOTON, RadiationType.ELECTRON, RadiationType.PROTON}
    )

    def __init__(self, file_path: str | Path, *, layer: str | int = "latest") -> None:
        super().__init__(file_path)
        if not (layer in ("latest", "raw") or (type(layer) is int and layer > 0)):
            raise ValueError("SNC layer must be 'latest', 'raw', or a positive layer number.")
        self.layer = layer

    def parse(self) -> SNCSunScan:
        path = Path(self.file_path)
        if path.suffix.lower() == ".sncxml":
            metadata, scans = read_xml(path, self.layer)
        elif path.suffix.lower() == ".snctxt":
            if self.layer != "latest":
                raise ValueError(
                    "SNC text exports contain only the processed dose table; use XML for layer selection."
                )
            metadata, scans = read_text(path)
        else:
            raise ValueError(f"Unsupported input format for SNCSunScan: {path.suffix}")
        try:
            declared_count = int(metadata["File Scan Count"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("SNC file must declare a valid nonnegative scan count.") from None
        if declared_count < 0:
            raise ValueError("SNC file must declare a valid nonnegative scan count.")
        if declared_count != len(scans):
            raise ValueError(
                f"SNC scan count mismatch: header declares {declared_count}, found {len(scans)}."
            )
        measurements = [measurement_from_table(*scan) for scan in scans]
        self.file_meta, self.measurements = metadata, measurements
        return self

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        if path.suffix.lower() != ".snctxt":
            raise ValueError(
                "Unsupported output format for SNCSunScan; use .snctxt (XML writing is not supported)."
            )
        header = {key: value for key, value in self.file_meta.items() if isinstance(value, str)}
        header.update(
            {
                "File Name": path.name,
                "File Date": datetime.now().strftime("%m/%d/%Y %H:%M"),
                "File Export Version": __version__,
                "File Version": "6.0",
                "File Scan Count": str(len(self.measurements)),
            }
        )
        lines = ["Tab-Delimited Scan Output", "FILE HEADER"]
        lines.extend(self._metadata_lines(header))
        for measurement in self.measurements:
            lines.extend(self._scan_lines(measurement))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _metadata_lines(metadata: dict) -> list[str]:
        result = []
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool)):
                continue
            if any(char in str(key) + str(value) for char in "\r\n") or "\t" in str(key):
                raise ValueError("SNC metadata keys and values must be single-line fields.")
            result.append(f"{key}\t{value}\t")
        return result

    def _scan_lines(self, measurement: Measurement) -> list[str]:
        kind = measurement.type
        if (
            measurement.acquisition_type not in self.supported_acquisition_types
            or kind is None
            or kind not in self.supported_measurement_types
        ):
            raise ValueError(
                "SNC output supports only static 1D measurements; "
                "select a frame and/or profile explicitly before exporting."
            )
        measurement.validate()
        scan = copy.copy(measurement)
        positions, doses = np.asarray(scan.positions, dtype=float), np.asarray(
            scan.doses, dtype=float
        )
        if scan.startpoint is None:
            raise ValueError("SNC output requires a startpoint.")
        if scan.endpoint is None and kind != MeasurementType.DiagonalProfile:
            endpoint = list(scan.startpoint)
            axis = {
                MeasurementType.CrosslineProfile: 0,
                MeasurementType.InlineProfile: 1,
                MeasurementType.DepthDose: 2,
            }[kind]
            endpoint[axis] = float(positions[-1])
            scan.endpoint = tuple(endpoint)
        xyz = np.column_stack(scan.reconstruct_xyz())
        if not np.all(np.isfinite(xyz)):
            raise ValueError("SNC output coordinates must be finite.")
        meta = dict(scan.metadata)
        mode = first(meta, "FluenceMode", "Rate Type")
        if scan.radiation_type == RadiationType.PHOTON and mode:
            mode = photon_fluence_mode(mode)
            meta["FluenceMode"] = mode
            meta["Rate Type"] = "FF" if mode == "Standard" else mode
        beam_type = scan.radiation_type.value if scan.radiation_type is not None else ""
        energy = "" if scan.energy is None else f"{scan.energy:.17g}"
        wedge = "" if scan.wedge_angle is None else f"{scan.wedge_angle:.17g}"
        meta.update(
            {
                "Summary Beam Type": beam_type,
                "Beam Type": beam_type,
                "Summary Energy (MV/MeV)": energy,
                "Energy (MV / MeV)": energy,
                "Summary Scan Type": SCAN_NAMES[kind],
                "Scan Type": SCAN_NAMES[kind],
                "Summary Wedge Angle (degrees)": wedge,
                "Wedge Angle (degrees)": wedge,
                "Source to Surface Distance (cm)": (
                    "" if scan.SSD is None else f"{scan.SSD / 10:.17g}"
                ),
                "Measurement Unit": "cm",
            }
        )
        for index, axis in enumerate(("X", "Y")):
            value = "" if scan.field_size is None else f"{scan.field_size[index] / 10:.17g}"
            meta[f"Summary FieldSize {axis} (cm)"] = value
            meta[f"Field Size {axis} (cm)"] = value
        action = meta.pop("Action", "Standard")
        sections = meta.pop("SNC Sections", {})
        summary = {
            key: meta.pop(key)
            for key in list(meta)
            if key.startswith("Summary ")
            or key in {"Beam Type", "Energy (MV / MeV)", "Rate Type", "Is PDI to PDD"}
        }
        lines = ["BEGIN SCAN", *self._metadata_lines(summary), "BEGIN DOSE TABLE"]
        lines.extend(self._metadata_lines({"Action": action}))
        lines.append("\tX (cm)\tY (cm)\tZ (cm)\tRelative Dose (%)")
        lines.extend(
            "\t" + "\t".join(f"{value:.17g}" for value in (*point / 10, dose))
            for point, dose in zip(xyz, doses)
        )
        lines.extend(["END DOSE TABLE", "SCAN HEADER"])
        for section in HEADER_SECTIONS:
            keys = sections.get(section, [])
            fields = {key: meta.pop(key) for key in keys if key in meta}
            if section == "DELIVERY SYSTEM":
                fields.update(
                    {
                        key: meta.pop(key)
                        for key in list(meta)
                        if key.startswith(
                            ("Field Size ", "Wedge ", "Gantry Angle", "Collimator Angle")
                        )
                        or key == "Measurement Unit"
                    }
                )
            elif section == "MEASUREMENT DETAILS":
                fields.update(meta)
            if fields:
                lines.extend([section, *self._metadata_lines(fields)])
        lines.append("END SCAN")
        return lines
