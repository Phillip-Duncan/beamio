"""IBA OmniPro/myQA OPG planes, OPD datasets and OPW workspaces.

Grid dimensions come from the file, independently of the MatriXX model. OPG
and OPD dose images use Gy and mm in the shared model. Binary non-dose images
(for example film scans and gamma results) remain in native metadata.
"""

from configparser import ConfigParser
from io import StringIO
from pathlib import Path, PureWindowsPath
import re
import struct
import warnings
import zlib

import numpy as np

from beamio.common.beamdata import BeamData
from beamio.common.measurement import AcquisitionType, Measurement, MeasurementType, RadiationType

_KEY = "IBA MatriXX"
_LENGTH = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
_DOSE = {"Gy": 1.0, "cGy": 0.01, "mGy": 0.001, "%": 1.0}
_RADIATION = {
    "Photons": RadiationType.PHOTON,
    "Electrons": RadiationType.ELECTRON,
    "Protons": RadiationType.PROTON,
}
_MAGIC = b"opdfile\0"


def _decode(raw):
    utf8 = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    for encoding in (utf8, "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("Unsupported IBA text encoding.")


def _fields(lines):
    result = {}
    for line in lines:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep or key.strip() in result:
            raise ValueError(f"Malformed or duplicate IBA header field: {line!r}")
        result[key.strip()] = value.strip()
    return result


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid IBA {label}: {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"IBA {label} must be finite.")
    return result


def _length(value, default="mm"):
    parts = value.split()
    if (
        len(parts) not in (1, 2)
        or (unit := parts[-1] if len(parts) == 2 else default) not in _LENGTH
    ):
        raise ValueError(f"Invalid IBA length: {value!r}")
    return _number(parts[0], "length") * _LENGTH[unit]


def _positive_int(value, label):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid IBA {label}.") from error
    if number < 1:
        raise ValueError(f"IBA {label} must be positive.")
    return number


def _version(value, format_name):
    if value != 4:
        warnings.warn(
            f"IBA {format_name} version {value} has not been validated; "
            "attempting to read it, but it may be unsupported or contain bugs.",
            UserWarning,
            stacklevel=3,
        )


def _check_text_fields(fields):
    for key, value in fields.items():
        if not isinstance(key, str) or not key or any(c in key for c in ":\r\n<>"):
            raise ValueError("Invalid IBA OPG field name.")
        if any(c in str(value) for c in "\r\n<>"):
            raise ValueError("IBA OPG field values cannot contain newlines or tags.")


def _rotation_warning(fields):
    for key, value in fields.items():
        if "rotation" in key.lower() and value.strip() not in ("", "0", "0.0"):
            warnings.warn(
                "IBA OPG rotation metadata is retained; coordinates are in "
                "the exported plane frame. Global rotation is not applied.",
                UserWarning,
                stacklevel=4,
            )


def _plane(doses, x, y, position=0.0, plane="XY"):
    m = Measurement()
    m.type = MeasurementType.DosePlane
    m.doses = doses
    m.axes = (y, x)
    col, row = ("XYZ".index(c) for c in plane)
    normal = next(i for i in range(3) if i not in (col, row))
    start = np.zeros(3)
    stop = np.zeros(3)
    start[[col, row, normal]] = x[0], y[0], position
    stop[[col, row, normal]] = x[-1], y[-1], position
    m.startpoint = tuple(start)
    m.endpoint = tuple(stop)
    if plane == "XY":
        return m
    xyz = np.zeros(doses.shape[-2:] + (3,))
    xyz[..., col] = x[None, :]
    xyz[..., row] = y[:, None]
    xyz[..., normal] = position
    m.coordinates = xyz
    return m


class IBAMatriXX(BeamData):
    """Read MatriXX/OmniPro exports without detector-specific array assumptions.

    ``doses`` contains Gy (or OPG percentages), with axes in mm. OPD Snap images with matching
    geometry form TimeSeries measurements, with elapsed seconds in ``times``.
    Native film/analysis images and unknown records are retained in file_meta.
    Saving OPG supports static planes; OPD/OPW writing requires native metadata.
    """

    supported_acquisition_types = frozenset({AcquisitionType.Static, AcquisitionType.TimeSeries})
    supported_measurement_types = frozenset({MeasurementType.DosePlane})
    supported_radiation_types = frozenset(
        {None, RadiationType.PHOTON, RadiationType.ELECTRON, RadiationType.PROTON}
    )

    def parse(self) -> "IBAMatriXX":
        raw = Path(self.file_path).read_bytes()
        if raw.startswith(_MAGIC):
            measurements, native = self._read_opd(raw)
        else:
            text, encoding = _decode(raw)
            if text.lstrip().startswith("<opimrtascii>"):
                measurements, native = self._read_opg(text)
            elif text.lstrip().startswith("[FileVersion]"):
                measurements, native = self._read_opw(text)
            else:
                raise ValueError("Not an IBA OPG, OPD or OPW file.")
            native["Encoding"] = encoding
        self.measurements = measurements
        self.file_meta = {_KEY: native}
        return self

    def save(self, output_path: str) -> None:
        """Serialize primary dose arrays, validating before opening destinations."""
        destination = Path(output_path)
        suffix = destination.suffix.lower()
        if suffix == ".opg":
            outputs = {destination: self._opg_output()}
        elif suffix == ".opd":
            outputs = {destination: self._opd_output()}
        elif suffix == ".opw":
            outputs = self._opw_output(destination)
        else:
            raise ValueError("IBA output must have an .opg, .opd or .opw extension.")
        for path, data in outputs.items():
            path.write_bytes(data)

    @staticmethod
    def _read_opg(text):
        # This labelled ASCII format resembles XML, but grid/header text is not XML.
        match = re.fullmatch(
            r"\s*<opimrtascii>\s*<asciiheader>(.*?)</asciiheader>" r"(.*?)</opimrtascii>\s*",
            text,
            re.S,
        )
        if match is None:
            raise ValueError("Malformed IBA OPG structure.")
        header = _fields(match[1].splitlines())
        required = (
            "File Version",
            "Data Factor",
            "Data Unit",
            "Length Unit",
            "No. of Columns",
            "No. of Rows",
            "Number of Bodies",
        )
        if any(key not in header for key in required):
            raise ValueError("Missing required IBA OPG header fields.")
        _version(int(header["File Version"]), "OPG")
        if header.get("Separator", "[TAB]") != "[TAB]":
            raise ValueError("Only tab-separated IBA OPG grids are supported.")
        unit = header["Length Unit"]
        dose_unit = header["Data Unit"]
        if unit not in _LENGTH or dose_unit not in _DOSE:
            raise ValueError(f"Unsupported IBA OPG units: {unit!r}, {dose_unit!r}.")
        factor = _number(header["Data Factor"], "Data Factor")
        if factor * _DOSE[dose_unit] <= 0:
            raise ValueError("IBA Data Factor must be positive.")
        plane = header.get("Plane", "XY").upper()
        if plane not in ("XY", "XZ", "YX", "YZ", "ZX", "ZY"):
            raise ValueError(f"Unsupported IBA OPG plane: {plane!r}.")
        _rotation_warning(header)
        nx = _positive_int(header["No. of Columns"], "column count")
        ny = _positive_int(header["No. of Rows"], "row count")
        bodies = re.findall(r"<asciibody>(.*?)</asciibody>", match[2], re.S)
        remainder = re.sub(r"<asciibody>.*?</asciibody>", "", match[2], flags=re.S)
        if remainder.strip() or len(bodies) != _positive_int(
            header["Number of Bodies"], "body count"
        ):
            raise ValueError("IBA OPG body count or structure does not match its header.")
        measurements = []
        for body in bodies:
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            axis = re.compile(r"([XYZ])\[\s*(mm|cm|m)\s*\]")
            index = next((i for i, line in enumerate(lines) if axis.match(line)), None)
            if index is None or index + 1 >= len(lines):
                raise ValueError("Missing IBA OPG coordinate labels.")
            properties = _fields(lines[:index])
            _rotation_warning(properties)
            columns = lines[index].split()
            rows = lines[index + 1].split()
            if (
                columns[0] != f"{plane[0]}[{unit}]"
                or rows != [f"{plane[1]}[{unit}]"]
                or len(columns) != nx + 1
                or len(lines[index + 2 :]) != ny
            ):
                raise ValueError("IBA OPG coordinate labels or grid dimensions do not match.")
            try:
                x = np.array(columns[1:], dtype=float) * _LENGTH[unit]
                table = np.array([line.split() for line in lines[index + 2 :]], dtype=float)
            except ValueError as error:
                raise ValueError("Invalid numeric IBA OPG grid.") from error
            if table.shape != (ny, nx + 1):
                raise ValueError("IBA OPG dose row width does not match its header.")
            position = _length(properties.get("Plane Position", "0"), unit)
            m = _plane(
                table[:, 1:] * factor * _DOSE[dose_unit],
                x,
                table[:, 0] * _LENGTH[unit],
                position,
                plane,
            )
            radiation = header.get("Radiation Type", "")
            if radiation and radiation not in _RADIATION:
                warnings.warn(
                    f"Unknown IBA radiation type {radiation!r}; retained in metadata.",
                    UserWarning,
                    stacklevel=3,
                )
            m.radiation_type = _RADIATION.get(radiation)
            if header.get("Energy"):
                m.energy = _number(header["Energy"].split()[0], "energy")
            if header.get("SSD"):
                m.SSD = _length(header["SSD"], unit)
            if header.get("Field Size Cr") and header.get("Field Size In"):
                m.field_size = tuple(
                    _length(header[k], unit) for k in ("Field Size Cr", "Field Size In")
                )
            m.metadata = {
                "DoseUnit": "%" if dose_unit == "%" else "Gy",
                _KEY: {"Format": "OPG", "Body": properties, "Plane": plane},
            }
            m.validate()
            measurements.append(m)
        return measurements, {"Format": "OPG", "Header": header}

    @staticmethod
    def _read_opd(raw):
        if len(raw) < 22 or raw[:8] != _MAGIC:
            raise ValueError("Truncated or invalid IBA OPD header.")
        if zlib.crc32(raw[:-4]) != struct.unpack_from("<I", raw, len(raw) - 4)[0]:
            raise ValueError("IBA OPD checksum does not match file contents.")
        version = struct.unpack_from("<I", raw, 10)[0]
        _version(version, "OPD")
        count = struct.unpack_from("<H", raw, 16)[0]
        images = []
        current = None
        offset = 18
        end = len(raw) - 4
        while offset < end:
            if offset + 6 > end:
                raise ValueError("Truncated IBA OPD record header.")
            kind, size = struct.unpack_from("<HI", raw, offset)
            finish = offset + 6 + size
            if finish > end:
                raise ValueError("Truncated IBA OPD record payload.")
            payload = raw[offset + 6 : finish]
            if kind == 2:
                if size != 105 or (current is not None and "Grid" not in current):
                    raise ValueError("Unsupported IBA OPD image header or missing grid.")
                current = {
                    "Records": [],
                    "Name": payload[:25].split(b"\0")[0].decode("cp1252"),
                    "SignalType": struct.unpack_from("<I", payload, 57)[0],
                }
                images.append(current)
            if current is None:
                raise ValueError("IBA OPD records precede the first image header.")
            current["Records"].append((kind, payload))
            offset = finish
            if kind == 6:
                if (
                    size != 151
                    or "Grid" in current
                    or struct.unpack_from("<I", payload, 87)[0] != 4
                ):
                    raise ValueError("Unsupported IBA OPD pixel layout.")
                nx, ny = struct.unpack_from("<HH", payload, 81)
                pixels_end = finish + nx * ny * 4
                if not nx or not ny or pixels_end > end:
                    raise ValueError("Truncated or empty IBA OPD dose grid.")
                bounds = (
                    *struct.unpack_from("<dd", payload, 95),
                    *struct.unpack_from("<dd", payload, 113),
                )
                current["Grid"] = (nx, ny, bounds)
                current["Pixels"] = (
                    np.frombuffer(raw, dtype="<u4", count=nx * ny, offset=finish)
                    .reshape(ny, nx)
                    .copy()
                )
                offset = pixels_end
        if len(images) != count or any("Grid" not in image for image in images):
            raise ValueError("IBA OPD image count or grids do not match its header.")
        measurements = []
        pending = []

        def append_group(indices):
            if not indices:
                return
            first = images[indices[0]]
            nx, ny, (xmin, xmax, ymin, ymax) = first["Grid"]
            arrays = []
            dates = []
            for index in indices:
                image = images[index]
                info = image["Records"][0][1]
                factor = _number(struct.unpack_from("<d", info, 61)[0], "OPD dose factor")
                if factor <= 0:
                    raise ValueError("IBA OPD dose factor must be positive.")
                pixels = image.pop("Pixels")
                flags = (pixels & np.uint32(0x80000000)) != 0
                if flags.any():
                    image["PixelFlags"] = flags
                arrays.append((pixels & np.uint32(0x7FFFFFFF)).astype(float) * (factor / 1000))
                dates.append(_number(struct.unpack_from("<d", info, 49)[0], "OPD date"))
            movie = first["Name"].startswith("Snap ")
            m = _plane(
                np.stack(arrays) if movie else arrays[0],
                np.linspace(xmin, xmax, nx),
                np.linspace(ymax, ymin, ny),
            )
            if movie:
                m.acquisition_type = AcquisitionType.TimeSeries
                m.times = (np.asarray(dates) - dates[0]) * 86400
            setup = next((data for kind, data in first["Records"] if kind == 3), None)
            if setup is not None and len(setup) == 86:
                # Code 0 is confirmed by the paired photon OPD/OPG exports.
                # Other binary radiation codes have not been validated.
                if struct.unpack_from("<H", setup, 41)[0] == 0:
                    m.radiation_type = RadiationType.PHOTON
                m.energy = _number(struct.unpack_from("<d", setup, 43)[0], "OPD energy")
                fields = struct.unpack_from("<4hH", setup, 31)
                m.field_size = ((fields[1] - fields[0]) / 10, (fields[3] - fields[2]) / 10)
                m.SSD = fields[4] / 10
            m.metadata = {"DoseUnit": "Gy", _KEY: {"Format": "OPD", "ImageIndices": tuple(indices)}}
            m.validate()
            measurements.append(m)

        for index, image in enumerate(images):
            snap = image["SignalType"] == 1 and image["Name"].startswith("Snap ")
            if pending and (
                not snap
                or image["Grid"] != images[pending[0]]["Grid"]
                or [r[1][31:51] for r in image["Records"] if r[0] == 3]
                != [r[1][31:51] for r in images[pending[0]]["Records"] if r[0] == 3]
            ):
                append_group(pending)
                pending = []
            if snap:
                pending.append(index)
            elif image["SignalType"] == 1:
                append_group([index])
        append_group(pending)
        if not measurements:
            warnings.warn(
                "IBA OPD contains no dose images; non-dose images are retained in "
                "file_meta['IBA MatriXX']['Images'].",
                UserWarning,
                stacklevel=3,
            )
        return measurements, {"Format": "OPD", "Header": raw[:18], "Images": images}

    def _read_opw(self, text):
        config = ConfigParser(interpolation=None)
        config.optionxform = str
        config.read_string(text)
        sections = {section: dict(config[section]) for section in config.sections()}
        if config.get("FileVersion", "Version") != "1":
            warnings.warn(
                "Unvalidated IBA OPW version; attempting to read it.", UserWarning, stacklevel=3
            )
        datasets = []
        measurements = []
        source = Path(self.file_path)
        for section, properties in sections.items():
            if not (re.fullmatch(r"DataSet\d+", section) or section == "ResultSet"):
                continue
            stored = properties.get("Path", "")
            if not stored:
                raise ValueError(f"Missing IBA OPW {section} Path.")
            reference = Path(stored.replace("\\", "/"))
            foreign_absolute = PureWindowsPath(stored).is_absolute() and not reference.is_absolute()
            path = reference if reference.is_absolute() else source.parent / reference
            if foreign_absolute or not path.is_file():
                suffix = "res" if section == "ResultSet" else section.removeprefix("DataSet")
                candidates = (
                    source.parent / PureWindowsPath(stored).name,
                    source.with_name(f"{source.stem}_{suffix}.opd"),
                )
                path = next((candidate for candidate in candidates if candidate.is_file()), None)
                if path is None:
                    raise FileNotFoundError(
                        f"IBA OPW {section} references missing dataset {stored!r}."
                    )
                warnings.warn(
                    f"IBA OPW {section}: using {path.name!r} because stored path "
                    f"{stored!r} is unavailable.",
                    UserWarning,
                    stacklevel=3,
                )
            raw = path.read_bytes()
            if not raw.startswith(_MAGIC):
                raise ValueError(f"IBA OPW {section} dataset {path.name!r} must be an OPD file.")
            child_measurements, child_metadata = self._read_opd(raw)
            for m in child_measurements:
                m.metadata[_KEY]["WorkspaceDataset"] = section
            measurements.extend(child_measurements)
            datasets.append({"Section": section, "Metadata": child_metadata})
        if not datasets:
            raise ValueError("IBA OPW contains no dataset references.")
        return measurements, {"Format": "OPW", "Sections": sections, "Datasets": datasets}

    def _opg_output(self):
        if not self.measurements:
            raise ValueError("IBA OPG requires at least one static dose plane.")
        native = self.file_meta.get(_KEY, {})
        header = dict(native.get("Header", {})) if native.get("Format") == "OPG" else {}
        header.setdefault("File Version", "4")
        header["Separator"] = "[TAB]"
        if any(m.metadata.get("DoseUnit") not in ("Gy", "%") for m in self.measurements):
            raise ValueError("IBA OPG writing requires explicit DoseUnit metadata (Gy or %).")
        unit = header.setdefault("Length Unit", "mm")
        dose_unit = header.setdefault("Data Unit", self.measurements[0].metadata["DoseUnit"])
        if unit not in _LENGTH or dose_unit not in _DOSE:
            raise ValueError("Unsupported IBA OPG output units.")
        factor = _number(header.setdefault("Data Factor", "1"), "Data Factor")
        if factor * _DOSE[dose_unit] <= 0:
            raise ValueError("IBA Data Factor must be positive.")
        bodies = []
        common = None
        for m in self.measurements:
            m.validate()
            if (
                m.type != MeasurementType.DosePlane
                or m.acquisition_type != AcquisitionType.Static
                or m.radiation_type not in self.supported_radiation_types
            ):
                raise ValueError("IBA OPG writing requires supported static dose planes.")
            if not np.all(np.isfinite(m.doses)):
                raise ValueError("IBA OPG output doses must be finite.")
            if m.wedge_angle is not None:
                raise ValueError("IBA OPG wedge angle writing is not supported.")
            if m.metadata["DoseUnit"] != ("%" if dose_unit == "%" else "Gy"):
                raise ValueError("IBA OPG dose units do not match Measurement metadata.")
            plane = m.metadata.get(_KEY, {}).get("Plane", "XY")
            if plane not in ("XY", "XZ", "YX", "YZ", "ZX", "ZY"):
                raise ValueError("Unsupported IBA OPG output plane.")
            xyz = m.reconstruct_xyz()
            col, row = ("XYZ".index(c) for c in plane)
            normal = next(i for i in range(3) if i not in (col, row))
            x, y = m.axes[1], m.axes[0]
            if (
                not np.allclose(xyz[col], x[None, :], rtol=0, atol=1e-8)
                or not np.allclose(xyz[row], y[:, None], rtol=0, atol=1e-8)
                or not np.allclose(xyz[normal], xyz[normal][0, 0], rtol=0, atol=1e-8)
            ):
                raise ValueError("IBA OPG requires coordinates aligned with its plane axes.")
            signature = (m.energy, m.radiation_type, m.SSD, m.field_size, m.spatial_shape, plane)
            if common is not None and signature != common:
                raise ValueError("IBA OPG bodies must share beam metadata, shape and plane.")
            common = signature
            props = dict(m.metadata.get(_KEY, {}).get("Body", {}))
            props["Plane Position"] = f"{xyz[normal][0, 0]/_LENGTH[unit]:.17g} {unit}"
            _check_text_fields(props)
            lines = [
                "<asciibody>",
                *(f"{k}: {v}" for k, v in props.items()),
                f"{plane[0]}[{unit}]\t" + "\t".join(f"{v/_LENGTH[unit]:.17g}" for v in x),
                f"{plane[1]}[{unit}]",
            ]
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = m.doses / (factor * _DOSE[dose_unit])
            if not np.all(np.isfinite(scaled)):
                raise ValueError("IBA OPG scaled output values must be finite.")
            for position, values in zip(y, scaled):
                lines.append("\t".join(f"{v:.17g}" for v in (position / _LENGTH[unit], *values)))
            bodies.append("\r\n".join([*lines, "</asciibody>"]))
        m = self.measurements[0]
        header.update(
            {
                "Plane": common[-1],
                "No. of Columns": str(m.spatial_shape[1]),
                "No. of Rows": str(m.spatial_shape[0]),
                "Number of Bodies": str(len(bodies)),
            }
        )
        if m.energy is None:
            header["Energy"] = ""
        else:
            source_energy = header.get("Energy", "").split(maxsplit=1)
            source_radiation = _RADIATION.get(header.get("Radiation Type", ""))
            if len(source_energy) == 2 and (
                m.radiation_type is None or m.radiation_type is source_radiation
            ):
                energy_unit = source_energy[1]
            elif m.radiation_type is RadiationType.PHOTON:
                energy_unit = "MV"
            elif m.radiation_type in (RadiationType.ELECTRON, RadiationType.PROTON):
                energy_unit = "MeV"
            else:
                raise ValueError(
                    "IBA OPG energy with undeclared radiation requires a retained source energy unit."
                )
            header["Energy"] = f"{_number(m.energy, 'energy'):.17g} {energy_unit}"
        if m.radiation_type is not None:
            header["Radiation Type"] = next(
                k for k, v in _RADIATION.items() if v == m.radiation_type
            )
        elif header.get("Radiation Type") in _RADIATION:
            header["Radiation Type"] = ""
        for key, value in (
            ("SSD", m.SSD),
            ("Field Size Cr", None if m.field_size is None else m.field_size[0]),
            ("Field Size In", None if m.field_size is None else m.field_size[1]),
        ):
            header[key] = (
                "" if value is None else f"{_number(value, key)/_LENGTH[unit]:.17g} {unit}"
            )
        header.setdefault("Data Type", "Relative Dose" if dose_unit == "%" else "Abs. Dose")
        _check_text_fields(header)
        output = "\r\n".join(
            [
                "<opimrtascii>",
                "<asciiheader>",
                *(f"{k}: {v}" for k, v in header.items()),
                "</asciiheader>",
                *bodies,
                "</opimrtascii>",
                "",
            ]
        )
        return output.encode(native.get("Encoding", "utf-8"))

    def _opd_output(self):
        native = self.file_meta.get(_KEY, {})
        if native.get("Format") != "OPD":
            raise ValueError("IBA OPD writing requires retained native OPD metadata.")
        images = native["Images"]
        mapped = {}
        for m in self.measurements:
            m.validate()
            if m.type != MeasurementType.DosePlane or m.metadata.get("DoseUnit") != "Gy":
                raise ValueError("IBA OPD requires dose planes in Gy.")
            indices = m.metadata.get(_KEY, {}).get("ImageIndices", ())
            frames = m.doses if m.acquisition_type == AcquisitionType.TimeSeries else m.doses[None]
            if len(indices) != len(frames):
                raise ValueError("IBA OPD frame count differs from native image metadata.")
            if m.acquisition_type == AcquisitionType.TimeSeries and m.times[0] != 0:
                raise ValueError("IBA OPD elapsed times must start at zero.")
            for frame, (index, values) in enumerate(zip(indices, frames)):
                if index in mapped or not isinstance(index, int) or not 0 <= index < len(images):
                    raise ValueError("Invalid or duplicated IBA OPD native image index.")
                snap = images[index]["Name"].startswith("Snap ")
                if snap != (m.acquisition_type == AcquisitionType.TimeSeries):
                    raise ValueError("IBA OPD acquisition type differs from its native image.")
                mapped[index] = (m, frame, values)
        expected = {i for i, image in enumerate(images) if image["SignalType"] == 1}
        if set(mapped) != expected:
            raise ValueError("IBA OPD requires all original dose images exactly once.")
        output = bytearray(native["Header"])
        for index, image in enumerate(images):
            records = [(kind, bytearray(data)) for kind, data in image["Records"]]
            nx, ny, (xmin, xmax, ymin, ymax) = image["Grid"]
            if index in mapped:
                m, frame, values = mapped[index]
                if (
                    values.shape != (ny, nx)
                    or m.coordinates is not None
                    or not np.allclose(m.axes[1], np.linspace(xmin, xmax, nx), rtol=0, atol=1e-8)
                    or not np.allclose(m.axes[0], np.linspace(ymax, ymin, ny), rtol=0, atol=1e-8)
                    or any(
                        point is not None and point[2] != 0 for point in (m.startpoint, m.endpoint)
                    )
                ):
                    raise ValueError("IBA OPD native grid geometry cannot be changed.")
                setup = next((data for kind, data in image["Records"] if kind == 3), None)
                radiation = (
                    RadiationType.PHOTON
                    if setup is not None
                    and len(setup) == 86
                    and struct.unpack_from("<H", setup, 41)[0] == 0
                    else None
                )
                if m.radiation_type is not radiation or m.wedge_angle is not None:
                    raise ValueError("IBA OPD radiation and wedge editing is not supported.")
                if setup is None and any(v is not None for v in (m.energy, m.SSD, m.field_size)):
                    raise ValueError("IBA OPD beam edits require a native beam setup record.")
                info = records[0][1]
                factor = _number(struct.unpack_from("<d", info, 61)[0], "OPD dose factor") / 1000
                if factor <= 0:
                    raise ValueError("IBA OPD dose factor must be positive.")
                with np.errstate(over="ignore", invalid="ignore"):
                    counts = np.rint(np.asarray(values, dtype=float) / factor)
                if (
                    not np.all(np.isfinite(counts))
                    or np.any(values < 0)
                    or np.any(counts > 0x7FFFFFFF)
                ):
                    raise ValueError("IBA OPD doses must fit its nonnegative 31-bit scaled counts.")
                pixels = counts.astype("<u4")
                if "PixelFlags" in image:
                    pixels |= np.asarray(image["PixelFlags"], dtype=np.uint32) * np.uint32(
                        0x80000000
                    )
                if m.acquisition_type == AcquisitionType.TimeSeries:
                    first = images[m.metadata[_KEY]["ImageIndices"][0]]["Records"][0][1]
                    origin = struct.unpack_from("<d", first, 49)[0]
                    struct.pack_into("<d", info, 49, origin + float(m.times[frame]) / 86400)
                for kind, payload in records:
                    if kind == 3:
                        self._update_opd_setup(payload, m)
            else:
                pixels = image["Pixels"]
                if (
                    np.shape(pixels) != (ny, nx)
                    or np.asarray(pixels).dtype.kind not in "biuf"
                    or not np.all(np.isfinite(pixels))
                    or np.any(pixels < 0)
                    or np.any(pixels > 0xFFFFFFFF)
                    or np.any(pixels != np.rint(pixels))
                ):
                    raise ValueError("IBA OPD non-dose pixels must match their native uint32 grid.")
            for kind, payload in records:
                output.extend(struct.pack("<HI", kind, len(payload)))
                output.extend(payload)
                if kind == 6:
                    output.extend(np.asarray(pixels, dtype="<u4").tobytes())
        output.extend(struct.pack("<I", zlib.crc32(output)))
        return bytes(output)

    @staticmethod
    def _update_opd_setup(payload, m):
        if len(payload) != 86:
            raise ValueError("Unsupported IBA OPD beam setup record.")
        if m.energy is None or m.SSD is None or m.field_size is None:
            raise ValueError("IBA OPD beam setup fields cannot be removed.")
        struct.pack_into("<d", payload, 43, _number(m.energy, "energy"))
        old = struct.unpack_from("<4hH", payload, 31)
        fields = []
        for start, stop, width in zip(old[::2][:2], old[1::2][:2], m.field_size):
            center = (start + stop) / 2
            half = _number(width, "field size") * 5
            fields.extend((center - half, center + half))
        fields.append(_number(m.SSD, "SSD") * 10)
        if (
            any(abs(v - round(v)) > 1e-7 for v in fields)
            or any(not -32768 <= v <= 32767 for v in fields[:4])
            or not 0 <= fields[4] <= 65535
            or any(v < 0 for v in m.field_size)
        ):
            raise ValueError("IBA OPD beam setup is outside native range or precision.")
        struct.pack_into("<4hH", payload, 31, *(round(v) for v in fields))

    def _opw_output(self, destination):
        native = self.file_meta.get(_KEY, {})
        if native.get("Format") != "OPW":
            raise ValueError("IBA OPW writing requires retained native workspace metadata.")
        sections = {key: dict(value) for key, value in native["Sections"].items()}
        outputs = {}
        assigned = set()
        for dataset in native["Datasets"]:
            section = dataset["Section"]
            suffix = "res" if section == "ResultSet" else section.removeprefix("DataSet")
            path = destination.with_name(f"{destination.stem}_{suffix}.opd")
            child = IBAMatriXX("")
            child.file_meta = {_KEY: dataset["Metadata"]}
            child.measurements = [
                m
                for m in self.measurements
                if m.metadata.get(_KEY, {}).get("WorkspaceDataset") == section
            ]
            assigned.update(id(m) for m in child.measurements)
            outputs[path] = child._opd_output()
            sections[section]["Path"] = path.name
        if len(assigned) != len(self.measurements):
            raise ValueError("IBA OPW measurements must belong to a retained dataset.")
        config = ConfigParser(interpolation=None)
        config.optionxform = str
        config.read_dict(sections)
        stream = StringIO(newline="\r\n")
        config.write(stream, space_around_delimiters=False)
        outputs[destination] = stream.getvalue().encode(native.get("Encoding", "utf-8"))
        return outputs
