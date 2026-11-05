"""ArcCHECK raw ASCII movies and processed binary movie-lite records."""

from pathlib import Path
import re
import struct
import warnings
import zlib

import numpy as np

from beamio.common.measurement import AcquisitionType, Measurement, MeasurementType, RadiationType

LITE_SIGNATURE = b"ArcCHECK Measurement file: Header, Definition, Data"
_ACM_SIGNATURE = re.compile(
    r"(?:\d+[A-Za-z0-9-]*\s+)?ArcCHECK(?:-MR)? Measurement file: Header, Definition, Data", re.I
)
_SECTIONS = {
    "ArcCHECK Setup",
    "Collector Config",
    "LINAC Configuration",
    "Hardware Data",
    "Dose Calibration",
    "Measurement Corrections",
}
_LITE_AUX = np.dtype(
    [
        ("TIMETIC1", "<f4"),
        ("TIMETIC2", "<f4"),
        ("STATUS1", "<u4"),
        ("STATUS2", "<u4"),
        ("Gantry Angle", "<f4"),
        ("Ref Channel", "<u4"),
    ]
)
_LITE_LABELS = (
    "TIMETIC1",
    "TIMETIC2",
    "STATUS1",
    "STATUS2",
    "Gantry Angle (ArcCHECK Coordinate System)",
    "Ref Channel",
)


def is_acm(text):
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return _ACM_SIGNATURE.fullmatch(first) is not None


def is_acml(raw):
    return len(raw) >= 100 and raw[:100].rstrip(b"\0") == LITE_SIGNATURE


def _decode(raw):
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        return raw.decode(encoding), encoding
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252"


def _cells(line):
    return [cell.strip() for cell in line.rstrip("\t ").split("\t")]


def _finite(values, label):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"ArcCHECK {label} must be finite numeric values.")
    return array


def _integer(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid ArcCHECK {label}.") from None
    if not np.isfinite(number) or number < 1 or number % 1:
        raise ValueError(f"ArcCHECK {label} must be a positive integer.")
    return int(number)


def _numbers(cells, label):
    try:
        return _finite(np.array(cells, dtype=float), label)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid ArcCHECK {label}.") from None


def _warn_revision(revision, expected, kind):
    if not revision:
        raise ValueError(f"ArcCHECK {kind} requires a file revision.")
    if revision != expected:
        warnings.warn(
            f"ArcCHECK {kind} revision {revision!r} is unvalidated (tested {expected}); "
            "attempting the known schema, which may be unsupported or contain bugs.",
            UserWarning,
            stacklevel=3,
        )


def _verify_crc(raw):
    end = len(raw)
    while end and raw[end - 1] in b"\r\n\t ":
        end -= 1
    start = max(raw.rfind(b"\n", 0, end), raw.rfind(b"\r", 0, end)) + 1
    token = raw[start:end].strip()
    if re.fullmatch(rb"[0-9a-fA-F]{1,8}", token) and zlib.crc32(raw[:start]) != int(token, 16):
        raise ValueError("ArcCHECK ACM checksum does not match file contents.")


def _geometry_source(raw):
    # Lite needs the ACM header, not a second large raw-count array.
    start = re.search(rb"(?m)^Data:\t", raw)
    if start is None:
        raise ValueError("ArcCHECK ACM geometry source has no data records.")
    text, _ = _decode(raw[: start.start()])
    sections, _, ids, geometry, _ = _acm_layout(text.splitlines())
    _warn_revision(sections["Header"].get("File Revision", ""), "D", "ACM")
    _verify_crc(raw)
    serial = sections.get("Collector Config", {}).get("Collector Serial", "").split("\t")[0]
    return serial, len(ids), geometry


def _acm_layout(lines):
    if not is_acm("\n".join(lines[:2])):
        raise ValueError("Expected an ArcCHECK ACM movie signature.")
    first_data = next((i for i, line in enumerate(lines) if line.startswith("Data:\t")), len(lines))
    header = lines[:first_data]
    sections = {"Header": {}}
    target = sections["Header"]
    definitions = {}
    rows = {}
    type_cells = detector_cells = None
    for line in header[1:]:
        cells = _cells(line)
        if len(cells) > 1 and not cells[0] and cells[1] in _SECTIONS:
            name = cells[1]
            if name in sections:
                raise ValueError(f"Duplicate ArcCHECK ACM section {name}.")
            target = sections[name] = {}
        elif cells[0].endswith(":"):
            key = cells[0][:-1]
            if key in target:
                raise ValueError(f"Duplicate ArcCHECK ACM header field {key}.")
            target[key] = "\t".join(cells[1:9]).rstrip("\t")
        if len(cells) > 12 and cells[11] in ("NC =", "SC =", "SR =", "L =", "OO ="):
            definitions[cells[11].rstrip(" =")] = float(cells[12])
        if len(cells) > 9 and cells[9] in ("z(n)", "X(n)", "Y(n) cm", "Detector Number"):
            rows[cells[9]] = cells
            if cells[9] == "Detector Number":
                detector_cells = cells
        if cells[0] == "TYPE":
            if type_cells is not None:
                raise ValueError("Duplicate ArcCHECK ACM TYPE row.")
            type_cells = cells
        if cells[0] in ("Background", "Calibration"):
            if cells[0] in rows:
                raise ValueError("Duplicate ArcCHECK ACM correction row.")
            rows[cells[0]] = cells
    if type_cells is None or detector_cells is None or len(type_cells) != 10:
        raise ValueError("ArcCHECK ACM requires its TYPE and Detector Number rows.")
    expected = (
        "UPDATE#",
        "TIMETIC1",
        "TIMETIC2",
        "PULSES",
        "STATUS1",
        "STATUS2",
        "VirtualInclinometer",
        "CorrectedAngle",
        "FieldSize",
    )
    if tuple(type_cells[1:]) != expected or detector_cells[10] != "Reference Diode":
        raise ValueError("Unsupported ArcCHECK ACM auxiliary columns.")
    ids = tuple(detector_cells[11:])
    count = len(ids)
    if not count or ids != tuple(str(i + 1) for i in range(count)):
        raise ValueError("ArcCHECK ACM requires complete ordered detector identifiers.")
    if not {"NC", "SC", "SR", "L", "OO"}.issubset(definitions):
        raise ValueError("ArcCHECK ACM requires native NC/SC/SR/L/OO layout definitions.")
    _finite(list(definitions.values()), "layout definitions")
    nc = _integer(definitions["NC"], "detectors per turn")
    if count % nc or min(definitions.values()) <= 0:
        raise ValueError("ArcCHECK ACM detector count does not match its helical layout.")
    xyz = []
    for label in ("X(n)", "Y(n) cm", "z(n)"):
        if label not in rows or len(rows[label]) != count + 11:
            raise ValueError("ArcCHECK ACM has missing or inconsistent XYZ detector coordinates.")
        xyz.append(_numbers(rows[label][11:], label) * 10)
    coordinates = np.stack(xyz, axis=-1).reshape(count // nc, nc, 3)
    x = (definitions["SC"] * (np.arange(nc) + 1 / definitions["OO"]) - definitions["L"] / 2) * 10
    y = (np.arange(count // nc) - (count // nc - 1) / 2) * definitions["SR"] * 10
    corrections = {}
    for key in ("Background", "Calibration"):
        if key not in rows or len(rows[key]) != count + 11:
            raise ValueError(f"ArcCHECK ACM requires a complete {key} row.")
        corrections[key] = _numbers(rows[key][10:], key)
    return (
        sections,
        tuple(type_cells[1:]) + ("Reference Diode",),
        ids,
        (y, x, coordinates),
        corrections,
    )


def _measurement(doses, times, geometry, unit, source):
    y, x, xyz = geometry
    m = Measurement()
    m.type = MeasurementType.DosePlane
    m.acquisition_type = AcquisitionType.TimeSeries
    m.doses = doses.reshape((len(times), len(y), len(x)))
    m.axes = (y.copy(), x.copy())
    m.coordinates = xyz.copy()
    m.startpoint, m.endpoint = tuple(xyz[0, 0]), tuple(xyz[-1, -1])
    m.times = np.asarray(times, dtype=float)
    m.radiation_type = RadiationType.PHOTON
    m.metadata = {
        "DoseUnit": unit,
        "Cumulative": True,
        "Interpolation": "none (native detector grid)",
        "ArcCHECK": {"Source": source, "Coordinate System": "detector helix; unwrapped axes (mm)"},
    }
    m.validate()
    return m


def read_acm(raw):
    text, encoding = _decode(raw)
    lines = text.splitlines()
    sections, columns, ids, geometry, corrections = _acm_layout(lines)
    revision = sections["Header"].get("File Revision", "")
    _warn_revision(revision, "D", "ACM")
    first = next((i for i, line in enumerate(lines) if line.startswith("Data:\t")), None)
    if first is None:
        raise ValueError("ArcCHECK ACM requires at least one Data row.")
    frames = []
    checksum = None
    for i, line in enumerate(lines[first:], first):
        if not line.strip():
            continue
        if re.fullmatch(r"[0-9a-fA-F]{1,8}", line.strip()) and not any(
            l.strip() for l in lines[i + 1 :]
        ):
            checksum = line.strip()
            break
        if not line.startswith("Data:\t"):
            raise ValueError("Unexpected ArcCHECK ACM data row.")
        try:
            values = np.array(line.rstrip("\t ").split("\t")[1:], dtype=float)
        except ValueError:
            raise ValueError("Invalid numeric ArcCHECK ACM data row.") from None
        if values.shape != (len(columns) + len(ids),):
            raise ValueError("ArcCHECK ACM data row width differs from its detector columns.")
        frames.append(values)
    data = _finite(np.stack(frames), "ACM movie data")
    updates = _integer(
        sections.get("Measurement Corrections", {}).get("Number of Updates"), "Number of Updates"
    )
    if len(data) != updates:
        raise ValueError("ArcCHECK ACM Number of Updates differs from its Data row count.")
    if checksum is not None:
        _verify_crc(raw)
    aux = data[:, : len(columns)].copy()
    signal = data[:, len(columns) :].copy()
    m = _measurement(
        signal, aux[:, columns.index("TIMETIC1")] / 1000, geometry, "counts", "ACM raw counts"
    )
    energy = sections.get("Measurement Corrections", {}).get("Energy", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:MV|X)?\s*(FFF|FF)?", energy, re.I)
    if match is not None:
        m.energy = float(match[1])
        if match[2]:
            m.metadata["FluenceMode"] = match[2].upper()
    native = {
        "Format": "ACM",
        "Header Lines": tuple(lines[:first]),
        "Encoding": encoding,
        "Auxiliary Columns": columns,
        "Auxiliary Data": aux,
        "Detector IDs": ids,
        "Corrections": corrections,
        "Geometry": geometry,
        "Checksum": checksum,
        "Beam Energy": m.energy,
        "FluenceMode": m.metadata.get("FluenceMode", ""),
    }
    return {"File Type": "Raw Movie Data", "Sections": sections, "ArcCHECK": native}, m


def _label(raw, offset, expected):
    if raw[offset : offset + 50].split(b"\0", 1)[0].decode("ascii", errors="replace") != expected:
        raise ValueError(f"Unsupported ArcCHECK ACML layout: expected {expected!r}.")


def _lite_layout(raw):
    if not is_acml(raw) or len(raw) < 1382:
        raise ValueError("Truncated or invalid ArcCHECK ACML header.")
    strings = {}
    for offset, key in (
        (100, "File Revision"),
        (200, "Date"),
        (300, "Time"),
        (400, "Software Version"),
        (500, "Firmware Version"),
        (600, "Hardware Revision"),
        (700, "ArcCHECK Serial"),
        (800, "Temperature"),
        (900, "Diode Type"),
    ):
        _label(raw, offset, key + ":")
        strings[key] = raw[offset + 50 : offset + 100].split(b"\0", 1)[0].decode("cp1252")
    _warn_revision(strings["File Revision"], "C", "ACML")
    scalars = {}
    for offset, label, dtype in (
        (1000, "Dose Per Count:", "d"),
        (1058, "Inclinomer Tilt:", "f"),
        (1112, "Inclinomer Rotate:", "f"),
        (1166, "Cavity Plug Present:", "i"),
        (1220, "Y-Shift (mm):", "f"),
        (1274, "# Updates:", "I"),
        (1328, "# Detectors:", "I"),
    ):
        _label(raw, offset, label)
        scalars[label[:-1]] = struct.unpack_from("<" + dtype, raw, offset + 50)[0]
    _finite(list(scalars.values()), "ACML scalar headers")
    n = _integer(scalars["# Detectors"], "ACML detector count")
    nt = _integer(scalars["# Updates"], "ACML update count")
    offset = 1382
    arrays = {}
    for label, dtype, count in (
        ("Background:", "<f4", n + 1),
        ("Calibration:", "<f4", n + 1),
        ("DetectorFlags:", "u1", n),
    ):
        _label(raw, offset, label)
        offset += 50
        size = np.dtype(dtype).itemsize * count
        if offset + size > len(raw):
            raise ValueError("Truncated ArcCHECK ACML correction vectors.")
        arrays[label[:-1]] = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
        _finite(arrays[label[:-1]], f"ACML {label}")
        offset += size
    for label in _LITE_LABELS:
        _label(raw, offset, label)
        offset += 50
    dtype = np.dtype(_LITE_AUX.descr + [("Doses", "<f4", (n,))])
    return strings, scalars, arrays, offset, dtype


def _companion(path):
    candidate = Path(path).with_suffix(".acm")
    if candidate.is_file():
        return candidate
    matches = [
        p
        for p in Path(path).parent.iterdir()
        if p.name.casefold() == candidate.name.casefold() and p.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "ArcCHECK ACML requires a matching same-stem ACM file for detector geometry."
        )
    return matches[0]


def read_acml(raw, path):
    strings, scalars, vectors, offset, dtype = _lite_layout(raw)
    if offset + scalars["# Updates"] * dtype.itemsize != len(raw):
        raise ValueError("ArcCHECK ACML frame count or detector count differs from file size.")
    source = _companion(path)
    source_raw = source.read_bytes()
    serial, detector_count, geometry = _geometry_source(source_raw)
    n = scalars["# Detectors"]
    if not serial or serial != strings["ArcCHECK Serial"] or n != detector_count:
        raise ValueError(
            "ArcCHECK ACML and its ACM geometry source have different serials or detector counts."
        )
    data = np.frombuffer(raw, dtype=dtype, count=scalars["# Updates"], offset=offset)
    aux = np.empty(len(data), dtype=_LITE_AUX)
    for name in _LITE_AUX.names:
        aux[name] = data[name]
    doses = data["Doses"].copy()
    _finite(doses, "ACML doses")
    for name in ("TIMETIC1", "TIMETIC2", "Gantry Angle"):
        _finite(aux[name], f"ACML {name}")
    m = _measurement(
        doses,
        aux["TIMETIC1"].astype(float) / 1000,
        geometry,
        "cGy",
        "ACML processed dose",
    )
    native = {
        "Format": "ACML",
        "Lite Header": raw[:offset],
        "Header": strings,
        "Scalars": scalars,
        **vectors,
        "Auxiliary Data": aux,
        "Geometry": geometry,
        "Geometry Source": str(source.resolve()),
        "Geometry Source Serial": serial,
    }
    return {"File Type": "Movie Lite", "ArcCHECK": native}, m


def _check_measurement(scan, native):
    scan.validate()
    if (
        scan.type is not MeasurementType.DosePlane
        or scan.acquisition_type is not AcquisitionType.TimeSeries
        or scan.radiation_type not in (None, RadiationType.PHOTON)
    ):
        raise ValueError("ArcCHECK movies require a photon dose-plane time series.")
    y, x, xyz = native["Geometry"]
    if (
        scan.spatial_shape != xyz.shape[:2]
        or not np.array_equal(scan.axes[0], y)
        or not np.array_equal(scan.axes[1], x)
        or scan.coordinates is None
        or not np.array_equal(scan.coordinates, xyz)
        or (scan.startpoint is not None and not np.array_equal(scan.startpoint, xyz[0, 0]))
        or (scan.endpoint is not None and not np.array_equal(scan.endpoint, xyz[-1, -1]))
    ):
        raise ValueError("ArcCHECK movie export requires unchanged native detector geometry.")
    if any(value is not None for value in (scan.SSD, scan.field_size, scan.wedge_angle)):
        raise ValueError("ArcCHECK movie SSD, field-size and wedge edits are not supported.")
    _finite(scan.doses, "movie signal")


def movie_output(scan, metadata, destination):
    native = metadata.get("ArcCHECK", {})
    if native.get("Format") not in ("ACM", "ACML"):
        raise ValueError("ArcCHECK movie writing requires retained native movie metadata.")
    _check_measurement(scan, native)
    if destination.suffix.lower() != "." + native["Format"].lower():
        raise ValueError("ArcCHECK movies must be saved in their original ACM or ACML format.")
    if native["Format"] == "ACM":
        if scan.metadata.get("DoseUnit") != "counts" or scan.energy != native["Beam Energy"]:
            raise ValueError("ArcCHECK ACM output requires raw counts and unchanged beam energy.")
        if scan.metadata.get("FluenceMode", "") != native["FluenceMode"]:
            raise ValueError("ArcCHECK ACM fluence-mode edits are not supported.")
        lines = list(native["Header Lines"])
        sections, columns, ids, geometry, _ = _acm_layout(lines)
        if len(scan.times) != _integer(
            sections.get("Measurement Corrections", {}).get("Number of Updates"),
            "Number of Updates",
        ):
            raise ValueError("ArcCHECK ACM export must retain all native movie frames.")
        if any(not np.array_equal(a, b) for a, b in zip(geometry, native["Geometry"])):
            raise ValueError("ArcCHECK ACM retained header no longer matches detector geometry.")
        aux = np.asarray(native["Auxiliary Data"])
        if aux.shape != (len(scan.times), len(columns)):
            raise ValueError("ArcCHECK ACM auxiliary channels must match every movie frame.")
        aux = _finite(aux, "ACM auxiliary data").astype(float, copy=True)
        aux[:, columns.index("TIMETIC1")] = np.asarray(scan.times, dtype=float) * 1000
        for i, line in enumerate(lines):
            cells = _cells(line)
            if cells[0] in ("Background", "Calibration"):
                values = _finite(native["Corrections"][cells[0]], cells[0])
                if values.shape != (len(ids) + 1,):
                    raise ValueError(
                        "ArcCHECK ACM corrections must include the reference channel and every detector."
                    )
                lines[i] = "\t".join([cells[0], *([""] * 9), *(f"{v:.17g}" for v in values)])
        signal = np.asarray(scan.doses).reshape(len(scan.times), len(ids))
        lines.extend(
            "Data:\t" + "\t".join(f"{v:.17g}" for v in (*channels, *values))
            for channels, values in zip(aux, signal)
        )
        payload = ("\r\n".join(lines) + "\r\n").encode(native["Encoding"])
        return {destination: payload + f"{zlib.crc32(payload):08X}\r\n".encode("ascii")}
    if scan.metadata.get("DoseUnit") != "cGy" or scan.energy is not None:
        raise ValueError("ArcCHECK ACML stores cGy frames without an editable beam-energy field.")
    prefix = bytearray(native["Lite Header"])
    _, scalars, _, offset, dtype = _lite_layout(prefix)
    if offset != len(prefix):
        raise ValueError("ArcCHECK ACML retained header has an inconsistent size.")
    count = scalars["# Updates"]
    if count != len(scan.times) or np.asarray(native["Auxiliary Data"]).shape != (count,):
        raise ValueError(
            "ArcCHECK ACML export must retain all movie frames and auxiliary channels."
        )
    records = np.empty(count, dtype=dtype)
    for name in _LITE_AUX.names:
        records[name] = native["Auxiliary Data"][name]
    with np.errstate(over="ignore", invalid="ignore"):
        records["Doses"] = np.asarray(scan.doses).reshape(count, scalars["# Detectors"])
        records["TIMETIC1"] = np.asarray(scan.times, dtype=float) * 1000
    for name in ("Doses", "TIMETIC1", "TIMETIC2", "Gantry Angle"):
        _finite(records[name], f"ACML {name} within float32 precision")
    # The same-stem ACM is a geometry dependency. Copy it alongside new output,
    # preserving its raw signal independently of edited processed lite doses.
    source = Path(native["Geometry Source"])
    source_raw = source.read_bytes()
    serial, _, source_geometry = _geometry_source(source_raw)
    if serial != native["Geometry Source Serial"] or any(
        not np.array_equal(a, b) for a, b in zip(source_geometry, native["Geometry"])
    ):
        raise ValueError("ArcCHECK ACM geometry dependency changed since loading the lite file.")
    payload = bytes(prefix) + records.tobytes()
    outputs = {destination: payload}
    companion = destination.with_suffix(".acm")
    if companion.resolve() != source.resolve():
        if companion.exists():
            existing_serial, _, existing_geometry = _geometry_source(companion.read_bytes())
            if existing_serial != serial or any(
                not np.array_equal(a, b) for a, b in zip(existing_geometry, native["Geometry"])
            ):
                raise ValueError(
                    "ArcCHECK ACML output has an incompatible ACM companion; choose a new output name."
                )
        else:
            outputs[companion] = source_raw
    return outputs
