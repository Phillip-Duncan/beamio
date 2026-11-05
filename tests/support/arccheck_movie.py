"""Small native detector movies, constructed without measurement data."""

import struct
import zlib

import numpy as np


def acm(*, serial="12345", revision="D", encoding="utf-8", nc=4, nr=2):
    n = nc * nr
    head = [
        "1220 ArcCHECK Measurement file: Header, Definition, Data",
        "Header:",
        f"File Revision:\t{revision}",
        "Software Version:\t8.5.1.9",
        "Description:\tSynthetic acquisition",
        "\tCollector Config",
        "Collector Model:\tArcCHECK 1220",
        f"Collector Serial:\t{serial}\tRevision\tE1",
        "Collection Interval:\t50\tms",
        "\tLINAC Configuration",
        "Energy:\t6\tMV",
        "\tDose Calibration",
        "Dose Per Count:\t0.001",
        "\tMeasurement Corrections",
        "Energy:\t6MV",
        "Number of Updates:\t3",
        "Total Time:\t100\tms",
    ]
    for key, value in (("NC", nc), ("SC", 1), ("SR", 1), ("L", nc), ("OO", 2)):
        head.append("\t".join([*([""] * 11), f"{key} =", str(value)]))
    angle = (np.arange(n) % nc + 0.5) * 2 * np.pi / nc
    x = -np.sin(angle) * nc / (2 * np.pi)
    y = np.repeat(np.arange(nr) - (nr - 1) / 2, nc) + np.tile(np.arange(nc) / nc, nr)
    z = np.cos(angle) * nc / (2 * np.pi)
    for key, values in (("X(n)", x), ("Y(n) cm", y), ("z(n)", z)):
        head.append("\t".join([*([""] * 9), key, "", *(f"{v:.10f}" for v in values)]))
    head.append(
        "\t".join(
            [*([""] * 9), "Detector Number", "Reference Diode", *(str(i + 1) for i in range(n))]
        )
    )
    head.append(
        "TYPE\tUPDATE#\tTIMETIC1\tTIMETIC2\tPULSES\tSTATUS1\tSTATUS2\tVirtualInclinometer\tCorrectedAngle\tFieldSize"
    )
    head.append("\t".join(["Background", *([""] * 9), *(["0.01"] * (n + 1))]))
    head.append("\t".join(["Calibration", *([""] * 9), *(["1.05"] * (n + 1))]))
    for t in range(3):
        values = np.arange(n) + t * 100 - 5
        aux = [73 + t, 3649.966 + t * 50, 3650 + t * 50, t, 4, 4, 180 - t, 179 - t, 10, 1000 + t]
        head.append("Data:\t" + "\t".join(str(v) for v in (*aux, *values)))
    raw = ("\r\n".join(head) + "\r\n").encode(encoding)
    return raw + f"{zlib.crc32(raw):08X}\r\n".encode()


def acml(*, serial="12345", revision="C", n=8):
    def field(value):
        return value.encode("cp1252").ljust(50, b"\0")

    raw = bytearray(b"ArcCHECK Measurement file: Header, Definition, Data".ljust(100, b"\0"))
    for key, value in (
        ("File Revision", revision),
        ("Date", "01/10/2026"),
        ("Time", "12:34:56"),
        ("Software Version", "8.5.1.9"),
        ("Firmware Version", "1000005"),
        ("Hardware Revision", "E1"),
        ("ArcCHECK Serial", serial),
        ("Temperature", "20, 21, 22"),
        ("Diode Type", "3"),
    ):
        raw.extend(field(key + ":"))
        raw.extend(field(value))
    for key, dtype, value in (
        ("Dose Per Count", "d", 0.001),
        ("Inclinomer Tilt", "f", 0.1),
        ("Inclinomer Rotate", "f", 0.2),
        ("Cavity Plug Present", "i", 1),
        ("Y-Shift (mm)", "f", 0),
        ("# Updates", "I", 2),
        ("# Detectors", "I", n),
    ):
        raw.extend(field(key + ":"))
        raw.extend(struct.pack("<" + dtype, value))
    for key, values in (
        ("Background", np.full(n + 1, 0.01, dtype="<f4")),
        ("Calibration", np.full(n + 1, 1.05, dtype="<f4")),
        ("DetectorFlags", np.full(n, 49, dtype="u1")),
    ):
        raw.extend(field(key + ":"))
        raw.extend(values.tobytes())
    for label in (
        "TIMETIC1",
        "TIMETIC2",
        "STATUS1",
        "STATUS2",
        "Gantry Angle (ArcCHECK Coordinate System)",
        "Ref Channel",
    ):
        raw.extend(field(label))
    for i in range(2):
        raw.extend(struct.pack("<ffIIfI", 3699.966 + i * 50, 3700 + i * 50, 4, 8, 179 - i, 0))
        raw.extend((np.arange(n, dtype="<f4") + i + 0.125).tobytes())
    return bytes(raw)
