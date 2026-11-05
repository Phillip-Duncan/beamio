"""Small, synthetic OmniPro records; no clinical data."""

import struct
import zlib

import numpy as np


def opg_export(**changes):
    header = {
        "File Version": "4",
        "Separator": "[TAB]",
        "Image Name": "Synthetic plane",
        "Radiation Type": "Photons",
        "Energy": "6 MV",
        "SSD": "90 cm",
        "SID": "100 cm",
        "Field Size Cr": "10 cm",
        "Field Size In": "8 cm",
        "Data Type": "Abs. Dose",
        "Data Factor": "2",
        "Data Unit": "mGy",
        "Length Unit": "cm",
        "Plane": "XY",
        "No. of Columns": "3",
        "No. of Rows": "2",
        "Number of Bodies": "1",
        "Operators Note": "Synthetic export",
        "Custom Header": "retain this",
    }
    header.update(changes)
    return "\n".join(
        [
            "<opimrtascii>",
            "<asciiheader>",
            *(f"{k}: {v}" for k, v in header.items()),
            "</asciiheader>",
            "<asciibody>",
            "Plane Position: 1.5 cm",
            "Custom Body: keep this too",
            "X[cm]\t-1\t0\t1",
            "Y[cm]",
            "-2\t10\t20\t30",
            "2\t40\t50\t60",
            "</asciibody>",
            "</opimrtascii>",
            "",
        ]
    )


def opd_export(images=None, version=4):
    """Images are (name, signal_type, elapsed_seconds, factor_mGy, pixels)."""
    if images is None:
        images = [("Integral 0001", 1, 0, 2.0, np.arange(1, 7).reshape(2, 3))]
    result = bytearray(struct.pack("<8sHIHH", b"opdfile\0", 1, version, 36, len(images)))
    for name, signal_type, seconds, factor, pixels in images:
        info = bytearray(105)
        info[:25] = name.encode("cp1252").ljust(25, b"\0")
        struct.pack_into("<dId", info, 49, 45000 + seconds / 86400, signal_type, factor)
        setup = bytearray(86)
        struct.pack_into("<4hH", setup, 31, -500, 500, -400, 400, 9000)
        struct.pack_into("<d", setup, 43, 6.0)
        grid = bytearray(151)
        ny, nx = np.shape(pixels)
        struct.pack_into("<HHHI", grid, 81, nx, ny, 2, 4)
        struct.pack_into("<dd", grid, 95, -10.0, 10.0)
        struct.pack_into("<dd", grid, 113, -20.0, 20.0)
        for kind, payload in ((2, info), (3, setup), (99, b"opaque record"), (6, grid)):
            result.extend(struct.pack("<HI", kind, len(payload)))
            result.extend(payload)
            if kind == 6:
                result.extend(np.asarray(pixels, dtype="<u4").tobytes())
    return with_crc(result)


def with_crc(data):
    return bytes(data) + struct.pack("<I", zlib.crc32(data))
