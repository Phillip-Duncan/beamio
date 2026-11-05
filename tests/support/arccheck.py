"""Synthetic measured/merged ArcCHECK exports; no acquired data."""

import zlib

import numpy as np


def export(
    *,
    rows=3,
    cols=5,
    kind="Measured File",
    model="1220 ArcCHECK",
    revision="E",
    energy="6FFF",
    encoding="utf-8",
    checksum=True,
    y_shift=0,
    orientation=0,
    dose=None,
    flags=None,
    x=None,
    y=None,
):
    x = (np.arange(cols) - (cols - 1) / 2) * 5 if x is None else np.asarray(x)
    y = ((rows - 1) / 2 - np.arange(rows)) * 5 + y_shift if y is None else np.asarray(y)
    dose = (
        np.arange(rows * cols, dtype=float).reshape(rows, cols) + 0.25
        if dose is None
        else np.asarray(dose)
    )
    merged = kind == "Merged File"
    header = [
        "** WARNING! Synthetic SNC export.",
        "",
        "8.5.1.9",
        f"File Revision:\t{revision}",
        kind,
        "Sun Nuclear Corporation",
        model,
        "",
        "FileName:\tsynthetic.txt",
        "Hardware Revision:\tE1",
        "Diode Type:\t3",
        "Measured Cavity Dose:\t",
    ]
    if merged:
        header += [
            "Combined from files:",
            "C:\\synthetic\\first.txt\t10\t20\t0",
            "C:\\synthetic\\second.txt\t10\t2\t0",
            "",
        ]
    header += [
        "Date:\t06/04/26\tTime:\t12:34:56",
        "Serial No:\t123456",
        "Cal File:\tsynthetic.cal",
        f"Dose per Count:\t{0 if merged else .002}",
        "Time:\t1000",
        f"Orientation:\t{orientation}",
        f"Rows:\t{rows}",
        f"Cols:\t{cols}",
        f"CAX X:\t{cols//2+1}",
        f"CAX Y:\t{rows//2+1}",
        "Shift X:\t0.0\tmm",
        "Shift Y:\t0.0\tmm",
        "Shift Z:\t0.0\tmm",
        "Rotation X:\t0.0\tdeg",
        "Rotation Y:\t0.0\tdeg",
        "Rotation Z:\t0.0\tdeg",
        "Unknown Field:\tKeep this",
        "",
        "Measurement Corrections",
        "Manufacturer:\tSynthetic linac",
        f"Energy:\t{energy}",
        "Plug Present:\t1",
        "Applied Angular:\t1",
        "Applied Heterogeneity:\t1\t\tApplied Out Of Field:\t0",
        "",
    ]
    tables = {
        "Background": np.full_like(dose, 0.01),
        "Calibration Factors": np.ones_like(dose),
        "Offset": np.zeros_like(dose),
        "Raw Counts": dose * 1000,
        "Corrected Counts": dose * 500,
        "Dose Counts": dose / 2,
        "Data Flags": np.full_like(dose, 69),
        "Interpolated": dose * 3,
        "Dose Interpolated": dose,
        "Corrected Counts (No Angular Correction)": dose * 400,
    }
    if merged:
        tables["Dose Counts"] = dose.copy()
        if flags is None:
            tables["Dose Counts"][1::2, 1::2] = 0
        else:
            tables["Dose Counts"][np.asarray(flags) == 0] = 0
        tables["Dose Interpolated"] = np.zeros_like(dose)
        tables["Interpolated"] = np.zeros_like(dose)
        tables["Data Flags"] = (tables["Dose Counts"] != 0).astype(float)
    if flags is not None:
        tables["Data Flags"] = np.asarray(flags)
        tables["Calibration Factors"][np.asarray(flags) == 0] = 0
    for name, values in tables.items():
        header.extend((name, "Ycm\tROW"))
        header.extend("\t".join(map(str, (v / 10, rows - i, *values[i]))) for i, v in enumerate(y))
        header.extend(
            (
                "\tCOL\t" + "\t".join(str(i + 1) for i in range(cols)),
                "\tXcm\t" + "\t".join(str(v / 10) for v in x),
                "",
            )
        )
    payload = ("\r\n".join(header) + "\r\n").encode(encoding)
    return payload + (f"{zlib.crc32(payload):08X}\r\n".encode() if checksum else b"")


def change(data, old, new):
    """Change synthetic text while maintaining a valid checksum."""
    payload = data.rsplit(b"\r\n", 2)[0] + b"\r\n"
    payload = payload.replace(old.encode(), new.encode())
    return payload + f"{zlib.crc32(payload):08X}\r\n".encode()
