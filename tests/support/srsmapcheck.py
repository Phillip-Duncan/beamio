"""Small asymmetric SRS MapCHECK maps and movies, generated without acquired data."""

import numpy as np

X = np.array([-3.5, -1.75, 0, 1.75, 3.5])
Y = np.array([1.75, 0, -1.75])
DOSE = np.arange(15, dtype=float).reshape(3, 5) + 0.25
DETECTOR_XY = np.array(
    [[x, y] for row, y in enumerate(Y[::-1]) for col, x in enumerate(X) if (row + col) % 2 == 0]
)


def header(kind):
    return f"""SRS MapCHECK Measurement File:\t{kind}

Header:
File Revision:\tA
File Name:\tSynthetic export
Date:\t01/02/26\tTime:\t12:34:56\t
Calibration File:\tsynthetic.cal
Software Version:\t8.5.1.9

\tDevice Config
Model:\t1179 SRS MapCHECK
Serial:\t12345
Collection Interval:\t50\tms

\tDevice Setup
Calibration Beam Energy:\t10\tFF
StereoPHAN Angle:\t270
Orientation:\t0
Shift From Iso X:\t2\tmm
Shift From Iso Y:\t-3\tmm
Shift From Iso Z:\t4\tmm

\tDose Calibration
Dose Per Count:\t0.004
Energy:\t10 FF
Temperature:\t22\tC
Comments:\tKeep this label:\tliterally

\tData Corrections Applied
Absolute Dose:\t1
Temperature:\t1
Angular:\t1
Field Size:\t1
Dose Rate:\t1
Couch Correction:\t0

\tMeasurement Info
Temperature:\t23
Update Count:\t2
Total Pulses:\t4
Total Time:\t150\tms
Beam Energy:\t6\tFFF
CaxY:\t2
CaxX:\t3
Row Counts in 2D:\t3
Col Counts in 2D:\t5

"""


def table(name, values, *, x=X, y=Y):
    rows = [name, "Ycm\tROW\t"]
    for row, (coordinate, values_row) in enumerate(zip(y, values)):
        rows.append("\t".join(map(str, [coordinate / 10, len(y) - row, *values_row])))
    rows.append("\tCOL\t" + "\t".join(str(i) for i in range(1, len(x) + 1)))
    rows.append("\tXcm\t" + "\t".join(str(v / 10) for v in x))
    return "\n".join(rows) + "\n\n"


def measured_export():
    return (
        header("Measured File")
        + table("Raw Counts", np.arange(15).reshape(3, 5) * 100)
        + table("Dose Counts", DOSE / 2)
        + table("Data Flags", np.full((3, 5), 69))
        + table("Interpolated", DOSE / DOSE.max() * 100)
        + table("Dose Interpolated", DOSE)
        + "A1B2C3\n"
    )


def movie_export():
    detector_ids = [f"D{i}" for i in range(1, len(DETECTOR_XY) + 1)]
    labels = [
        "TYPE",
        "UPDATE#",
        "TIMETIC",
        "PULSES",
        "STATUS",
        "T1",
        "Angle",
        "Reference",
        *detector_ids,
        "EIB",
    ]
    lines = [header("Raw Movie Data")]
    for axis, name in enumerate(("X (mm)", "Y (mm)")):
        lines.append("\t".join([name, *[""] * 7, *map(str, DETECTOR_XY[:, axis]), ""]))
    lines.append("\t".join(labels))
    for name, value in (
        ("Background", -0.02),
        ("Calibration(AP)", 2),
        ("Calibration(PA)", 3),
        ("AP Open", 1000),
        ("PA Open", 2000),
        ("DetectorFlags", 1),
    ):
        lines.append("\t".join([name, *[""] * 6, "1", *[str(value)] * len(detector_ids), "0", ""]))
    for frame in range(2):
        values = 100 * (frame + 1) + np.arange(len(detector_ids))
        lines.append(
            "\t".join(
                map(
                    str,
                    [
                        "Data:",
                        11 + frame,
                        100 + frame * 50,
                        frame * 4,
                        0,
                        22.5,
                        -1000,
                        5,
                        *values,
                        10,
                    ],
                )
            )
        )
    return "\n".join(lines) + "\nABC123\n"
