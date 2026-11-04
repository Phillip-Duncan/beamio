"""Synthetic PRM acquisitions, with the physical detector numbering and row widths."""


def prm_export(*, beam="X-Ray FFF", energy="6 MV", wedge="Static\tat\t60", frames=2):
    detectors = [
        f"{axis}{number}"
        for axis in ("X", "Y", "PD", "ND")
        for number in range(1, 66)
        if axis == "Y" or number not in (32, 34)
    ]
    columns = [
        "UPDATE#",
        "TIMETIC",
        "PULSES",
        "STATUS",
        *detectors,
        "Ref",
        "Z1",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "Pressure",
        "UltraSonic",
        "Temperature",
    ]
    lines = [
        "Version:\t 25",
        "Filename:\tsynthetic.prm",
        "Date:\t11/4/2026\t Time:\t12:34:56",
        "Description:\tSynthetic acquisition",
        "\tProfiler Setup",
        "Buildup:\t2.9\tcm\tWaterEquiv",
        "Orientation:\tY Axis",
        "SSD:\t97.1\tcm",
        "\tCollector Config",
        "Collector Model:\tIC PROFILER",
        "Collector Serial:\t123456\tRevision:\tL",
        "\tMachine Data",
        f"Beam Type:\t{beam}\tEnergy:\t{energy}",
        "Collimator:\tLeft: 10 Right: 15 Top: 12 Bottom: 8 cm",
        f"Wedge:\t{wedge}",
        "Rate:\t450\tmu/Min\tDose:\t200",
        "Gantry Angle:\t90 deg\tCollimator Angle:\t0 deg",
        "\tHardware Data",
        "Temperature:\t21.5\t12345",
        "\tDose Calibration",
        "Dose Per Count:\t0.01",
        "Dose:\t100",
        "Absolute Calibration:\ttrue",
        "Energy:\t6 MV",
        "\tReference Detector Dose Calibration",
        "Dose Per Count:\t123",
        "Dose:\t999",
        "Absolute Calibration:\ttrue",
        "Energy:\tUndefined",
        "Multi Frame:\ttrue",
        f"Total Pulses:\t{frames * 10}",
        f"Total Time:\t{frames * 0.001:.9g}",
        "Detectors:\t65\t63\t63\t63\t1\t5",
        "Detector Spacing:\t0.5",
        "Concatenation:\tfalse",
        "Imported Data:\tfalse",
        "Definitions:",
        "\tData:",
        "\t\tIntegrated relative dose = {RawCount - TimeTic*LeakRate}*cf",
        "\t".join(["TYPE", *columns]),
        "\t".join(["BIAS1", "", "500000", "", "", *["0.01"] * len(detectors), *["0"] * 7]),
        "\t".join(["Calibration", "", "", "", "", *["2"] * len(detectors)]),
        "\t".join(["IgnoreDet", "", "", "", "", *["0"] * len(detectors), *["0"] * 7]),
    ]
    for frame in range(frames):
        counts = [str(100 * (frame + 1) + i) for i in range(len(detectors))]
        lines.append(
            "\t".join(
                [
                    "Data:",
                    str(frame),
                    str(1000 * (frame + 1)),
                    str(10 * (frame + 1)),
                    "67",
                    *counts,
                    *["0"] * 10,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def change_table_cell(text, row, column, value):
    lines = text.splitlines()
    columns = next(line.split("\t") for line in lines if line.startswith("TYPE\t"))
    for i, line in enumerate(lines):
        cells = line.split("\t")
        if cells[0] == row:
            cells[columns.index(column)] = str(value)
            lines[i] = "\t".join(cells)
    return "\n".join(lines) + "\n"


def moving_prm(
    *,
    x_centers=(-10, 0, 10),
    y_centers=(3, -2, 7),
    amplitudes=None,
    ticks=None,
    interval_pulses=None,
    half_width=50,
):
    """Analytic moving fields accumulated into a realistic raw PRM table.

    Each interval has a 20 mm raised-cosine penumbra, with analytic inflection
    points and half-maximum edges at centre +/- half_width.
    """
    import numpy as np

    frames = len(x_centers)
    assert len(y_centers) == frames
    amplitudes = np.full(frames, 100.0) if amplitudes is None else amplitudes
    ticks = np.arange(1, frames + 1) * 125000 if ticks is None else ticks
    interval_pulses = np.full(frames, 10) if interval_pulses is None else interval_pulses
    text = prm_export(beam="X-Ray", wedge="None\tat\t0", frames=0)
    columns = next(line.split("\t") for line in text.splitlines() if line.startswith("TYPE\t"))
    labels = columns[5:-10]
    total_signal = np.zeros(len(labels))
    pulses = 0
    lines = text.splitlines()
    for frame in range(frames):
        signal = []
        for label in labels:
            axis = label.rstrip("0123456789")
            position = (int(label[len(axis) :]) - 33) * 5
            if axis in ("PD", "ND"):
                position *= np.sqrt(2)
            center = y_centers[frame] if axis == "Y" else x_centers[frame]
            phase = np.clip((abs(position - center) - (half_width - 10)) / 20, 0, 1)
            signal.append(amplitudes[frame] * (1 + np.cos(np.pi * phase)) / 2)
        total_signal += signal
        raw_counts = total_signal / 2 + ticks[frame] * 0.01
        pulses += interval_pulses[frame]
        lines.append(
            "\t".join(
                str(value)
                for value in [
                    "Data:",
                    frame,
                    ticks[frame],
                    pulses,
                    67,
                    *raw_counts,
                    *[0] * 10,
                ]
            )
        )
    lines = [
        (
            f"Total Pulses:\t{pulses}"
            if line.startswith("Total Pulses:\t")
            else (
                f"Total Time:\t{ticks[-1] * 1e-6:.12g}"
                if line.startswith("Total Time:\t")
                else line
            )
        )
        for line in lines
    ]
    return "\n".join(lines) + "\n"
