"""Synthetic MCC exports; no acquired measurement files are included."""


def mcc_scan(metadata=None, rows=None, number=1):
    fields = {
        "SCAN_CURVETYPE": "CROSSPLANE_PROFILE",
        "MODALITY": "X",
        "ENERGY": "6.00",
        "SSD": "900.00",
        "FIELD_CROSSPLANE": "100.00",
        "FIELD_INPLANE": "200.00",
        "WEDGE_ANGLE": "0.00",
        "SCAN_DEPTH": "100.00",
        "SCAN_OFFAXIS_CROSSPLANE": "2.50",
        "SCAN_OFFAXIS_INPLANE": "-5.00",
        "CROSSPLANE_AXIS": "X",
        "INPLANE_AXIS": "Y",
        "SCAN_DIAGONAL": "NOT_DIAGONAL",
        "SCAN_ANGLE": "0.00",
        "DETECTOR": "DIAMOND_DETECTOR",
        "FILTER": "FF",
        "MEAS_UNIT": "A.U.",
        "COMMENT": "Synthetic=value",
    }
    fields.update(metadata or {})
    if rows is None:
        rows = ["-10.00  5.50E-03  2.0", "0.00  1.00E-02  2.1", "10.00  6.50E-03  2.2"]
    lines = [f"\tBEGIN_SCAN  {number}"]
    lines += [f"\t\t{key}={value}" for key, value in fields.items() if value is not None]
    lines += ["\t\tBEGIN_DATA", *("\t\t\t" + row for row in rows), "\t\tEND_DATA"]
    lines += [f"\tEND_SCAN  {number}"]
    return "\n".join(lines)


def mcc_export(*scans):
    return (
        "\n".join(
            [
                "BEGIN_SCAN_DATA",
                "\tFORMAT=CC-Export V1.9",
                *(scans or [mcc_scan()]),
                "END_SCAN_DATA",
            ]
        )
        + "\n"
    )
