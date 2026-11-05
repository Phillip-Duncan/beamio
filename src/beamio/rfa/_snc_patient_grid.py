"""SNC Patient text grids shared by ArcCHECK and SRS MapCHECK.

Coordinates use mm; signal values retain their exported scale.
"""

import re

import numpy as np


def _number(value, label):
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"Invalid {label}: {value!r}") from None
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _integer(value, label):
    result = _number(value, label)
    if result % 1 or result < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return int(result)


def _cells(line):
    return [cell.strip() for cell in line.rstrip("\t ").split("\t")]


def _axis(values, label):
    differences = np.diff(values)
    if not np.all(np.isfinite(values)) or not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"{label} coordinates must be finite and strictly monotonic.")


def read_grids(
    lines: list[str],
    nrows: int,
    ncols: int,
    format_name: str,
    required_table: str = "Dose Interpolated",
):
    tables: dict[str, np.ndarray] = {}
    reference = None
    index = 0
    checksum = None
    while index < len(lines):
        name = lines[index].strip()
        index += 1
        if not name:
            continue
        if re.fullmatch(r"[0-9A-Fa-f]{1,8}", name) and not any(
            line.strip() for line in lines[index:]
        ):
            checksum = name
            break
        if name in tables:
            raise ValueError(f"Duplicate {format_name} table: {name}")
        block = lines[index : index + nrows + 3]
        if len(block) != nrows + 3 or _cells(block[0]) != ["Ycm", "ROW"]:
            raise ValueError(f"{format_name} {name}: incomplete table or missing Ycm/ROW heading.")
        data = []
        ys, row_ids = [], []
        for row in block[1 : nrows + 1]:
            cells = _cells(row)
            if len(cells) != ncols + 2:
                raise ValueError(
                    f"{format_name} {name}: row width differs from the declared column count."
                )
            ys.append(_number(cells[0], f"{format_name} {name} Ycm") * 10)
            row_ids.append(_integer(cells[1], f"{format_name} {name} ROW"))
            data.append([_number(value, f"{format_name} {name}") for value in cells[2:]])
        columns, xs = _cells(block[-2]), _cells(block[-1])
        if (
            len(columns) != ncols + 2
            or columns[:2] != ["", "COL"]
            or len(xs) != ncols + 2
            or xs[:2] != ["", "Xcm"]
        ):
            raise ValueError(f"{format_name} {name}: missing or incomplete COL/Xcm coordinates.")
        column_ids = [_integer(value, f"{format_name} {name} COL") for value in columns[2:]]
        if sorted(row_ids) != list(range(1, nrows + 1)) or sorted(column_ids) != list(
            range(1, ncols + 1)
        ):
            raise ValueError(
                f"{format_name} {name}: duplicate or inconsistent row/column identifiers."
            )
        x = np.array([_number(value, f"{format_name} {name} Xcm") * 10 for value in xs[2:]])
        y = np.array(ys)
        _axis(x, f"{format_name} X")
        _axis(y, f"{format_name} Y")
        geometry = (x, y, np.array(row_ids), np.array(column_ids))
        if reference is not None and any(
            not np.array_equal(a, b) for a, b in zip(reference, geometry)
        ):
            raise ValueError(f"{format_name} {name}: grid geometry differs from the other tables.")
        reference = geometry
        tables[name] = np.array(data)
        index += nrows + 3
    if required_table not in tables:
        raise ValueError(f"{format_name} files require a {required_table} table.")
    return reference, tables, checksum


def write_grids(tables, geometry, format_name, required_table="Dose Interpolated"):
    """Render every source table, checking dimensions, IDs and finite values."""
    x, y, rows, columns = map(np.asarray, geometry)
    if rows.shape != y.shape or columns.shape != x.shape or x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"{format_name} grid identifiers must match the dose grid.")
    lines = []
    for name, values in tables.items():
        values = np.asarray(values)
        if (
            values.dtype.kind not in "biuf"
            or values.shape != (len(y), len(x))
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{format_name} {name} must be a finite table matching the dose grid.")
        if not isinstance(name, str) or not name.strip() or any(c in name for c in "\r\n\t"):
            raise ValueError(f"{format_name} table names must be single-line labels.")
        lines.extend((name, "Ycm\tROW"))
        lines.extend(
            "\t".join([f"{position/10:.17g}", f"{row:.17g}", *(f"{v:.17g}" for v in sample)])
            for position, row, sample in zip(y, rows, values)
        )
        lines.extend(
            (
                "\tCOL\t" + "\t".join(f"{v:.17g}" for v in columns),
                "\tXcm\t" + "\t".join(f"{v/10:.17g}" for v in x),
                "",
            )
        )
    read_grids(lines, len(y), len(x), format_name, required_table)
    return lines
