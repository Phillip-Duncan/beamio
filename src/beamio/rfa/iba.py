from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from beamio.common.measurement import Measurement, MeasurementType, RadiationType
from beamio.common.beamdata import BeamData

class IBAAccept6(BeamData):
    """
    IBA myQA Accept 6.X ASCII measurement dump (RFA300 BDS format) parser/writer.

    - parse(path): if ASCII, calls _parse_ascii_file
    - save(path): if .asc, calls _save_ascii_file

    Uses a mapping layer from IBA %KEY parameters to your Measurement fields.
    """

    # ---------- Mapping layer ----------
    #
    # Each entry maps IBA %KEY -> (measurement_attr, converter)
    # Converters take the raw string (without the leading %KEY) and return the typed value.
    #
    # Keep this small and focused: only what your generic model uses.
    # Everything else remains in Measurement.metadata for round-tripping.
    #
    IBA_TO_MEASUREMENT: Dict[str, Tuple[str, Callable[[str], object]]] = {}

    MEASUREMENT_TO_IBA: Dict[str, Tuple[str, Callable[[object], str]]] = {}

    FILE_META_DEFAULTS: Dict[str, str] = {
        "MSR": "0",
        "SYS": "BDS 0",
    }

    RADIATION_TYPE_MAPPING: Dict[str, RadiationType] = {
        "PHO": RadiationType.PHOTON,
        "ELE": RadiationType.ELECTRON,
        "PRO": RadiationType.PROTON,
    }

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self._init_mappings()

    def _init_mappings(self) -> None:
        # Build once per instance (could also be class-level constants).
        def parse_float(s: str) -> float:
            return float(s.strip().replace(",", "."))

        def parse_field_size(s: str) -> Tuple[float, float]:
            parts = s.split()
            return (parse_float(parts[0]), parse_float(parts[1]))

        def parse_triplet(s: str) -> Tuple[float | str, float | str, float | str]:
            # strip inline comment
            if "#" in s:
                s = s.split("#", 1)[0].strip()
            parts = s.split()
            return tuple(parse_float(p) if self.is_numeric(p) else p for p in parts)

        # Forward mapping: %KEY -> Measurement attribute
        self.IBA_TO_MEASUREMENT = {
            "SSD": ("SSD", parse_float),
            "FSZ": ("field_size", parse_field_size),
            "STS": ("startpoint", parse_triplet),
            "EDS": ("endpoint", parse_triplet),
            "WEG": ("wedge_angle", parse_float),
        }

        # Reverse mapping for saving: Measurement attribute -> %KEY
        def fmt_float(x: object) -> str:
            return f"{float(x):.6g}"

        def fmt_field_size(fs: object) -> str:
            a, b = fs  # type: ignore[misc]
            return f"{fmt_float(a)}\t{fmt_float(b)}"

        def fmt_triplet(tp: object) -> str:
            x, y, z = tp  # type: ignore[misc]
            return f"{fmt_float(x)}\t{fmt_float(y)}\t{fmt_float(z)}"

        self.MEASUREMENT_TO_IBA = {
            "SSD": ("SSD", fmt_float),
            "field_size": ("FSZ", fmt_field_size),
            "startpoint": ("STS", fmt_triplet),
            "endpoint": ("EDS", fmt_triplet),
            "wedge_angle": ("WEG", fmt_float),
        }

    # ---------- Public API ----------

    def parse(self) -> "IBAAccept6":   
        _, ext = os.path.splitext(self.file_path)
        if ext.lower() == '.asc':
            self._parse_ascii_file(self.file_path)
        else:
            raise ValueError(f'Unsupported file extension: {ext}')
        return self

    def save(self, filepath: str) -> None:
        """
        Save current self.measurements. If filepath ends with .asc, write IBA ASCII.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".asc":
            self._save_ascii_file(filepath)
            return
        raise ValueError(f"Unsupported output format for IBAAccept6: {filepath}")

    # ---------- ASCII parsing ----------

    def _parse_ascii_file(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()

        self.measurements.clear()
        self.file_meta = {}  # reset per parse

        current_block: List[str] = []

        for raw in lines:
            s = raw.strip()
            if not s:
                continue

            if s.startswith(":MSR"):
                parts = s.split()
                if len(parts) >= 2:
                    self.file_meta["MSR"] = parts[1]
                continue

            if s.startswith(":SYS"):
                self.file_meta["SYS"] = " ".join(s.split()[1:])
                continue

            if s.startswith(":EOF"):
                if current_block:
                    self._parse_measurement_block(current_block)
                break

            current_block.append(raw)

            if s.startswith(":EOM"):
                self._parse_measurement_block(current_block)
                current_block = []

        if current_block:
            self._parse_measurement_block(current_block)

    def _parse_measurement_block(self, block_lines: List[str]) -> None:
        meta: Dict[str, str] = {}

        # 1) Gather all %KEY lines
        for ln in block_lines:
            s = ln.strip()
            if not s.startswith("%"):
                continue
            parts = s.split(None, 1)
            key = parts[0][1:].strip()
            val = parts[1].strip() if len(parts) > 1 else ""
            meta[key] = val

        # 2) Parse data rows '=' => X Y Z Dose
        xs, ys, zs, ds = [], [], [], []
        for ln in block_lines:
            s = ln.strip()
            if not s.startswith("="):
                continue
            payload = s[1:].strip()
            if "#" in payload:
                payload = payload.split("#", 1)[0].strip()
            parts = payload.split()
            if len(parts) < 4:
                continue
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            d = float("nan") if parts[3].lower() == "nan" else float(parts[3])
            xs.append(x)
            ys.append(y)
            zs.append(z)
            ds.append(d)

        if not ds:
            return

        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        z_arr = np.asarray(zs, dtype=float)
        d_arr = np.asarray(ds, dtype=float)

        m = Measurement()
        m.metadata = meta

        # 3) Apply mapping: %KEY -> Measurement fields (generic)
        for iba_key, (attr, conv) in self.IBA_TO_MEASUREMENT.items():
            if iba_key in meta and meta[iba_key].strip():
                try:
                    setattr(m, attr, conv(meta[iba_key]))
                except Exception:
                    # keep raw in metadata; don't hard fail unless you want to
                    pass

        # 3.5) Round startpoint to nearest 0.2 mm if present, to mitigate precision/uncertainties
        if m.startpoint is not None:
            m.startpoint = tuple(round(coord * 5) / 5 for coord in m.startpoint)

        # 4) Special hook for BMT -> radiation_type, energy
        if "BMT" in meta:
            bmt = meta["BMT"]
            try:
                m.radiation_type = self._parse_bmt_radiation(bmt)
            except Exception:
                pass
            try:
                m.energy = self._parse_bmt_energy(bmt)
            except Exception:
                pass

        # 5) Determine measurement type + positions
        scn = (meta.get("SCN") or "").strip().upper()

        # Which coordinate varies the most?
        rx = float(np.nanmax(x_arr) - np.nanmin(x_arr))
        ry = float(np.nanmax(y_arr) - np.nanmin(y_arr))
        rz = float(np.nanmax(z_arr) - np.nanmin(z_arr))

        # Assign, DEPTH, DIAGONAL, otherwise crossline/inline
        if scn == "DPT" or (rz >= max(rx, ry) and rz > 0.5):
            m.type = MeasurementType.DepthDose
            m.positions = z_arr
        elif scn == "DIA":
            m.type = MeasurementType.DiagonalProfile
            # For diagonal, we can store the distance along the diagonal as positions
            m.positions = np.sqrt(x_arr**2 + y_arr**2) * np.sign(x_arr)  # signed distance along diagonal
        else:
            if rx >= ry:
                m.type = MeasurementType.CrosslineProfile
                m.positions = x_arr
            else:
                m.type = MeasurementType.InlineProfile
                m.positions = y_arr

        # Doses
        m.doses = d_arr

        # Optional: trim leading/trailing NaNs in dose (common for depth scans)
        m.positions, m.doses = self._trim_nan_pairs(m.positions, m.doses)

        self.measurements.append(m)

    def _parse_bmt_radiation(self, bmt: str) -> RadiationType:
        token = bmt.strip().split()[0].upper()
        return self.RADIATION_TYPE_MAPPING.get(token, None)

    def _parse_bmt_energy(self, bmt: str) -> float:
        parts = bmt.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid BMT value: {bmt!r}")
        return float(parts[1].replace(",", ".")) # handle comma as decimal separator if present (European formats)

    def _trim_nan_pairs(
        self,
        positions: np.ndarray,
        doses: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove all pairs where either positions or doses are NaN/inf.

        Args:
            positions (np.ndarray): Array of position values.
            doses (np.ndarray): Array of dose values.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered positions and doses with NaNs removed.
        """
        if positions.size == 0 or doses.size == 0:
            return positions, doses

        if positions.shape != doses.shape:
            raise ValueError("positions and doses must have identical shapes")

        # True only where BOTH arrays are finite
        valid = np.isfinite(positions) & np.isfinite(doses)

        # Fast exit if nothing survives
        if not np.any(valid):
            return np.array([], dtype=positions.dtype), np.array([], dtype=doses.dtype)

        return positions[valid], doses[valid]

    # ---------- ASCII saving ----------

    def _save_ascii_file(self, filepath: str) -> None:
        """
        Write an IBA-style ASCII file.

        Philosophy:
        - Preserve metadata where available (m.metadata)
        - Ensure required keys exist for round-tripping
        - Recreate the data rows as '= X Y Z Dose' using startpoint and varying axis
        """
        out: List[str] = []

        out.append(f":MSR \t{len(self.measurements)}\t # No. of measurement in file")
        out.append(f":SYS \t{self.file_meta.get('SYS', self.FILE_META_DEFAULTS['SYS'])} # Beam Data Scanner System")
        out.append("#")
        out.append("# RFA300 ASCII Measurement Dump ( BDS format )")
        out.append("#")

        for idx, m in enumerate(self.measurements, start=1):
            out.append(f"# Measurement number \t{idx}")
            out.append("#")

            # Start from stored metadata if present; otherwise build minimal meta.
            meta = dict(getattr(m, "metadata", {}) or {})

            # Ensure basic keys exist / updated from measurement fields.
            meta.setdefault("VNR", "1.0")
            meta.setdefault("MOD", "RAT")
            meta.setdefault("TYP", "SCN")
            # SCN depends on type
            if m.type == MeasurementType.DepthDose:
                meta["SCN"] = "DPT"
            else:
                meta["SCN"] = "PRO"
            meta.setdefault("FLD", "SEM")

            # BMT from radiation_type + energy if available
            if m.radiation_type is not None and m.energy is not None:
                meta["BMT"] = self._format_bmt(m.radiation_type, m.energy)

            # Apply reverse mapping for known fields
            for attr, (iba_key, fmt) in self.MEASUREMENT_TO_IBA.items():
                val = getattr(m, attr, None)
                if val is None:
                    continue
                meta[iba_key] = fmt(val)

            # FSZ/SSD etc. now in meta if present

            # PTS
            meta["PTS"] = str(int(m.doses.size))

            # STS / EDS for scan start/end
            # If startpoint missing, infer from first coordinate with other axes = 0
            sts = m.startpoint
            eds = m.endpoint

            meta["STS"] = self._fmt_triplet(sts) + " # Start Scan values in mm ( X , Y , Z )"
            meta["EDS"] = self._fmt_triplet(eds) + " # End Scan values in mm ( X , Y , Z )"

            # Write %KEY lines in a reasonable order; keep extras at end
            preferred_order = [
                "VNR", "MOD", "TYP", "SCN", "FLD", "DAT", "TIM", "FSZ", "BMT", "SSD",
                "BUP", "BRD", "FSH", "ASC", "WEG", "GPO", "CPO", "MEA", "PRD", "PTS", "STS", "EDS"
            ]
            written = set()

            for k in preferred_order:
                if k in meta:
                    out.append(f"%{k} \t{meta[k]}")
                    written.add(k)

            # Write remaining metadata keys (stable order)
            for k in sorted(meta.keys()):
                if k in written:
                    continue
                out.append(f"%{k} \t{meta[k]}")

            out.append("#")
            out.append("#\t  X      Y      Z     Dose")
            out.append("#")

            # Data lines: reconstruct full X,Y,Z for each point
            X, Y, Z = m.reconstruct_xyz()
            # round to nearest 0.1 mm for cleaner output and to mitigate precision issues
            X, Y, Z = np.round(X * 10) / 10, np.round(Y * 10) / 10, np.round(Z * 10) / 10
            D = np.asarray(m.doses, dtype=float)

            for x, y, z, d in zip(X, Y, Z, D):
                if np.isnan(d):
                    d_str = "NaN"
                else:
                    d_str = f"{d:.6g}"
                out.append(f"= \t{float(x):.6g}\t{float(y):.6g}\t{float(z):.6g}\t{d_str}")

            out.append(":EOM  # End of Measurement")

            if idx < len(self.measurements):
                out.append("#")
                out.append("# RFA300 ASCII Measurement Dump ( BDS format )")
                out.append("#")

        out.append(":EOF # End of File")

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out))

    def _format_bmt(self, rad: RadiationType, energy: float) -> str:
        rtm_reversed = {v: k for k, v in self.RADIATION_TYPE_MAPPING.items()}
        tok = rtm_reversed.get(rad)
        return f"{tok}\t    {float(energy):.6g}"

    def _fmt_triplet(self, p: Tuple[float, float, float]) -> str:
        return f"{p[0]:.6g}\t{p[1]:.6g}\t{p[2]:.6g}"
