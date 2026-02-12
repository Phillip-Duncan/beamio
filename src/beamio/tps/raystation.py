from __future__ import annotations

import os
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from beamio.common.beamdata import BeamData
from beamio.common.measurement import Measurement
from beamio.common.measurement import MeasurementType, RadiationType  # adjust import path if different


class Raystation(BeamData):
    """
    RayStation measured-curves CSV handler (semicolon-delimited).

    File header:
      - comment lines beginning with '#'
      - we map '#Key: value' into self.file_meta (file-level metadata)
      - unknown header comment lines are stored in self._file_header_extras (optional)

    Saving:
      - uses FILE_META_DEFAULTS (stable, deterministic defaults) rather than parsed header values
    """

    FILE_META_DEFAULTS: Dict[str, str] = {
        "Exported from": "RayStation-compatible writer",
        "Time of export": "",  # filled on save
        "Machine name": "Unknown",
        "Commission status": "Unknown",
        "Measured curves": "",  # marker line (no value)
        "Field collimation": "Jaws and MLC collimated",
        "Dose unit": "Gy",
    }

    CURVE_TYPE_MAPPING: Dict[str, MeasurementType] = {
        "Depth": MeasurementType.DepthDose,
        "Crossline": MeasurementType.CrosslineProfile,
        "Inline": MeasurementType.InlineProfile,
        "Diagonal": MeasurementType.DiagonalProfile,
    }

    RADIATION_TYPE_MAPPING: Dict[str, RadiationType] = {
        "Photon": RadiationType.PHOTON,
        "Electron": RadiationType.ELECTRON,
        "Proton": RadiationType.PROTON,
    }

    def __init__(self, filepath: str):
        super().__init__(filepath)
        # extras: header comments we couldn't parse into key/value
        self._file_header_extras: List[str] = []
        # seed defaults (keeps your shared self.file_meta as the canonical store)
        self.file_meta = dict(self.FILE_META_DEFAULTS)

    # ---------- Public API ----------

    def parse(self) -> "Raystation":
        _, ext = os.path.splitext(self.file_path)
        if ext.lower() == ".csv":
            self._parse_csv_file(self.file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        return self

    def save(self, output_path: str) -> None:
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".csv":
            self._save_csv_file(output_path)
            return
        raise ValueError(f"Unsupported output format for Raystation: {output_path}")

    # ---------- CSV parsing ----------

    def _parse_csv_file(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()

        self.measurements.clear()

        # Reset file_meta to defaults each parse
        self.file_meta = dict(self.FILE_META_DEFAULTS)
        self._file_header_extras = []

        # 1) Parse leading header comments
        i = 0
        while i < len(lines):
            raw = lines[i]
            s = raw.strip()
            if not s:
                i += 1
                continue

            if s.startswith("#"):
                self._ingest_header_comment_line(s)
                i += 1
                continue

            break  # first non-comment => measurement blocks start

        # 2) Parse measurement blocks terminated by exact "End"
        block: List[str] = []
        for j in range(i, len(lines)):
            raw = lines[j]
            s = raw.strip()

            if not s:
                continue

            if s == "End":
                if block:
                    m = self._parse_raystation_block(block)
                    if m is not None:
                        self.measurements.append(m)
                block = []
                continue

            block.append(raw)

        # trailing block without End
        if block:
            m = self._parse_raystation_block(block)
            if m is not None:
                self.measurements.append(m)

    def _ingest_header_comment_line(self, s: str) -> None:
        """
        Parse '#Key: value' into self.file_meta; otherwise store as extras.

        '#Measured curves' is treated as a marker key with empty value if present in defaults.
        """
        txt = s.lstrip("#").strip()
        if not txt:
            return

        if ":" not in txt:
            # marker
            if txt in self.FILE_META_DEFAULTS:
                self.file_meta[txt] = ""
            else:
                self._file_header_extras.append(s)
            return

        key, val = txt.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Keep only schema keys in file_meta; others go into extras
        if key in self.FILE_META_DEFAULTS:
            self.file_meta[key] = val
        else:
            self._file_header_extras.append(s)

    def _parse_raystation_block(self, block_lines: List[str]) -> Optional[Measurement]:
        def split_semicolon(line: str) -> List[str]:
            return [p.strip() for p in line.split(";")]

        def to_float(token: str) -> float:
            return float(token.strip().replace(",", "."))

        meta: Dict[str, str] = {}
        positions: List[float] = []
        doses: List[float] = []

        for ln in block_lines:
            s = ln.strip()
            if not s:
                continue

            parts = split_semicolon(s)
            if not parts:
                continue

            # Meta line: first token contains ':'
            if ":" in parts[0]:
                key = parts[0].rstrip(":").strip()
                vals = parts[1:]
                meta[key] = "; ".join([v for v in vals if v != ""])
                continue

            # Data line: position; value
            try:
                p = to_float(parts[0])
                d = to_float(parts[1]) if len(parts) > 1 and parts[1] != "" else float("nan")
            except Exception:
                continue

            positions.append(p)
            doses.append(d)

        if not doses:
            return None

        m = Measurement()
        m.metadata = dict(meta)

        # Map common RayStation fields into generic model (best-effort)

        if "energy[MV]" in meta and meta["energy[MV]"].strip():
            try:
                m.energy = to_float(meta["energy[MV]"].split(";")[0].strip())
            except Exception:
                pass

        if "SSD[mm]" in meta and meta["SSD[mm]"].strip():
            try:
                m.SSD = to_float(meta["SSD[mm]"].split(";")[0].strip())
            except Exception:
                pass

        if "StartPoint[mm]" in meta and meta["StartPoint[mm]"].strip():
            try:
                sp = [to_float(x) for x in split_semicolon(meta["StartPoint[mm]"])]
                if len(sp) >= 3:
                    m.startpoint = (float(sp[0]), float(sp[1]), float(sp[2]))
            except Exception:
                pass
        
        if "DiagonalAngle[deg]" in meta and meta["DiagonalAngle[deg]"].strip():
            try:
                m.metadata["DiagonalAngle[deg]"] = to_float(meta["DiagonalAngle[deg]"].split(";")[0].strip())

                xend = float(positions[-1]) * np.cos(np.radians(m.metadata["DiagonalAngle[deg]"]))
                yend = float(positions[-1]) * np.sin(np.radians(m.metadata["DiagonalAngle[deg]"]))
                m.endpoint = (xend, yend, m.startpoint[2] if m.startpoint is not None else 0.0)
            except Exception:
                pass

        if "Fieldsize[mm]" in meta and meta["Fieldsize[mm]"].strip():
            try:
                fs_tokens = [to_float(x) for x in split_semicolon(meta["Fieldsize[mm]"])]
                if len(fs_tokens) >= 4:
                    x1, y1, x2, y2 = fs_tokens[:4]
                    m.field_size = (float(x2 - x1), float(y2 - y1))
                    m.metadata.setdefault("FieldsizeCorners[mm]", f"{x1}; {y1}; {x2}; {y2}")
            except Exception:
                pass

        if "WedgeAngle[deg]" in meta and meta["WedgeAngle[deg]"].strip():
            try:
                m.wedge_angle = to_float(meta["WedgeAngle[deg]"].split(";")[0].strip())
            except Exception:
                pass

        m.radiation_type = self.RADIATION_TYPE_MAPPING.get(meta.get("RadiationType"))
        m.type = self.CURVE_TYPE_MAPPING.get(meta.get("CurveType"))
        m.positions = np.asarray(positions, dtype=float)
        m.doses = np.asarray(doses, dtype=float)
        m.positions, m.doses = self._trim_nan_pairs(m.positions, m.doses)

        return m

    def _trim_nan_pairs(self, positions: np.ndarray, doses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if doses.size == 0:
            return positions, doses
        valid = ~np.isnan(doses)
        if not np.any(valid):
            return positions, doses
        first = int(np.argmax(valid))
        last = int(len(valid) - 1 - np.argmax(valid[::-1]))
        return positions[first:last + 1], doses[first:last + 1]

    # ---------- CSV saving (defaults-only header) ----------

    def _save_csv_file(self, filepath: str) -> None:
        out: List[str] = []

        # Header: write DEFAULTS (deterministic), not parsed file_meta
        out.extend(self._emit_default_header_lines())

        # No measurements => header only
        if not self.measurements:
            with open(filepath, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(out) + "\n")
            return

        for m in self.measurements:
            out.extend(self._format_measurement_block(m))
            out.append("End")

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out) + "\n")

    def _emit_default_header_lines(self) -> List[str]:
        d = dict(self.FILE_META_DEFAULTS)
        now = datetime.datetime.now()
        d["Time of export"] = now.strftime("%d %b %Y, %H:%M:%S") + " (hr:min:sec)"

        order = [
            "Exported from",
            "Time of export",
            "Machine name",
            "Commission status",
            "Measured curves",
            "Field collimation",
            "Dose unit",
        ]

        lines: List[str] = []
        for k in order:
            if k == "Measured curves":
                lines.append("#Measured curves")
            else:
                lines.append(f"#{k}: {d.get(k, '')}")
        return lines

    # ---------- Measurement block formatting (schema-locked keys + defaults) ----------

    def _format_measurement_block(self, m: Measurement) -> List[str]:
        meta_in: Dict[str, str] = dict(getattr(m, "metadata", {}) or {})
        meta: Dict[str, str] = {}

        # Energy
        if getattr(m, "energy", None) is not None:
            meta["energy[MV]"] = self._fmt_float(float(m.energy))
        else:
            meta["energy[MV]"] = self._first_nonempty(meta_in, ["energy[MV]", "Energy[MV]"], default="6")

        # SSD
        if getattr(m, "SSD", None) is not None:
            meta["SSD[mm]"] = self._fmt_float(float(m.SSD))
        else:
            meta["SSD[mm]"] = self._first_nonempty(meta_in, ["SSD[mm]", "SSD"], default="900")

        # Field size corners
        meta["Fieldsize[mm]"] = self._fieldsize_corners_mm(m, meta_in)

        
        # Only put in wedge info if the wedge_angle is > 0.
        if m.wedge_angle > 0:
            # WedgeType and WedgeAngle if they exist, otherwise fall back to default value.
            if "WedgeType" in meta_in and meta_in["WedgeType"].strip():
                meta["WedgeType"] = meta_in["WedgeType"].strip()
            else:
                meta["WedgeType"] = "XXXX"  # RayStation seems to require this key if WedgeAngle is present

            meta["WedgeAngle[deg]"] = self._fmt_float(float(m.wedge_angle))

        # CurveType, Reverse the map and lookup
        ctm_reversed = {v: k for k, v in self.CURVE_TYPE_MAPPING.items()}
        meta["CurveType"] = ctm_reversed.get(m.type, None)

        # Compute wedge angle from start and end points.
        if m.type == MeasurementType.DiagonalProfile:
            if m.startpoint is not None and m.endpoint is not None:
                dx = m.endpoint[0] - m.startpoint[0]
                dy = m.endpoint[1] - m.startpoint[1]
                diagonal_angle_rad = np.arctan2(dy, dx)
                diagonal_angle_deg = np.degrees(diagonal_angle_rad)
                # Round to nearest 0.1 degree for cleaner output and to avoid floating point precision issues
                diagonal_angle_deg = round(diagonal_angle_deg * 10.0) / 10.0
                meta["DiagonalAngle[deg]"] = self._fmt_float(diagonal_angle_deg)

        # RadiationType
        rtm_reversed = {v: k for k, v in self.RADIATION_TYPE_MAPPING.items()}
        meta["RadiationType"] = rtm_reversed.get(getattr(m, "radiation_type", None), None)

        # Defaults
        meta["FluenceMode"] = self._first_nonempty(meta_in, ["FluenceMode"], default="Standard")
        meta["Quantity"] = self._first_nonempty(meta_in, ["Quantity"], default="RelativeDose")

        # StartPoint
        sp = getattr(m, "startpoint", None)
        meta["StartPoint[mm]"] = f"{self._fmt_float(sp[0])}; {self._fmt_float(sp[1])}; {self._fmt_float(sp[2])}"

        # Optional passthroughs
        passthrough_allow = [
            "GantryAngle[deg]", "CollimatorAngle[deg]", "CouchAngle[deg]",
            "SAD[mm]", "Wedge", "Accessory", "Applicator", "Setup",
        ]
        for k in passthrough_allow:
            if k in meta_in and k not in meta and str(meta_in[k]).strip():
                meta[k] = str(meta_in[k]).strip()

        preferred_order = [
            "energy[MV]",
            "SSD[mm]",
            "Fieldsize[mm]",
            "WedgeType",
            "WedgeAngle[deg]",
            "CurveType",
            "RadiationType",
            "DiagonalAngle[deg]",
            "FluenceMode",
            "Quantity",
            "StartPoint[mm]",
        ]

        lines: List[str] = []
        for k in preferred_order:
            if k not in meta:
                continue

            lines.append(self._fmt_meta_line(k, meta[k]))
        for k in sorted(meta.keys()):
            if k in preferred_order:
                continue
            lines.append(self._fmt_meta_line(k, meta[k]))

        # Data: position; dose
        pos = np.asarray(getattr(m, "positions", []), dtype=float)
        dose = np.asarray(getattr(m, "doses", []), dtype=float)

        for p, d in zip(pos, dose):
            d_str = "nan" if np.isnan(d) else self._fmt_float(float(d))
            lines.append(f"{self._fmt_float(float(p))}; {d_str}")

        return lines

    # ---------- Helpers ----------

    def _fmt_meta_line(self, key: str, value: str) -> str:
        parts = [p.strip() for p in str(value).split(";") if p.strip() != ""]
        return f"{key}:; " + "; ".join(parts)

    def _fmt_float(self, x: float) -> str:
        return f"{float(x):.15g}"

    def _first_nonempty(self, meta: Dict[str, str], keys: List[str], default: str) -> str:
        for k in keys:
            v = meta.get(k)
            if v is not None and str(v).strip() != "":
                return str(v).strip()
        return default

    def _fieldsize_corners_mm(self, m: Measurement, meta_in: Dict[str, str]) -> str:
        fs = (meta_in.get("Fieldsize[mm]") or "").strip()
        if fs:
            return fs

        fs = (meta_in.get("FieldsizeCorners[mm]") or "").strip()
        if fs:
            return fs

        fs_wh = getattr(m, "field_size", None)
        if fs_wh is not None:
            w, h = fs_wh

            # NOTE: if your generic model stores field size in cm, convert:
            # w = float(w) * 10.0
            # h = float(h) * 10.0

            x1, x2 = -float(w) / 2.0, float(w) / 2.0
            y1, y2 = -float(h) / 2.0, float(h) / 2.0
            return f"{self._fmt_float(x1)}; {self._fmt_float(y1)}; {self._fmt_float(x2)}; {self._fmt_float(y2)}"

        return None
