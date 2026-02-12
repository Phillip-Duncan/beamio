import numpy as np
from enum import Enum
from typing import Tuple

class RadiationType(Enum):
    PHOTON = 'Photon'
    ELECTRON = 'Electron'
    PROTON = 'Proton'

class MeasurementType(Enum):
    DepthDose = 'DepthDose'
    CrosslineProfile = 'CrosslineProfile'
    InlineProfile = 'InlineProfile'
    DiagonalProfile = 'DiagonalProfile'


class Measurement:
    def __init__(self):
        """Initialize the Measurement class."""

        self.doses: np.ndarray = np.array([])
        self.positions: np.ndarray = np.array([])

        self.type: MeasurementType | None = None
        self.radiation_type: RadiationType | None = None
        self.energy: float | None = None
        self.SSD: float | None = None
        self.field_size: Tuple[float, float] | None = None
        self.wedge_angle: float | None = None

        self.startpoint: Tuple[float, float, float] | None = None
        self.endpoint: Tuple[float, float, float] | None = None

        # Other metadata fields depending on the file format can be added here
        # These will not be used in generic methods using measurement data, 
        # but can be useful for format-specific processing or saving
        self.metadata: dict = {}

    # ---------- Public methods/helpers ----------

    def reconstruct_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct X, Y, Z positions from the stored positions and startpoint/endpoint."""
        if self.startpoint is None or self.endpoint is None:
            raise ValueError("Startpoint and endpoint must be defined to reconstruct XYZ positions.")

        pos = np.asarray(self.positions, dtype=float)
        xs, ys, zs = self.startpoint
        xe, ye, ze = self.endpoint
        X = np.full_like(pos, xs, dtype=float)
        Y = np.full_like(pos, ys, dtype=float)
        Z = np.full_like(pos, zs, dtype=float)
        if self.type == MeasurementType.DepthDose:
            Z = pos
        elif self.type == MeasurementType.CrosslineProfile:
            X = pos
        elif self.type == MeasurementType.InlineProfile:
            Y = pos
        elif self.type == MeasurementType.DiagonalProfile:
            # For diagonal, we stored signed distance along diagonal in positions; need to convert back to X/Y
            angle_rad = np.arctan2(ye - ys, xe - xs)
            X = pos * np.sin(angle_rad)
            Y = pos * np.cos(angle_rad)

        return X, Y, Z
