import numpy as np
from enum import Enum
from typing import Tuple


def diagonal_direction(
    startpoint: Tuple[float, float, float], endpoint: Tuple[float, float, float]
) -> np.ndarray:
    """Return a unit profile axis toward positive X (positive Y for a vertical line)."""
    direction = np.asarray(endpoint[:2], dtype=float) - np.asarray(startpoint[:2], dtype=float)
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length == 0:
        raise ValueError("Diagonal profiles require distinct, finite start and end points.")
    direction /= length
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    return direction


class RadiationType(Enum):
    PHOTON = "Photon"
    ELECTRON = "Electron"
    PROTON = "Proton"


class MeasurementType(Enum):
    DepthDose = "DepthDose"
    CrosslineProfile = "CrosslineProfile"
    InlineProfile = "InlineProfile"
    DiagonalProfile = "DiagonalProfile"


class Measurement:
    def __init__(self):
        """Initialize the Measurement class."""

        self.doses: np.ndarray = np.array([])
        self.positions: np.ndarray = np.array([])
        # Optional measured XYZ samples in mm, including off-axis acquisition motion.
        self.coordinates: np.ndarray | None = None

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
            raise ValueError(
                "Startpoint and endpoint must be defined to reconstruct XYZ positions."
            )

        pos = np.asarray(self.positions, dtype=float)
        if self.coordinates is not None:
            xyz = np.asarray(self.coordinates, dtype=float)
            if pos.ndim != 1 or xyz.shape != (pos.size, 3) or not np.all(np.isfinite(xyz)):
                raise ValueError(
                    "Stored coordinates must be finite XYZ samples matching positions."
                )
            axes = {
                MeasurementType.CrosslineProfile: 0,
                MeasurementType.InlineProfile: 1,
                MeasurementType.DepthDose: 2,
            }
            if self.type == MeasurementType.DiagonalProfile:
                expected = xyz[:, :2] @ diagonal_direction(self.startpoint, self.endpoint)
            elif self.type in axes:
                expected = xyz[:, axes[self.type]]
            else:
                raise ValueError("A supported measurement type is required for stored coordinates.")
            if not np.allclose(pos, expected, rtol=0, atol=1e-9) or (
                pos.size
                and (
                    not np.array_equal(xyz[0], self.startpoint)
                    or not np.array_equal(xyz[-1], self.endpoint)
                )
            ):
                raise ValueError(
                    "Stored coordinates no longer match positions or endpoints; update coordinates or set them to None."
                )
            return xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy()
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
            direction = diagonal_direction(self.startpoint, self.endpoint)
            start = np.asarray(self.startpoint[:2], dtype=float)
            offset = start - np.dot(start, direction) * direction
            X = offset[0] + pos * direction[0]
            Y = offset[1] + pos * direction[1]

        return X, Y, Z
