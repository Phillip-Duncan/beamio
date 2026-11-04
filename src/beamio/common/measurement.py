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
    DosePlane = "DosePlane"
    DoseVolume = "DoseVolume"

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial array dimensions, independent of acquisition time."""
        return {MeasurementType.DosePlane: 2, MeasurementType.DoseVolume: 3}.get(self, 1)


class AcquisitionType(Enum):
    Static = "Static"
    TimeSeries = "TimeSeries"


class Measurement:
    def __init__(self) -> None:
        """Initialize the Measurement class."""

        self.doses: np.ndarray = np.array([])
        self._axes: tuple[np.ndarray, ...] = (np.array([]),)
        self.acquisition_type: AcquisitionType = AcquisitionType.Static
        # Seconds, with one timestamp per leading dose frame; None for static data.
        self.times: np.ndarray | None = None
        # Fixed spatial XYZ samples in mm, with shape spatial_shape + (3,).
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

    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        """Spatial axes in dose-array order: (position,), (y, x), or (z, y, x), in mm."""
        return self._axes

    @axes.setter
    def axes(self, values: tuple[np.ndarray, ...]) -> None:
        axes = tuple(np.asarray(axis) for axis in values)
        if any(np.iscomplexobj(axis) for axis in axes):
            raise ValueError("Spatial axes must contain real coordinates.")
        self._axes = tuple(np.asarray(axis, dtype=float) for axis in axes)

    @property
    def positions(self) -> np.ndarray:
        """The single spatial axis of a 1D measurement; use axes for planes/volumes."""
        if (
            self.type in (MeasurementType.DosePlane, MeasurementType.DoseVolume)
            or len(self._axes) != 1
        ):
            raise ValueError("positions is only available for 1D measurements; use axes.")
        return self._axes[0]

    @positions.setter
    def positions(self, values: np.ndarray) -> None:
        self.axes = (values,)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Dose-array shape without its optional leading time dimension."""
        shape = np.shape(self.doses)
        return shape[1:] if self.acquisition_type == AcquisitionType.TimeSeries else shape

    def validate(self) -> None:
        """Check spatial/temporal shapes without filtering or changing acquired values.

        NaN signals are allowed. Coordinates describe fixed spatial geometry and
        never repeat across time. Timestamps retain acquisition order, including
        instrument clock anomalies. Grid axes must be strictly monotonic; 1D
        scans may retain repeated or unordered positions.
        """
        if not isinstance(self.type, MeasurementType):
            raise ValueError("A supported MeasurementType is required.")
        if not isinstance(self.acquisition_type, AcquisitionType):
            raise ValueError("A supported AcquisitionType is required.")
        doses = np.asarray(self.doses)
        if doses.dtype.kind not in "biuf":
            raise ValueError("doses must contain real numeric values.")
        ndim = self.type.spatial_ndim
        temporal = self.acquisition_type == AcquisitionType.TimeSeries
        if doses.ndim != ndim + int(temporal) or any(size == 0 for size in doses.shape):
            raise ValueError(
                f"{self.type.value} / {self.acquisition_type.value} requires nonempty doses "
                f"with {ndim + int(temporal)} dimensions (time first when present)."
            )
        if len(self.axes) != ndim:
            raise ValueError(f"{self.type.value} requires {ndim} spatial axes.")
        for axis, size in zip(self.axes, self.spatial_shape):
            if axis.ndim != 1 or axis.size != size or not np.all(np.isfinite(axis)):
                raise ValueError(
                    "Spatial axes must be finite 1D arrays matching the spatial shape."
                )
            if ndim > 1 and not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0)):
                raise ValueError("Plane and volume axes must be strictly monotonic.")
        if temporal:
            times = np.asarray(self.times)
            if (
                times.dtype.kind not in "biuf"
                or times.ndim != 1
                or times.size != doses.shape[0]
                or not np.all(np.isfinite(times))
            ):
                raise ValueError(
                    "TimeSeries requires finite 1D times matching the leading dose axis."
                )
        elif self.times is not None:
            raise ValueError("Static measurements require times=None.")
        if self.coordinates is not None:
            xyz = np.asarray(self.coordinates)
            if (
                xyz.dtype.kind not in "biuf"
                or xyz.shape != self.spatial_shape + (3,)
                or not np.all(np.isfinite(xyz))
            ):
                raise ValueError(
                    "Coordinates must be finite XYZ samples matching the spatial shape."
                )
            if ndim == 1 and self.startpoint is not None and self.endpoint is not None:
                self.reconstruct_xyz()

    def reconstruct_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return spatial XYZ arrays, using explicit samples or regular-axis geometry."""
        if self.type in (MeasurementType.DosePlane, MeasurementType.DoseVolume):
            self.validate()
            if self.coordinates is not None:
                xyz = np.asarray(self.coordinates, dtype=float)
                return xyz[..., 0].copy(), xyz[..., 1].copy(), xyz[..., 2].copy()
            if self.type == MeasurementType.DoseVolume:
                z, y, x = np.meshgrid(*self.axes, indexing="ij")
            else:
                y, x = np.meshgrid(*self.axes, indexing="ij")
                depth = self.startpoint[2] if self.startpoint is not None else 0.0
                if not np.isfinite(depth):
                    raise ValueError("Plane depth must be finite.")
                z = np.full_like(x, depth)
            return x, y, z
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
