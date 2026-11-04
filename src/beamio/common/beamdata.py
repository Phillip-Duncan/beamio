from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import ClassVar
import warnings

from beamio.common.measurement import AcquisitionType, Measurement, MeasurementType, RadiationType


class BeamData(ABC):
    """Abstract base class for beam data files."""

    # Concrete formats declare what they can represent. An undeclared
    # capability accepts nothing, including newly introduced enum members.
    supported_acquisition_types: ClassVar[frozenset[AcquisitionType]] = frozenset()
    supported_measurement_types: ClassVar[frozenset[MeasurementType]] = frozenset()
    supported_radiation_types: ClassVar[frozenset[RadiationType | None]] = frozenset()

    def __init__(self, file_path):
        """Initialize the BeamData class.

        Args:
            file_path (str): Path to the beam data file.
        """
        self.file_path = file_path
        self.measurements: list[Measurement] = []
        # File-level metadata, including retained native acquisition records.
        self.file_meta = {}

    @abstractmethod
    def parse(self) -> "BeamData":
        """Parse the beam data file.

        Args:
            file_path (str): Path to the beam data file.
        Returns:
            'BeamData': An instance of the BeamData class with parsed data.
        """
        pass

    @abstractmethod
    def save(self, output_path: str) -> None:
        """Save the beam data to a specified output path.

        Args:
            output_path (str): Path to save the beam data file.
        """
        pass

    def convert(self, target_cls: type[BeamData], target_path: str = "") -> BeamData:
        """Copy measurements supported by the target, warning once for each omission.

        Only the three declared enum capabilities are checked here. Writers
        validate array shapes, geometry and required native metadata on save.
        Order and original values are preserved; no frame/row selection,
        normalization or recalibration is performed. An entirely unsupported
        source produces an empty target with copied file metadata.
        """
        if not isinstance(target_cls, type) or not issubclass(target_cls, BeamData):
            raise TypeError("target_cls must be a BeamData subclass.")
        out = target_cls(target_path)
        accepted = []
        for index, measurement in enumerate(self.measurements, 1):
            unsupported = []
            for attribute, supported in (
                ("acquisition_type", out.supported_acquisition_types),
                ("type", out.supported_measurement_types),
                ("radiation_type", out.supported_radiation_types),
            ):
                value = getattr(measurement, attribute)
                if not any(value is item for item in supported):
                    label = (
                        value.value
                        if isinstance(value, (AcquisitionType, MeasurementType, RadiationType))
                        else repr(value)
                    )
                    unsupported.append(f"{attribute}={label}")
            if unsupported:
                warnings.warn(
                    f"Skipping measurement {index} converting to {target_cls.__name__}: "
                    f"unsupported {', '.join(unsupported)}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                accepted.append(measurement)
        # One copy operation preserves aliases between measurements and native
        # metadata (e.g. the SRS final dose table) without aliasing the source.
        out.measurements, out.file_meta = copy.deepcopy((accepted, self.file_meta))
        return out

    # ---------- Common helper methods for subclasses can be added here ----------

    @staticmethod
    def is_numeric(s: str) -> bool:
        """Check if a string can be converted to a float."""
        try:
            float(s)
            return True
        except ValueError:
            return False
