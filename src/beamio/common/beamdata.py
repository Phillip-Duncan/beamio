from abc import ABC, abstractmethod
import os
import copy

import numpy as np

from beamio.common.measurement import Measurement


class BeamData(ABC):
    """ Abstract base class for beam data files. """

    def __init__(self, file_path):
        """Initialize the BeamData class.

        Args:
            file_path (str): Path to the beam data file.
        """
        self.file_path = file_path
        self.measurements: list[Measurement] = []
        self.file_meta = {}  # Dictionary to hold any file-level metadata (e.g., machine settings, date, etc.)

    @abstractmethod
    def parse(self) -> 'BeamData':
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

    def convert(self, target_cls: 'BeamData', target_path="") -> 'BeamData':
        """Convert the beam data to a different format.

        Args:
            target_cls (BeamData): The target format to convert to (e.g., 'IBAAccept6', 'RayStation').
            target_path (str): Optional path, really just used if parse() called, but this shouldn't be the case
        Returns:
            'BeamData': An instance of the BeamData class in the target format.
        """
        # This method can be implemented to use specific conversion logic based on the target format

        out = target_cls(target_path)      

        # Copy measurements and metadata to the new instance
        out.measurements = copy.deepcopy(self.measurements)  # This is a deep copy to ensure independent instances
        out.file_meta = copy.deepcopy(self.file_meta)
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
