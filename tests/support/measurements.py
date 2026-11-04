import numpy as np

from beamio.common.measurement import MeasurementType

AXIAL_SCANS = [
    (MeasurementType.DepthDose, (0, 0, 0), (0, 0, 20), [0, 10, 20]),
    (MeasurementType.CrosslineProfile, (-10, 0, 100), (10, 0, 100), [-10, 0, 10]),
    (MeasurementType.InlineProfile, (0, -10, 100), (0, 10, 100), [-10, 0, 10]),
]


def profile_samples(*, center=2.5, half_width=50.0):
    """A flat field with raised-cosine edges at center +/- half_width, in mm."""
    positions = np.arange(-160, 161, 5, dtype=float)
    phase = np.clip((np.abs(positions - center) - (half_width - 10)) / 20, 0, 1)
    return positions, 50 * (1 + np.cos(np.pi * phase))


def assert_scan_equal(actual, expected):
    np.testing.assert_allclose(actual.positions, expected.positions)
    np.testing.assert_allclose(actual.doses, expected.doses)
    for attr in ("type", "radiation_type", "energy", "SSD", "field_size", "startpoint"):
        assert getattr(actual, attr) == getattr(expected, attr)
