import numpy as np

from beamio.common.measurement import AcquisitionType, Measurement, MeasurementType

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


def array_measurement(kind, temporal=False):
    scan = Measurement()
    scan.type = kind
    axes = (np.array([4.0, 1.0, -2.0]), np.array([-3.0, 2.0]), np.array([-2.0, -0.5, 1.0, 4.0]))
    scan.axes = axes[-kind.spatial_ndim :]
    spatial_shape = tuple(len(axis) for axis in scan.axes)
    scan.doses = np.arange(np.prod(spatial_shape), dtype=float).reshape(spatial_shape)
    if temporal:
        scan.acquisition_type = AcquisitionType.TimeSeries
        scan.times = np.array([0.1, 0.2])
        scan.doses = np.stack((scan.doses, scan.doses + 100))
    return scan
