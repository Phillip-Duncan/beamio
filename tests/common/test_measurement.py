import numpy as np
import pytest

from beamio.common.measurement import MeasurementType


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
def test_diagonal_coordinates_preserve_axes_and_scan_direction(measurement, reverse, sign):
    measurement.type = MeasurementType.DiagonalProfile
    xyz = np.array([[-6, -8 * sign, 50], [0, 0, 50], [6, 8 * sign, 50]], dtype=float)
    positions = np.array([-10, 0, 10], dtype=float)
    if reverse:
        xyz, positions = xyz[::-1], positions[::-1]
    measurement.startpoint, measurement.endpoint = tuple(xyz[0]), tuple(xyz[-1])
    measurement.positions = positions
    np.testing.assert_allclose(np.column_stack(measurement.reconstruct_xyz()), xyz, atol=1e-12)


def test_offset_diagonal_preserves_distance_from_axis(measurement):
    measurement.type = MeasurementType.DiagonalProfile
    measurement.startpoint, measurement.endpoint = (-2, -11, 50), (10, 5, 50)
    measurement.positions = np.array([-10, 0, 10], dtype=float)
    np.testing.assert_allclose(
        np.column_stack(measurement.reconstruct_xyz()),
        [[-2, -11, 50], [4, -3, 50], [10, 5, 50]],
        atol=1e-12,
    )


def test_stored_coordinates_preserve_measured_motion(measurement):
    measurement.coordinates = np.array([[0, 0, 0], [0.01, -0.02, 10], [0, 0, 20]])
    np.testing.assert_array_equal(
        np.column_stack(measurement.reconstruct_xyz()), measurement.coordinates
    )


def test_stored_coordinates_reject_stale_position_edits(measurement):
    measurement.coordinates = np.array([[0, 0, 0], [0.01, -0.02, 10], [0, 0, 20]])
    measurement.positions *= 2
    with pytest.raises(ValueError, match="no longer match"):
        measurement.reconstruct_xyz()
    measurement.coordinates = None
    np.testing.assert_array_equal(measurement.reconstruct_xyz()[2], [0, 20, 40])
