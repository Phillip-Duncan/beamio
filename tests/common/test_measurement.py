import numpy as np
import pytest

from beamio.common import AcquisitionType, Measurement, MeasurementType
from tests.support.measurements import array_measurement


@pytest.mark.parametrize("kind", list(MeasurementType))
@pytest.mark.parametrize("temporal", [False, True])
def test_spatial_and_temporal_shape_contract(kind, temporal):
    scan = array_measurement(kind, temporal)
    before = scan.doses.copy()
    scan.validate()
    assert len(scan.axes) == kind.spatial_ndim
    assert scan.spatial_shape == tuple(len(axis) for axis in scan.axes)
    assert scan.doses.ndim == kind.spatial_ndim + int(temporal)
    np.testing.assert_array_equal(scan.doses, before)
    if temporal:
        assert scan.doses[0].shape == scan.spatial_shape
        assert scan.times.shape == (2,)
    else:
        assert scan.times is None


def test_positions_is_a_view_of_the_single_axis(measurement):
    assert measurement.axes[0] is measurement.positions
    measurement.axes = (np.array([1.0, 2.0, 3.0]),)
    np.testing.assert_array_equal(measurement.positions, [1, 2, 3])
    measurement.positions[0] = -1
    assert measurement.axes[0][0] == -1
    plane = array_measurement(MeasurementType.DosePlane)
    with pytest.raises(ValueError, match="use axes"):
        _ = plane.positions


@pytest.mark.parametrize("attribute", ["axes", "positions"])
def test_complex_axes_are_rejected_without_changing_existing_geometry(measurement, attribute):
    original = measurement.positions.copy()
    complex_axis = np.array([0 + 0j, 1 + 2j, 3 + 0j])
    with pytest.raises(ValueError, match="real coordinates"):
        setattr(measurement, attribute, (complex_axis,) if attribute == "axes" else complex_axis)
    np.testing.assert_array_equal(measurement.positions, original)


@pytest.mark.parametrize("kind", [MeasurementType.DosePlane, MeasurementType.DoseVolume])
@pytest.mark.parametrize("temporal", [False, True])
def test_grids_reconstruct_xyz_without_repeating_geometry_over_time(kind, temporal):
    scan = array_measurement(kind, temporal)
    x, y, z = scan.reconstruct_xyz()
    for component in (x, y, z):
        assert component.shape == scan.spatial_shape
    np.testing.assert_array_equal(x[..., 0], np.full(scan.spatial_shape[:-1], scan.axes[-1][0]))
    np.testing.assert_array_equal(x[..., -1], np.full(scan.spatial_shape[:-1], scan.axes[-1][-1]))
    if kind is MeasurementType.DoseVolume:
        np.testing.assert_array_equal(z[:, 0, 0], scan.axes[0])
        np.testing.assert_array_equal(y[0, :, 0], scan.axes[1])
    else:
        np.testing.assert_array_equal(y[:, 0], scan.axes[0])
        np.testing.assert_array_equal(z, 0)
    # Dense coordinates can describe rotated/non-axial geometry explicitly.
    scan.coordinates = np.stack((z + 5, y - 3, x + 10), axis=-1)
    scan.validate()
    actual = np.stack(scan.reconstruct_xyz(), axis=-1)
    np.testing.assert_array_equal(actual, scan.coordinates)
    actual[...] = 999
    assert not np.any(scan.coordinates == 999)


def test_plane_depth_and_volume_singleton_axes():
    plane = array_measurement(MeasurementType.DosePlane)
    plane.startpoint = (-2, -3, 45)
    np.testing.assert_array_equal(plane.reconstruct_xyz()[2], 45)
    volume = Measurement()
    volume.type = MeasurementType.DoseVolume
    volume.axes = (np.array([7.0]), np.array([8.0]), np.array([9.0]))
    volume.doses = np.array([[[10.0]]])
    volume.validate()
    np.testing.assert_array_equal(np.stack(volume.reconstruct_xyz(), axis=-1), [[[[9, 8, 7]]]])


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("kind", [MeasurementType.DosePlane, MeasurementType.DoseVolume])
@pytest.mark.parametrize("field", ["doses", "axes", "coordinates", "times"])
def test_rejects_inconsistent_grid_arrays(kind, temporal, field):
    scan = array_measurement(kind, temporal)
    if field == "doses":
        scan.doses = scan.doses[..., 0]
    elif field == "axes":
        scan.axes = scan.axes[:-1]
    elif field == "coordinates":
        scan.coordinates = np.zeros(scan.doses.shape + (3,)) if temporal else np.zeros((1, 3))
    elif temporal:
        scan.times = np.array([0.1])
    else:
        scan.times = np.array([])
    with pytest.raises(ValueError):
        scan.validate()


@pytest.mark.parametrize("axis", [[0, 0], [0, np.nan], [0, np.inf], [[0, 1]]])
def test_grid_axes_reject_invalid_coordinates(axis):
    scan = array_measurement(MeasurementType.DosePlane)
    scan.axes = (np.asarray(axis), scan.axes[-1])
    with pytest.raises(ValueError, match="axes"):
        scan.validate()


def test_validation_preserves_missing_values_and_instrument_timestamp_order():
    scan = array_measurement(MeasurementType.DoseVolume, temporal=True)
    scan.doses[0, 0, 0, 0] = np.nan
    scan.times = np.array([0.2, 0.199])
    scan.validate()
    np.testing.assert_array_equal(scan.times, [0.2, 0.199])
    assert np.isnan(scan.doses[0, 0, 0, 0])
    scan.times[0] = np.nan
    with pytest.raises(ValueError, match="times"):
        scan.validate()


@pytest.mark.parametrize("times", [None, np.array(0.1), np.array([0.1]), np.array([0.1, np.inf])])
def test_time_series_requires_one_finite_time_per_frame(times):
    scan = array_measurement(MeasurementType.DoseVolume, temporal=True)
    scan.times = times
    with pytest.raises(ValueError, match="times"):
        scan.validate()


@pytest.mark.parametrize("field", ["type", "acquisition_type"])
def test_validation_requires_explicit_supported_enums(field):
    scan = array_measurement(MeasurementType.DoseVolume)
    setattr(scan, field, "unknown")
    with pytest.raises(ValueError, match="Type"):
        scan.validate()


def test_validation_rejects_empty_volumes_invalid_axes_and_nonfinite_xyz():
    volume = array_measurement(MeasurementType.DoseVolume)
    volume.doses = volume.doses[:0]
    with pytest.raises(ValueError, match="nonempty"):
        volume.validate()
    volume = array_measurement(MeasurementType.DoseVolume)
    volume.axes = (volume.axes[0][:-1], *volume.axes[1:])
    with pytest.raises(ValueError, match="spatial shape"):
        volume.validate()
    volume = array_measurement(MeasurementType.DoseVolume)
    volume.coordinates = np.stack(volume.reconstruct_xyz(), axis=-1)
    volume.coordinates[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Coordinates"):
        volume.validate()


def test_profiles_keep_duplicate_positions_and_reject_coordinate_time_axes(measurement):
    measurement.positions = np.array([0, 0, 20], dtype=float)
    measurement.validate()
    measurement.acquisition_type = AcquisitionType.TimeSeries
    measurement.times = np.array([0.1, 0.2])
    measurement.doses = np.stack((measurement.doses, measurement.doses))
    measurement.coordinates = np.zeros((2, 3, 3))
    with pytest.raises(ValueError, match="Coordinates"):
        measurement.validate()


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
