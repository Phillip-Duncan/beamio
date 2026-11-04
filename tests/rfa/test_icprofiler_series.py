"""IC Profiler exposes native temporal Measurements for caller-side NumPy operations."""

import numpy as np
import pytest

from beamio.common import AcquisitionType
from beamio.profile import beam_center, field_edges
from beamio.rfa import ICProfiler
from tests.support.icprofiler import change_table_cell, moving_prm, prm_export


def read_series(tmp_path, text=None):
    path = tmp_path / "moving.prm"
    path.write_text(moving_prm() if text is None else text)
    return ICProfiler(path).parse()


def test_acquisition_uses_four_shared_profile_time_series(tmp_path):
    beam = read_series(tmp_path)
    assert len(beam.measurements) == 4
    for scan, count in zip(beam.measurements, (63, 65, 63, 63)):
        assert scan.acquisition_type is AcquisitionType.TimeSeries
        assert scan.doses.shape == (3, count)
        assert scan.coordinates.shape == (count, 3)
        assert scan.axes[0] is scan.positions
        np.testing.assert_allclose(scan.times, [0.125, 0.25, 0.375])
        assert scan.metadata["Cumulative"] is True
        assert scan.metadata["Dose Scale"] == "corrected counts"
        scan.validate()
    assert beam.measurements[0].metadata["IC Profiler"]["Detector IDs"][31] == "X33"


@pytest.mark.parametrize("axis", range(4))
def test_numpy_selects_frames_and_detectors_without_changing_measurement(tmp_path, axis):
    series = read_series(tmp_path).measurements[axis]
    assert series.doses[-1].shape == series.positions.shape
    assert series.doses[1:3].shape == (2, len(series.positions))
    assert series.doses[:, 31].shape == series.times.shape
    assert series.doses.shape[0] == 3
    assert series.acquisition_type is AcquisitionType.TimeSeries


def test_users_calculate_intervals_and_cumulative_signals_with_numpy(tmp_path):
    scan = read_series(tmp_path).measurements[0]
    interval = np.diff(scan.doses, axis=0, prepend=np.zeros_like(scan.doses[:1]))
    np.testing.assert_allclose(np.cumsum(interval, axis=0), scan.doses, atol=1e-10)
    centers = np.array([beam_center(scan.positions, values) for values in interval])
    np.testing.assert_allclose(centers, [-10, 0, 10], atol=1e-7)
    assert np.isfinite(scan.doses).all()


@pytest.mark.parametrize("last_tick", [250000, 249999])
def test_timing_anomalies_preserve_all_frames_and_timestamps(tmp_path, last_tick):
    scan = read_series(tmp_path, moving_prm(ticks=[125000, 250000, last_tick])).measurements[0]
    assert scan.times[-1] == last_tick * 1e-6
    assert np.diff(scan.times)[-1] <= 0
    assert np.isfinite(scan.doses).all()
    assert scan.doses.shape[0] == 3
    scan.validate()


def test_beam_off_updates_are_retained_for_caller_side_filtering(tmp_path):
    beam = read_series(tmp_path, moving_prm(amplitudes=[100, 0, 100], interval_pulses=[10, 0, 10]))
    scan = beam.measurements[0]
    intervals = np.diff(scan.doses, axis=0, prepend=np.zeros_like(scan.doses[:1]))
    np.testing.assert_allclose(intervals[1], 0, atol=1e-10)
    assert np.isfinite(scan.doses).all()
    pulses = np.diff(beam.file_meta["IC Profiler"]["Pulses"], prepend=0)
    np.testing.assert_array_equal(pulses > 0, [True, False, True])


def test_dose_per_count_is_available_for_explicit_scaling(tmp_path):
    beam = read_series(tmp_path)
    factor = float(beam.file_meta["Dose Calibration"]["Dose Per Count"])
    for scan in beam.measurements:
        original = scan.doses.copy()
        scaled = scan.doses * factor
        np.testing.assert_allclose(scaled, original * 0.01, atol=1e-10)
        np.testing.assert_array_equal(scan.doses, original)


def test_ignored_detectors_keep_geometry_and_nan_for_every_frame(tmp_path):
    beam = read_series(tmp_path, change_table_cell(moving_prm(), "IgnoreDet", "X33", 1))
    scan = beam.measurements[0]
    assert np.isnan(scan.doses[:, 31]).all()
    assert scan.positions[31] == 0
    assert beam.file_meta["IC Profiler"]["Ignored Detectors"][31]


@pytest.mark.parametrize("method", ["InflectionPoint", "FWHM"])
def test_profile_analysis_stays_in_public_numpy_functions(tmp_path, method):
    x, y = read_series(tmp_path).measurements[:2]
    x_intervals = np.diff(x.doses, axis=0, prepend=np.zeros_like(x.doses[:1]))
    y_intervals = np.diff(y.doses, axis=0, prepend=np.zeros_like(y.doses[:1]))
    centers = [beam_center(x.positions, frame, method=method) for frame in x_intervals]
    np.testing.assert_allclose(centers, [-10, 0, 10], atol=0.5)
    for frame in y_intervals:
        assert np.isfinite(field_edges(y.positions, frame, method=method)).all()


def test_single_frame_retains_its_time_dimension(tmp_path):
    scan = read_series(tmp_path, prm_export(frames=1)).measurements[0]
    assert scan.acquisition_type is AcquisitionType.TimeSeries
    assert scan.doses.shape == (1, 63)
    assert scan.times.shape == (1,)
