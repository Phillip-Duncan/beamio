import numpy as np
import pytest

from beamio.rfa import ICProfiler, ICProfilerBeamCenter, ICProfilerTimeSeries
from tests.support.icprofiler import change_table_cell, moving_prm


def read_series(tmp_path, text=None, **options):
    path = tmp_path / "moving.prm"
    path.write_text(moving_prm() if text is None else text)
    return ICProfiler(path, **options).parse()


def test_interval_profiles_timing_and_moving_beam_center(tmp_path):
    acquisition = read_series(tmp_path)
    series = acquisition.time_series("X")
    assert isinstance(series, ICProfilerTimeSeries)
    assert series.axis == "X"
    assert not series.cumulative
    assert series.dose_scale == "corrected counts"
    assert series.doses.shape == (3, 63)
    np.testing.assert_allclose(series.times, [0.125, 0.25, 0.375])
    np.testing.assert_allclose(series.durations, 0.125)
    np.testing.assert_array_equal(series.pulses, [10, 10, 10])
    np.testing.assert_array_equal(series.updates, [0, 1, 2])
    np.testing.assert_array_equal(series.statuses, [67, 67, 67])
    assert series.valid_frames.all()
    np.testing.assert_allclose(series.doses[:, series.detector_ids.index("X33")], 100)
    signal = series.detector("X33")
    np.testing.assert_allclose(signal, [100, 100, 100])
    np.testing.assert_allclose(signal[1:3], [100, 100])
    signal[0] = -1
    assert series.detector("X33")[0] == 100
    np.testing.assert_allclose(series.field_edges(), [[-60, 40], [-50, 50], [-40, 60]])
    np.testing.assert_allclose(series.field_width(), 100)
    trend = acquisition.beam_center()
    assert isinstance(trend, ICProfilerBeamCenter)
    np.testing.assert_allclose(trend.x, [-10, 0, 10], atol=1e-10)
    np.testing.assert_allclose(trend.y, [3, -2, 7], atol=0.05)
    assert trend.method == "InflectionPoint"
    assert trend.level is None
    np.testing.assert_array_equal(trend.times, series.times)
    assert trend.valid_frames.all()


@pytest.mark.parametrize("axis,index", [("X", 0), ("Y", 1), ("PD", 2), ("ND", 3)])
def test_cumulative_series_agrees_with_integrated_measurements(tmp_path, axis, index):
    acquisition = read_series(tmp_path)
    series = acquisition.time_series(axis, cumulative=True)
    np.testing.assert_allclose(series.positions, acquisition.measurements[index].positions)
    np.testing.assert_allclose(series.doses[-1], acquisition.measurements[index].doses)
    np.testing.assert_array_equal(series.pulses, [10, 20, 30])
    # Time-series access always includes the full acquisition, even with frame=0.
    first_frame = read_series(tmp_path, frame=0)
    np.testing.assert_allclose(
        first_frame.time_series(axis).doses, acquisition.time_series(axis).doses
    )


def test_interval_and_cumulative_centers_are_distinct(tmp_path):
    acquisition = read_series(tmp_path, moving_prm(x_centers=(-10, 10), y_centers=(0, 0)))
    np.testing.assert_allclose(acquisition.beam_center(method="FWHM").x, [-10, 10])
    np.testing.assert_allclose(acquisition.beam_center(cumulative=True, method="FWHM").x, [-10, 0])


def test_absolute_time_series_uses_acquisition_calibration(tmp_path):
    relative = read_series(tmp_path)
    absolute = read_series(tmp_path, absolute=True)
    for cumulative in (False, True):
        expected = relative.time_series(cumulative=cumulative)
        actual = absolute.time_series(cumulative=cumulative)
        assert actual.dose_scale == "absolute"
        np.testing.assert_allclose(actual.doses, expected.doses * 0.01, atol=1e-12)
        np.testing.assert_allclose(actual.beam_center(), expected.beam_center(), atol=1e-10)


@pytest.mark.parametrize("last_tick", [250000, 249000])
def test_invalid_interval_timing_preserves_frame_alignment(tmp_path, last_tick):
    acquisition = read_series(
        tmp_path,
        moving_prm(
            x_centers=(0, 0, 0),
            y_centers=(0, 0, 0),
            ticks=[125000, 250000, last_tick],
        ),
    )
    series = acquisition.time_series()
    np.testing.assert_array_equal(series.valid_frames, [True, True, False])
    assert series.times[-1] == last_tick * 1e-6
    assert series.durations[-1] <= 0
    assert np.isnan(series.doses[-1]).all()
    assert np.isnan(acquisition.beam_center().x[-1])
    # The integrated final profile still has a valid elapsed acquisition time.
    assert acquisition.time_series(cumulative=True).valid_frames[-1]
    assert np.isfinite(acquisition.beam_center(cumulative=True).x[-1])


def test_no_pulses_and_low_signal_frames_are_not_reported_as_centers(tmp_path):
    no_pulses = read_series(
        tmp_path, moving_prm(amplitudes=[100, 0, 100], interval_pulses=[10, 0, 10])
    )
    series = no_pulses.time_series()
    assert not series.valid_frames[1]
    assert np.isnan(series.doses[1]).all()
    assert np.isnan(no_pulses.beam_center().x[1])
    faint = read_series(tmp_path, moving_prm(amplitudes=[100, 0.1, 100]))
    assert np.isnan(faint.beam_center().x[1])
    assert faint.time_series().valid_frames[1]
    np.testing.assert_allclose(faint.beam_center(min_signal=0).x, [-10, 0, 10], atol=1e-7)


def test_series_owns_frame_filtering_and_uses_only_valid_peaks_as_reference(tmp_path):
    series = read_series(tmp_path, moving_prm(amplitudes=[100, 100, 0.1])).time_series()
    series.doses[0] *= 10000
    series.valid_frames[0] = False
    edges = series.field_edges()
    np.testing.assert_allclose(edges[1], [-50, 50])
    assert np.isnan(edges[[0, 2]]).all()
    np.testing.assert_allclose(series.beam_center(), [np.nan, 0, np.nan])
    np.testing.assert_allclose(series.field_width(), [np.nan, 100, np.nan])
    series.valid_frames[:] = False
    assert np.isnan(series.field_edges()).all()


def test_missing_edges_are_nan_and_ignored_edge_samples_are_not_interpolated(tmp_path):
    clipped = read_series(tmp_path, moving_prm(half_width=200))
    assert np.isnan(clipped.beam_center().x).all()
    text = moving_prm(x_centers=(0,), y_centers=(0,))
    ignored_edge = read_series(tmp_path, change_table_cell(text, "IgnoreDet", "X23", 1))
    assert np.isnan(ignored_edge.beam_center().x).all()
    assert np.isfinite(ignored_edge.beam_center().y).all()
    ignored_center = read_series(tmp_path, change_table_cell(text, "IgnoreDet", "X33", 1))
    np.testing.assert_allclose(ignored_center.beam_center().x, [0])
    assert np.isnan(ignored_center.time_series().detector("X33")).all()


def test_separated_peaks_do_not_produce_a_single_field_center(tmp_path):
    series = read_series(tmp_path).time_series()
    positions = series.positions
    series.doses[0] = np.maximum(
        np.maximum(0, 100 - np.abs(positions - 60) * 10),
        np.maximum(0, 100 - np.abs(positions + 60) * 10),
    )
    assert np.isnan(series.beam_center()[0])


def test_inflection_and_fwhm_distinguish_central_peak_from_field_edges(tmp_path):
    series = read_series(tmp_path).time_series()
    positions = series.positions
    phase = np.clip((np.abs(positions) - 40) / 20, 0, 1)
    profile = (1 + np.cos(np.pi * phase)) / 2
    inside = np.abs(positions) < 30
    profile[inside] += 0.6 * (1 + np.cos(np.pi * positions[inside] / 30))
    series.doses[:] = 100 * profile
    # The known penumbra inflections are +/-50 mm. A 50% of peak threshold
    # instead finds the narrower central hump because its maximum is 220%.
    np.testing.assert_allclose(series.field_edges(), [[-50, 50]] * 3, atol=1e-10)
    np.testing.assert_allclose(series.field_width(), 100)
    assert np.all(series.field_width(method="FWHM") < 60)
    np.testing.assert_allclose(series.beam_center(), 0, atol=1e-10)


@pytest.mark.parametrize("axis", ["X", "Y", "PD", "ND"])
def test_inflection_positions_use_the_actual_detector_spacing(tmp_path, axis):
    acquisition = read_series(tmp_path, moving_prm(x_centers=(3.7,), y_centers=(3.7,)))
    series = acquisition.time_series(axis)
    np.testing.assert_allclose(series.field_edges(), [[-46.3, 53.7]], atol=0.5)
    np.testing.assert_allclose(series.beam_center(), 3.7, atol=0.5)


def test_fwhm_selection_and_threshold_fraction_are_forwarded(tmp_path):
    acquisition = read_series(tmp_path)
    trend = acquisition.beam_center(method="FWHM", level=0.25)
    assert trend.method == "FWHM"
    assert trend.level == 0.25
    for axis, actual in (("X", trend.x), ("Y", trend.y)):
        series = acquisition.time_series(axis)
        edges = series.field_edges(method="FWHM", level=0.25)
        np.testing.assert_allclose(actual, edges.mean(axis=1))
        np.testing.assert_allclose(
            series.field_width(method="FWHM", level=0.25), np.diff(edges)[:, 0]
        )


def test_fwhm_retains_linear_interpolation_on_a_trapezoid(tmp_path):
    series = read_series(tmp_path).time_series()
    series.doses[:] = 100 * np.clip((60 - np.abs(series.positions - 2.5)) / 20, 0, 1)
    for level, width in ((0.25, 110), (0.5, 100), (0.75, 90)):
        np.testing.assert_allclose(series.beam_center(method="FWHM", level=level), 2.5)
        np.testing.assert_allclose(series.field_width(method="FWHM", level=level), width)


@pytest.mark.parametrize("method", ["InflectionPoint", "FWHM"])
def test_methods_preserve_unknown_edges_instead_of_extrapolating(tmp_path, method):
    acquisition = read_series(tmp_path, moving_prm(half_width=200))
    assert np.isnan(acquisition.time_series().field_edges(method=method)).all()
    assert np.isnan(acquisition.beam_center(method=method).x).all()


def test_reversed_exported_positions_are_supported_without_mutating_source(tmp_path):
    acquisition = read_series(tmp_path)
    original_data = acquisition.data.copy()
    series = acquisition.time_series()
    series.positions = series.positions[::-1].copy()
    series.doses = series.doses[:, ::-1].copy()
    np.testing.assert_allclose(series.beam_center(), [-10, 0, 10])
    series.doses[0] = 0
    np.testing.assert_array_equal(acquisition.data, original_data)
    np.testing.assert_allclose(acquisition.beam_center().x, [-10, 0, 10])


@pytest.mark.parametrize(
    "argument,value",
    [
        ("level", 0),
        ("level", 1),
        ("level", np.nan),
        ("min_signal", -1),
        ("min_signal", 2),
        ("min_signal", np.inf),
        ("method", "Unknown"),
    ],
)
def test_invalid_detection_parameters(tmp_path, argument, value):
    with pytest.raises(ValueError, match=argument):
        read_series(tmp_path).beam_center(**{argument: value})


def test_time_series_requires_a_parsed_acquisition_and_known_axis(tmp_path):
    with pytest.raises(ValueError, match="Parse"):
        ICProfiler("").time_series()
    with pytest.raises(ValueError, match="axis"):
        read_series(tmp_path).time_series("Z")


@pytest.mark.parametrize("detector", ["X32", "Y33", "unknown"])
def test_detector_requires_an_existing_detector_on_the_selected_axis(tmp_path, detector):
    with pytest.raises(ValueError, match="not present on axis X"):
        read_series(tmp_path).time_series("X").detector(detector)
