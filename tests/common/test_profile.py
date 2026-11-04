import numpy as np
import pytest

from beamio.profile import beam_center, field_edges, field_width
from tests.support.measurements import profile_samples


@pytest.mark.parametrize("method", ["InflectionPoint", "FWHM"])
def test_public_functions_measure_a_single_profile_in_supplied_units(method):
    positions, doses = profile_samples()
    edges = field_edges(positions, doses, method=method)
    assert edges.shape == (2,)
    np.testing.assert_allclose(edges, [-47.5, 52.5], atol=1e-10)
    assert isinstance(field_width(positions, doses, method=method), float)
    assert isinstance(beam_center(positions, doses, method=method), float)
    assert field_width(positions, doses, method=method) == pytest.approx(100)
    assert beam_center(positions, doses, method=method) == pytest.approx(2.5)
    assert field_width(positions / 10, doses, method=method) == pytest.approx(10)


def test_minimum_signal_is_a_threshold_in_this_profiles_signal_units():
    positions, doses = profile_samples()
    assert field_width(positions, doses, min_signal=100) == pytest.approx(100)
    assert np.isnan(field_edges(positions, doses, min_signal=101)).all()
    assert np.isnan(beam_center(positions, doses * 0.01, min_signal=2))
    assert field_width(positions, doses * 0.001) == pytest.approx(100)


@pytest.mark.parametrize("method", ["InflectionPoint", "FWHM"])
def test_unsorted_samples_and_repeated_coordinates_are_analysed_without_mutation(method):
    positions, doses = profile_samples()
    repeated_positions = np.repeat(positions, 2)[::-1]
    repeated_doses = (np.repeat(doses, 2) + np.tile([-1.0, 1.0], len(doses)))[::-1]
    before_positions, before_doses = repeated_positions.copy(), repeated_doses.copy()
    np.testing.assert_allclose(
        field_edges(repeated_positions, repeated_doses, method=method), [-47.5, 52.5]
    )
    np.testing.assert_array_equal(repeated_positions, before_positions)
    np.testing.assert_array_equal(repeated_doses, before_doses)


def test_repeated_positions_average_available_readings_and_retain_missing_edges():
    positions, doses = profile_samples(center=0)
    x, values = np.repeat(positions, 2), np.repeat(doses, 2)
    values[::2] = np.nan
    np.testing.assert_allclose(field_edges(x, values), [-50, 50])
    values[x == -50] = np.nan
    assert np.isnan(field_edges(x, values)).all()


@pytest.mark.parametrize("method", ["InflectionPoint", "FWHM"])
@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_nonfinite_signals_are_missing_samples_not_false_edges(method, missing):
    positions, doses = profile_samples(center=0)
    doses[positions == -50] = missing
    assert np.isnan(field_edges(positions, doses, method=method)).all()


@pytest.mark.parametrize("missing_position", [-50, 50])
def test_missing_penumbra_cannot_be_replaced_by_an_internal_peaked_profile_gradient(
    missing_position,
):
    positions = np.arange(-160, 161, 5, dtype=float)
    phase = np.clip((np.abs(positions) - 47.5) / 5, 0, 1)
    doses = 50 * (1 + np.cos(np.pi * phase))
    inside = np.abs(positions) < 30
    doses[inside] += 60 * (1 + np.cos(np.pi * positions[inside] / 30))
    np.testing.assert_allclose(field_edges(positions, doses), [-50, 50])
    # Missing interior samples remain usable; a missing outer penumbra does not.
    interior_missing = doses.copy()
    interior_missing[positions == 0] = np.nan
    np.testing.assert_allclose(field_edges(positions, interior_missing), [-50, 50])
    doses[positions == missing_position] = np.nan
    assert np.isnan(field_edges(positions, doses)).all()
    assert np.isnan(field_width(positions, doses))
    assert np.isnan(beam_center(positions, doses))


def test_exterior_missing_samples_are_conservatively_ambiguous_for_inflection():
    positions, doses = profile_samples()
    doses[0] = np.nan
    assert np.isnan(field_edges(positions, doses)).all()


@pytest.mark.parametrize(
    "positions,doses,message",
    [
        ([[0, 1, 2]], [0, 1, 0], "positions"),
        ([0, np.nan, 2], [0, 1, 0], "positions"),
        ([0, 1, np.inf], [0, 1, 0], "positions"),
        ([0, 1, 2], [0, 1], "doses"),
        ([0, 1, 2], 1, "doses"),
        ([0, 1, 2], np.ones((2, 3)), "doses"),
        ([0, 1, 2], np.ones((2, 2, 3)), "doses"),
    ],
)
def test_malformed_arrays_have_clear_errors(positions, doses, message):
    with pytest.raises(ValueError, match=message):
        field_edges(positions, doses)


@pytest.mark.parametrize("option,value", [("method", "unknown"), ("level", 1), ("min_signal", -1)])
def test_public_options_are_validated(option, value):
    positions, doses = profile_samples()
    with pytest.raises(ValueError, match=option):
        field_edges(positions, doses, **{option: value})


def test_insufficient_samples_return_nan():
    assert np.isnan(field_edges([], [])).all()
    assert np.isnan(field_width([0, 1], [1, 1]))
    assert np.isnan(beam_center([0], [1]))
