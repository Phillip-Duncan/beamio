"""Flag-aware static-grid interpolation and native export semantics."""

import numpy as np
import pytest

from beamio.rfa import ArcCHECK, BeamDataFactory
from tests.support.arccheck import export


def boundary_case():
    x = np.array([-12.0, -8.0, -2.0, 1.0, 6.0, 9.0, 13.0])
    y = np.array([15.0, 9.0, 0.0, -3.0, -10.0])
    xx, yy = np.meshgrid(x, y)
    truth = 100 + 2 * xx + 3 * yy + 0.2 * xx * yy
    flags = np.full(truth.shape, 68)
    flags[::2, ::2] = 69
    flags[[0, -1], 1::2] = 0
    flags[1::2, 0] = 0
    flags[1::2, -1] = 0
    dose = truth.copy()
    dose[flags == 0] = 0
    return x, y, truth, dose, flags


def load(tmp_path, dose, flags, *, x=None, y=None, kind="Measured File", interpolate=True):
    path = tmp_path / "grid.txt"
    path.write_bytes(
        export(rows=dose.shape[0], cols=dose.shape[1], dose=dose, flags=flags, x=x, y=y, kind=kind)
    )
    return ArcCHECK(path).parse(interpolate=interpolate)


def test_default_fills_only_empty_perimeter_sites_using_physical_coordinates(tmp_path):
    x, y, truth, dose, flags = boundary_case()
    beam = load(tmp_path, dose, flags, x=x, y=y)
    m = beam.measurements[0]
    np.testing.assert_allclose(m.doses, truth, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(m.doses[flags != 0], dose[flags != 0])
    native = beam.file_meta["ArcCHECK"]
    np.testing.assert_array_equal(native["Tables"]["Dose Interpolated"], dose)
    np.testing.assert_array_equal(native["Tables"]["Data Flags"], flags)
    np.testing.assert_array_equal(native["Interpolation Mask"], flags == 0)
    assert not np.shares_memory(m.doses, native["Tables"]["Dose Interpolated"])
    detected = BeamDataFactory.from_file(tmp_path / "grid.txt")
    np.testing.assert_array_equal(detected.measurements[0].doses, m.doses)


def test_opt_out_retains_exact_native_arrays(tmp_path):
    x, y, truth, dose, flags = boundary_case()
    beam = load(tmp_path, dose, flags, x=x, y=y, interpolate=False)
    np.testing.assert_array_equal(beam.measurements[0].doses, dose)
    assert beam.measurements[0].doses is beam.file_meta["ArcCHECK"]["Tables"]["Dose Interpolated"]
    assert "Interpolation Mask" not in beam.file_meta["ArcCHECK"]
    output = tmp_path / "native.txt"
    beam.save(output)
    exact = ArcCHECK(output).parse(interpolate=False)
    for name, values in beam.file_meta["ArcCHECK"]["Tables"].items():
        np.testing.assert_array_equal(exact.file_meta["ArcCHECK"]["Tables"][name], values)
    np.testing.assert_allclose(ArcCHECK(output).parse().measurements[0].doses, truth, atol=1e-12)


def test_measured_and_vendor_zero_readings_are_never_treated_as_gaps(tmp_path):
    x, y, _, dose, flags = boundary_case()
    dose[0, 2] = 0  # A genuine detector zero.
    dose[1, 1] = 0  # A vendor-interpolated zero.
    dose[0, 1] = 7  # A supplied value, even though its native flag is zero.
    beam = load(tmp_path, dose, flags, x=x, y=y)
    result = beam.measurements[0].doses
    assert result[0, 2] == result[1, 1] == 0
    assert result[0, 1] == 7
    supplied = (flags != 0) | (dose != 0)
    np.testing.assert_array_equal(result[supplied], dose[supplied])


def test_no_extrapolation_beyond_available_brackets(tmp_path):
    dose = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 20.0], [0.0, 30.0, 40.0]])
    flags = (dose != 0).astype(int) * 69
    beam = load(tmp_path, dose, flags)
    np.testing.assert_array_equal(beam.measurements[0].doses, dose)
    assert "Interpolation Mask" not in beam.file_meta["ArcCHECK"]


@pytest.mark.parametrize("reverse_x,reverse_y", [(False, False), (True, False), (False, True)])
def test_sparse_merged_grid_uses_bilinear_interpolation_without_changing_detectors(
    tmp_path, reverse_x, reverse_y
):
    x = np.array([-12.0, -8.0, 0.0, 5.0, 13.0])
    y = np.array([15.0, 9.0, 0.0, -3.0, -10.0])
    if reverse_x:
        x = x[::-1]
    if reverse_y:
        y = y[::-1]
    xx, yy = np.meshgrid(x, y)
    truth = 100 + 2 * xx + 3 * yy + 0.2 * xx * yy
    flags = np.zeros(truth.shape, dtype=int)
    flags[::2, ::2] = 1
    beam = load(tmp_path, truth, flags, x=x, y=y, kind="Merged File")
    np.testing.assert_allclose(beam.measurements[0].doses, truth, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(beam.measurements[0].doses[flags == 1], truth[flags == 1])
    assert np.all(beam.file_meta["ArcCHECK"]["Tables"]["Dose Counts"][flags == 0] == 0)


def test_sparse_export_omits_synthetic_edits_and_reinterpolates_after_detector_edits(tmp_path):
    flags = np.zeros((5, 5), dtype=int)
    flags[::2, ::2] = 1
    truth = np.arange(25, dtype=float).reshape(5, 5) + 10
    beam = load(tmp_path, truth, flags, kind="Merged File")
    missing = beam.file_meta["ArcCHECK"]["Missing Mask"]
    beam.measurements[0].doses[missing] = np.nan
    beam.measurements[0].doses[2, 2] = 100
    beam.measurements[0].doses[0, 0] = 0
    output = tmp_path / "edited.txt"
    beam.save(output)
    raw = ArcCHECK(output).parse(interpolate=False)
    np.testing.assert_array_equal(raw.measurements[0].doses[missing], 0)
    assert raw.measurements[0].doses[2, 2] == 100
    assert raw.measurements[0].doses[0, 0] == 0
    reread = ArcCHECK(output).parse()
    assert np.isfinite(reread.measurements[0].doses).all()
    assert reread.measurements[0].doses[0, 0] == 0
    np.testing.assert_array_equal(reread.file_meta["ArcCHECK"]["Tables"]["Data Flags"], flags)
    # Generated neighbours are recomputed from the newly edited detector.
    assert reread.measurements[0].doses[2, 1] == (truth[2, 0] + 100) / 2
    assert np.count_nonzero(reread.file_meta["ArcCHECK"]["Tables"]["Dose Interpolated"]) == 0


def test_measured_export_writes_completed_dose_grid_but_preserves_other_source_tables(tmp_path):
    x, y, _, dose, flags = boundary_case()
    beam = load(tmp_path, dose, flags, x=x, y=y)
    original = {k: v.copy() for k, v in beam.file_meta["ArcCHECK"]["Tables"].items()}
    beam.measurements[0].doses[0, 1] = 123.5
    output = tmp_path / "completed.txt"
    beam.save(output)
    reread = ArcCHECK(output).parse(interpolate=False)
    np.testing.assert_array_equal(reread.measurements[0].doses, beam.measurements[0].doses)
    for name, values in original.items():
        if name not in ("Dose Interpolated", "Data Flags"):
            np.testing.assert_array_equal(reread.file_meta["ArcCHECK"]["Tables"][name], values)
    expected_flags = flags.copy()
    expected_flags[flags == 0] = 68
    np.testing.assert_array_equal(
        reread.file_meta["ArcCHECK"]["Tables"]["Data Flags"], expected_flags
    )
    np.testing.assert_array_equal(beam.file_meta["ArcCHECK"]["Tables"]["Data Flags"], flags)


def test_measured_zero_edits_survive_repeated_default_and_factory_round_trips(tmp_path):
    x, y, _, dose, flags = boundary_case()
    beam = load(tmp_path, dose, flags, x=x, y=y)
    original = {k: v.copy() for k, v in beam.file_meta["ArcCHECK"]["Tables"].items()}
    sites = [(0, 1), (1, 0)]
    for i, site in enumerate(sites):
        beam.measurements[0].doses[site] = 0
        expected = beam.measurements[0].doses.copy()
        output = tmp_path / f"zero-edit-{i}.txt"
        beam.save(output)
        # Export must not mutate any of the retained original source tables.
        if i == 0:
            for name, values in original.items():
                np.testing.assert_array_equal(beam.file_meta["ArcCHECK"]["Tables"][name], values)
        beam = BeamDataFactory.from_file(output)
        np.testing.assert_array_equal(beam.measurements[0].doses, expected)
        for edited in sites[: i + 1]:
            assert beam.measurements[0].doses[edited] == 0
            assert beam.file_meta["ArcCHECK"]["Tables"]["Data Flags"][edited] == 68


@pytest.mark.parametrize("interpolate", [False, True])
def test_zero_edit_to_supplied_unflagged_value_survives_export(tmp_path, interpolate):
    x, y, _, dose, flags = boundary_case()
    # A completed grid from an earlier exporter may still have zero flags.
    # No gaps remain, so the dose array shares the original native table.
    dose[flags == 0] = 7
    beam = load(tmp_path, dose, flags, x=x, y=y, interpolate=interpolate)
    beam.measurements[0].doses[0, 1] = 0
    output = tmp_path / "supplied-zero.txt"
    beam.save(output)
    reread = ArcCHECK(output).parse()
    np.testing.assert_array_equal(reread.measurements[0].doses, beam.measurements[0].doses)
    assert reread.file_meta["ArcCHECK"]["Tables"]["Data Flags"][0, 1] == 68


def test_measured_export_leaves_unfilled_padding_unflagged(tmp_path):
    x, y, _, dose, flags = boundary_case()
    dose[0, 0] = flags[0, 0] = 0  # Outside every available interpolation bracket.
    beam = load(tmp_path, dose, flags, x=x, y=y)
    filled = beam.file_meta["ArcCHECK"]["Interpolation Mask"]
    assert not filled[0, 0] and filled.any()
    output = tmp_path / "partial-grid.txt"
    beam.save(output)
    reread = ArcCHECK(output).parse()
    np.testing.assert_array_equal(reread.measurements[0].doses, beam.measurements[0].doses)
    exported_flags = reread.file_meta["ArcCHECK"]["Tables"]["Data Flags"]
    assert exported_flags[0, 0] == 0
    assert np.all(exported_flags[filled] == 68)


@pytest.mark.parametrize("key", ["Missing Mask", "Interpolation Mask"])
@pytest.mark.parametrize("invalid", [np.ones((1, 1), dtype=bool), np.ones((5, 7), dtype=float)])
def test_invalid_measured_validity_mask_preserves_existing_destination(tmp_path, key, invalid):
    x, y, _, dose, flags = boundary_case()
    beam = load(tmp_path, dose, flags, x=x, y=y)
    beam.file_meta["ArcCHECK"][key] = invalid
    output = tmp_path / "existing.txt"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="mask"):
        beam.save(output)
    assert output.read_bytes() == b"keep"


def test_all_zero_exposure_keeps_real_zeros_and_records_only_filled_padding(tmp_path):
    _, _, _, dose, flags = boundary_case()
    beam = load(tmp_path, np.zeros_like(dose), flags)
    assert not np.any(beam.measurements[0].doses)
    np.testing.assert_array_equal(beam.file_meta["ArcCHECK"]["Interpolation Mask"], flags == 0)
