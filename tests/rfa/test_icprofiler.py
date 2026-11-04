import warnings

import numpy as np
import pytest

from beamio.common.measurement import MeasurementType, RadiationType
from beamio.rfa import BeamDataFactory, ICProfiler
from tests.support.icprofiler import change_table_cell, prm_export


def read_prm(tmp_path, text=None, **options):
    path = tmp_path / "input.prm"
    path.write_text(prm_export() if text is None else text, encoding="utf-8")
    return ICProfiler(path, **options).parse()


def test_four_profiles_use_labelled_geometry_and_cumulative_corrected_counts(tmp_path):
    beam = read_prm(tmp_path)
    assert [m.type for m in beam.measurements] == [
        MeasurementType.CrosslineProfile,
        MeasurementType.InlineProfile,
        MeasurementType.DiagonalProfile,
        MeasurementType.DiagonalProfile,
    ]
    assert [len(m.doses) for m in beam.measurements] == [63, 65, 63, 63]
    for i, scan in enumerate(beam.measurements):
        assert scan.radiation_type == RadiationType.PHOTON
        assert scan.energy == 6
        assert scan.SSD == 971
        assert scan.field_size == (250, 200)
        assert scan.wedge_angle == 60
        assert scan.metadata["FluenceMode"] == "FFF"
        assert scan.metadata["FieldsizeCorners[mm]"] == "-100; -80; 150; 120"
        assert scan.metadata["GantryAngle[deg]"] == 90
        assert scan.metadata["IC Profiler"]["Frame Index"] == 1
        assert scan.metadata["IC Profiler"]["TimeTic"] == 2000
        assert scan.metadata["IC Profiler"]["Dose Scale"] == "corrected counts"
        xyz = np.column_stack(scan.reconstruct_xyz())
        np.testing.assert_array_equal(xyz[:, 2], 29)
        start = [0, 63, 128, 191][i]
        # Final raw frame, minus final TIMETIC * leakage, times array factor.
        # Do not sum the already cumulative frames or subtract BIAS1's TIMETIC.
        np.testing.assert_array_equal(
            scan.doses, (200 + np.arange(start, start + len(xyz)) - 20) * 2
        )
    cross, inline, positive, negative = beam.measurements
    np.testing.assert_array_equal(cross.positions[[0, 30, 31, 32, -1]], [-160, -10, 0, 10, 160])
    np.testing.assert_array_equal(inline.positions, np.arange(-160, 161, 5))
    for scan, sign in ((positive, 1), (negative, -1)):
        xyz = scan.coordinates
        np.testing.assert_array_equal(xyz[:, 0], cross.positions)
        np.testing.assert_array_equal(xyz[:, 1], cross.positions * sign)
        np.testing.assert_allclose(scan.positions, cross.positions * np.sqrt(2))


def test_retains_all_raw_frames_and_separate_calibration_headers(tmp_path):
    beam = read_prm(tmp_path)
    assert beam.data.shape == (2, 268)
    assert beam.columns[:4] == ("UPDATE#", "TIMETIC", "PULSES", "STATUS")
    assert beam.columns[-1] == "Temperature"
    assert len(beam.detector_ids) == 254
    np.testing.assert_array_equal(beam.data[:, beam.columns.index("X1")], [100, 200])
    np.testing.assert_array_equal(beam.leak_rates, 0.01)
    np.testing.assert_array_equal(beam.calibration, 2)
    assert not np.any(beam.ignored_detectors)
    assert beam.file_meta["Time"] == "12:34:56"
    assert beam.file_meta["Dose Calibration"]["Dose Per Count"] == "0.01"
    assert beam.file_meta["Reference Detector Dose Calibration"]["Dose Per Count"] == "123"


def test_frame_selection_and_absolute_calibration(tmp_path):
    relative = read_prm(tmp_path, frame=0)
    assert relative.measurements[0].doses[0] == 180
    absolute = read_prm(tmp_path, frame=-2, absolute=True)
    assert absolute.measurements[0].doses[0] == 1.8
    assert absolute.measurements[0].metadata["IC Profiler"]["Dose Scale"] == "absolute"
    # PRM absolute dose uses its declared Dose Per Count, not reference calibration.
    np.testing.assert_allclose(
        absolute.measurements[0].doses, relative.measurements[0].doses * 0.01
    )


def test_stop_frame_tick_can_precede_previous_tick(tmp_path):
    # Observed in three supplied exports: the stop update retains cumulative
    # counts/pulses, but its elapsed tick is a few milliseconds earlier.
    text = prm_export().replace("Data:\t1\t2000\t", "Data:\t1\t999\t")
    text = text.replace("Total Time:\t0.002", "Total Time:\t0.000999")
    beam = read_prm(tmp_path, text)
    assert beam.data[-1, beam.columns.index("TIMETIC")] == 999
    assert beam.measurements[0].metadata["IC Profiler"]["TimeTic"] == 999
    assert beam.measurements[0].doses[0] == (200 - 999 * 0.01) * 2


@pytest.mark.parametrize("frame", [2, -3, 0.5, True])
def test_invalid_frame(tmp_path, frame):
    with pytest.raises(ValueError, match="frame"):
        read_prm(tmp_path, frame=frame)


@pytest.mark.parametrize(
    "old,new",
    [
        ("Absolute Calibration:\ttrue", "Absolute Calibration:\tfalse"),
        ("Dose Per Count:\t0.01", "Dose Per Count:\t0"),
    ],
)
def test_absolute_requires_valid_acquisition_calibration(tmp_path, old, new):
    text = prm_export().replace(old, new, 1)
    read_prm(tmp_path, text)  # relative dose does not require an absolute calibration
    with pytest.raises(ValueError, match="absolute dose requires"):
        read_prm(tmp_path, text, absolute=True)


def test_ignored_channels_keep_geometry_and_raw_values(tmp_path):
    text = change_table_cell(prm_export(), "IgnoreDet", "X33", 1)
    beam = read_prm(tmp_path, text)
    scan = beam.measurements[0]
    assert len(scan.doses) == 63
    assert scan.positions[31] == 0
    assert np.isnan(scan.doses[31])
    assert scan.metadata["IC Profiler"]["Ignored Detectors"][31]
    assert scan.metadata["IC Profiler"]["Raw Counts"][31] == 231


def test_reordered_columns_preserve_exported_sample_order(tmp_path):
    text = prm_export()
    lines = text.splitlines()
    width = len(next(line for line in lines if line.startswith("TYPE\t")).split("\t"))
    for i, line in enumerate(lines):
        cells = line.split("\t")
        if cells[0] in ("TYPE", "BIAS1", "Calibration", "IgnoreDet", "Data:"):
            cells += [""] * (width - len(cells))
            lines[i] = "\t".join([cells[0], *reversed(cells[1:])])
    original = read_prm(tmp_path)
    reordered = read_prm(tmp_path, "\n".join(lines))
    for expected, actual in zip(original.measurements, reordered.measurements):
        np.testing.assert_array_equal(actual.positions, expected.positions[::-1])
        np.testing.assert_array_equal(actual.doses, expected.doses[::-1])
        np.testing.assert_array_equal(actual.coordinates, expected.coordinates[::-1])
        np.testing.assert_array_equal(np.column_stack(actual.reconstruct_xyz()), actual.coordinates)


def test_single_frame_and_negative_corrected_signal(tmp_path):
    text = change_table_cell(prm_export(frames=1), "Data:", "X1", 1)
    beam = read_prm(tmp_path, text)
    assert beam.measurements[0].metadata["IC Profiler"]["Frame Count"] == 1
    assert beam.measurements[0].doses[0] == -18


@pytest.mark.parametrize(
    "beam,energy,radiation,mode",
    [
        ("X-Ray", "10 MV", RadiationType.PHOTON, "Standard"),
        ("Electron", "12 MeV", RadiationType.ELECTRON, None),
        ("Proton", "12 MeV", RadiationType.PROTON, None),
        ("Undefined", "Undefined", None, None),
    ],
)
def test_beam_metadata(tmp_path, beam, energy, radiation, mode):
    scan = read_prm(tmp_path, prm_export(beam=beam, energy=energy)).measurements[0]
    assert scan.radiation_type == radiation
    assert scan.energy == (None if energy == "Undefined" else float(energy.split()[0]))
    assert scan.metadata.get("FluenceMode") == mode


@pytest.mark.parametrize("wedge,angle", [("None\tat\t60", 0), ("Dynamic\tat\t60", 60)])
def test_wedge_type_controls_angle(tmp_path, wedge, angle):
    assert read_prm(tmp_path, prm_export(wedge=wedge)).measurements[0].wedge_angle == angle


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_encoding_and_factory_detection_independent_of_extension(tmp_path, encoding):
    path = tmp_path / "input.dat"
    text = prm_export().replace("Synthetic acquisition", "Acquisition café")
    path.write_bytes(text.encode(encoding))
    result = BeamDataFactory.from_file(path)
    assert isinstance(result, ICProfiler)
    assert result.file_meta["Description"] == "Acquisition café"


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("Version:\t 25", "Version:\t", "Version"),
        ("IC PROFILER", "PROFILER 2", "Collector Model"),
        ("Orientation:\tY Axis", "Orientation:\tX Axis", "orientation"),
        ("Detector Spacing:\t0.5", "Detector Spacing:\t0.4", "spacing"),
        ("Concatenation:\tfalse", "Concatenation:\ttrue", "Concatenation"),
        ("Imported Data:\tfalse", "Imported Data:\ttrue", "Imported Data"),
        ("\tX33\t", "\tX32\t", "detector layout"),
        ("\tX1\tX2\t", "\tX1\tX1\t", "duplicate TYPE"),
        ("Left: 10 Right: 15", "Left: invalid Right: 15", "Collimator"),
        ("Buildup:\t2.9\tcm", "Buildup:\t2.9\tinches", "units"),
        ("Beam Type:\tX-Ray FFF", "Beam Type:\tUnknown", "Beam Type"),
    ],
)
def test_rejects_unsupported_or_malformed_headers(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        read_prm(tmp_path, prm_export().replace(old, new))


@pytest.mark.parametrize("version", ["24", "26", "26.1"])
def test_unvalidated_versions_warn_and_attempt_the_validated_layout(tmp_path, version):
    expected = read_prm(tmp_path)
    text = prm_export().replace("Version:\t 25", f"Version:\t {version}")
    with pytest.warns(UserWarning, match="is unvalidated.*tested version: 25") as caught:
        actual = read_prm(tmp_path, text)
    assert len(caught) == 1
    assert f"version {version!r}" in str(caught[0].message)
    assert "results may be incorrect" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert actual.file_meta["Version"] == version
    assert len(actual.measurements) == 4
    for result, reference in zip(actual.measurements, expected.measurements):
        np.testing.assert_array_equal(result.positions, reference.positions)
        np.testing.assert_array_equal(result.doses, reference.doses)
    np.testing.assert_array_equal(actual.time_series().doses, expected.time_series().doses)


def test_validated_version_is_quiet_and_missing_version_is_rejected(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_prm(tmp_path)
    assert not caught
    text = prm_export().replace("Version:\t 25\n", "")
    with pytest.raises(ValueError, match="nonempty Version"):
        read_prm(tmp_path, text)


def test_unvalidated_version_still_rejects_an_incomplete_acquisition(tmp_path):
    text = prm_export().replace("Version:\t 25", "Version:\t 26")
    text = "\n".join(text.splitlines()[:-1]) + "\n"
    with pytest.warns(UserWarning, match="is unvalidated"):
        with pytest.raises(ValueError, match="Total Pulses .* incomplete"):
            read_prm(tmp_path, text)


@pytest.mark.parametrize(
    "row,column,value,message",
    [
        ("Calibration", "X1", "", "Calibration for every detector"),
        ("Calibration", "X1", -1, "positive"),
        ("BIAS1", "X1", "NaN", "finite"),
        ("IgnoreDet", "X1", 2, "flags"),
        ("Data:", "X1", "NaN", "counts must be finite"),
        ("Data:", "TIMETIC", -1, "nonnegative"),
        ("Data:", "UPDATE#", 0, "increase cumulatively"),
    ],
)
def test_rejects_bad_table_values(tmp_path, row, column, value, message):
    with pytest.raises(ValueError, match=message):
        read_prm(tmp_path, change_table_cell(prm_export(), row, column, value))


@pytest.mark.parametrize("row", ["BIAS1", "Calibration", "IgnoreDet", "Data:"])
def test_rejects_missing_table_rows(tmp_path, row):
    text = "\n".join(line for line in prm_export().splitlines() if line.split("\t")[0] != row)
    with pytest.raises(ValueError):
        read_prm(tmp_path, text)


def test_truncated_data_rejected_with_line_context(tmp_path):
    lines = prm_export().splitlines()
    lines[-1] = lines[-1].rsplit("\t", 1)[0]
    with pytest.raises(ValueError, match="IC Profiler line .* does not match TYPE"):
        read_prm(tmp_path, "\n".join(lines))


def test_missing_complete_final_update_is_rejected_before_frame_selection(tmp_path):
    text = "\n".join(prm_export().splitlines()[:-1]) + "\n"
    for frame in (-1, 0):
        with pytest.raises(ValueError, match="Total Pulses .* incomplete"):
            read_prm(tmp_path, text, frame=frame)


def test_time_total_detects_a_missing_stop_frame_without_new_pulses(tmp_path):
    lines = prm_export().replace("Total Pulses:\t20", "Total Pulses:\t10").splitlines()
    cells = lines[-1].split("\t")
    cells[3] = "10"
    lines[-1] = "\t".join(cells)
    read_prm(tmp_path, "\n".join(lines))
    with pytest.raises(ValueError, match="Total Time .* incomplete"):
        read_prm(tmp_path, "\n".join(lines[:-1]))


@pytest.mark.parametrize(
    "key,value",
    [("Total Pulses", "-1"), ("Total Pulses", "1.5"), ("Total Time", "-1"), ("Total Time", "NaN")],
)
def test_invalid_summary_totals_are_rejected(tmp_path, key, value):
    old = "20" if key == "Total Pulses" else "0.002"
    text = prm_export().replace(f"{key}:\t{old}", f"{key}:\t{value}")
    with pytest.raises(ValueError, match=key):
        read_prm(tmp_path, text)


def test_native_time_rounding_and_absent_optional_totals_are_supported(tmp_path):
    read_prm(tmp_path, prm_export().replace("Total Time:\t0.002", "Total Time:\t0.0020008"))
    text = "\n".join(
        line
        for line in prm_export().splitlines()
        if not line.startswith(("Total Time:", "Total Pulses:"))
    )
    read_prm(tmp_path, text)


@pytest.mark.parametrize("value", ["Check flatness:", "Version:", "Energy:\tTime:", "C:"])
def test_free_text_values_are_not_reinterpreted_as_headers(tmp_path, value):
    text = prm_export().replace("Description:\tSynthetic acquisition", f"Description:\t{value}")
    beam = read_prm(tmp_path, text)
    assert beam.file_meta["Description"] == value
    assert beam.file_meta["Version"] == "25"
    assert beam.file_meta["Time"] == "12:34:56"
    assert beam.file_meta["Machine Data"]["Energy"] == "6 MV"


def test_compound_metadata_still_splits_only_known_labels(tmp_path):
    beam = read_prm(tmp_path)
    assert beam.file_meta["Collector Config"]["Revision"] == "L"
    assert beam.file_meta["Machine Data"]["Dose"] == "200"
    assert beam.file_meta["Machine Data"]["Collimator Angle"] == "0 deg"


def test_reparse_replaces_state_and_writing_is_explicitly_unsupported(tmp_path):
    beam = read_prm(tmp_path)
    beam.parse()
    assert len(beam.measurements) == 4
    with pytest.raises(NotImplementedError, match="PRM writing"):
        beam.save(tmp_path / "output.prm")
