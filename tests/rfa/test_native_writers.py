"""Native writers preserve source records and serialize shared-array edits."""

from copy import deepcopy
import zlib

import numpy as np
import pytest

from beamio.common import AcquisitionType, MeasurementType
from beamio.rfa import BeamDataFactory, ICProfiler, PTWMCC, SRSMapCheck
from tests.support.icprofiler import change_table_cell, moving_prm, prm_export
from tests.support.ptw import mcc_export, mcc_scan
from tests.support.srsmapcheck import movie_export, measured_export


def load(tmp_path, cls, text):
    path = tmp_path / "source.dat"
    path.write_text(text, encoding="utf-8")
    return cls(path).parse()


@pytest.mark.parametrize("curve", ["PDD", "CROSSPLANE_PROFILE", "INPLANE_PROFILE"])
@pytest.mark.parametrize("reference", [False, True])
def test_mcc_round_trip_all_axial_types_and_reference_data(tmp_path, curve, reference):
    rows = ["2 5", "0 nan", "0 inf", "-1 9"]
    if reference:
        rows = [row + " 2.345678901234567" for row in rows]
    text = mcc_export(mcc_scan({"SCAN_CURVETYPE": curve}, rows=rows, number=7))
    beam = load(tmp_path, PTWMCC, text)
    output = tmp_path / "roundtrip.mcc"
    beam.save(output)
    actual = BeamDataFactory.from_file(output)
    assert isinstance(actual, PTWMCC)
    scan, expected = actual.measurements[0], beam.measurements[0]
    np.testing.assert_array_equal(scan.positions, expected.positions)
    np.testing.assert_array_equal(scan.doses, expected.doses)
    np.testing.assert_array_equal(scan.coordinates, expected.coordinates)
    assert scan.metadata["PTW Scan Number"] == 7
    assert scan.metadata["COMMENT"] == "Synthetic=value"
    if reference:
        np.testing.assert_array_equal(
            scan.metadata["PTW Reference"], expected.metadata["PTW Reference"]
        )


def test_mcc_writes_generic_measurements_and_core_field_edits(tmp_path, measurement):
    beam = PTWMCC("")
    measurement.energy = 10
    measurement.doses *= 2
    beam.measurements = [measurement]
    output = tmp_path / "generic.mcc"
    beam.save(output)
    scan = PTWMCC(output).parse().measurements[0]
    assert scan.type is MeasurementType.DepthDose
    assert scan.energy == 10 and scan.SSD == 900
    np.testing.assert_array_equal(scan.doses, [80, 200, 170])


@pytest.mark.parametrize("problem", ["shape", "geometry", "reference", "temporal", "duplicate"])
def test_mcc_preflight_preserves_existing_destination(tmp_path, problem):
    beam = load(tmp_path, PTWMCC, mcc_export())
    scan = beam.measurements[0]
    if problem == "shape":
        scan.doses = scan.doses[:1]
    elif problem == "geometry":
        scan.coordinates[1, 1] += 1
    elif problem == "reference":
        scan.metadata["PTW Reference"] = np.ones(2)
    elif problem == "temporal":
        scan.acquisition_type = AcquisitionType.TimeSeries
    else:
        beam.measurements.append(deepcopy(scan))
    output = tmp_path / "existing.mcc"
    output.write_text("keep")
    with pytest.raises(ValueError):
        beam.save(output)
    assert output.read_text() == "keep"


@pytest.mark.parametrize("ignored", [False, True])
@pytest.mark.parametrize("text", [prm_export(), moving_prm(ticks=[125000, 250000, 249999])])
def test_prm_round_trip_preserves_native_signals_and_all_auxiliary_channels(
    tmp_path, text, ignored
):
    text = change_table_cell(text, "Data:", "Temperature", "12.3456789012345")
    if ignored:
        text = change_table_cell(text, "IgnoreDet", "X33", 1)
    beam = load(tmp_path, ICProfiler, text)
    output = tmp_path / "roundtrip.prm"
    beam.save(output)
    actual = BeamDataFactory.from_file(output)
    assert isinstance(actual, ICProfiler)
    for scan, expected in zip(actual.measurements, beam.measurements):
        np.testing.assert_allclose(scan.doses, expected.doses, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(scan.times, expected.times)
        np.testing.assert_array_equal(scan.coordinates, expected.coordinates)
    for key in ("Definitions", "Ignored Detector IDs", "Detector IDs"):
        assert actual.file_meta["IC Profiler"][key] == beam.file_meta["IC Profiler"][key]
    for key, values in beam.file_meta["IC Profiler"]["Auxiliary Channels"].items():
        np.testing.assert_array_equal(
            actual.file_meta["IC Profiler"]["Auxiliary Channels"][key], values
        )
    np.testing.assert_array_equal(
        actual.file_meta["IC Profiler"]["Ignored Raw Counts"],
        beam.file_meta["IC Profiler"]["Ignored Raw Counts"],
    )


def test_prm_writes_edited_corrected_signals_and_timestamps(tmp_path):
    beam = load(tmp_path, ICProfiler, prm_export())
    beam.measurements[0].doses[:, 2] += 7.5
    for scan in beam.measurements:
        scan.times = scan.times + 0.000003
    beam.file_meta["IC Profiler"]["Auxiliary Channels"]["Temperature"][0] = 31
    output = tmp_path / "edited.prm"
    beam.save(output)
    actual = ICProfiler(output).parse()
    for scan, expected in zip(actual.measurements, beam.measurements):
        np.testing.assert_allclose(scan.doses, expected.doses, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(scan.times, expected.times)
    assert actual.file_meta["IC Profiler"]["Auxiliary Channels"]["Temperature"][0] == 31


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
def test_prm_preserves_encoding_and_literal_header_values(tmp_path, encoding):
    path = tmp_path / "source.prm"
    path.write_bytes(
        prm_export().replace("Synthetic acquisition", "caf\u00e9 Version:").encode(encoding)
    )
    source = ICProfiler(path).parse()
    source.file_meta = dict(sorted(source.file_meta.items()))
    output = tmp_path / "written.prm"
    source.save(output)
    actual = ICProfiler(output).parse()
    assert isinstance(BeamDataFactory.from_file(output), ICProfiler)
    assert actual.file_meta["Description"] == "caf\u00e9 Version:"
    assert actual.file_meta["IC Profiler"]["Encoding"] == encoding


@pytest.mark.parametrize(
    "problem", ["missing", "times", "geometry", "energy", "counter", "fractional_time", "scale"]
)
def test_prm_preflight_preserves_existing_destination(tmp_path, problem):
    beam = load(tmp_path, ICProfiler, prm_export())
    scan = beam.measurements[0]
    if problem == "missing":
        beam.file_meta["IC Profiler"].pop("Auxiliary Channels")
    elif problem == "times":
        scan.times = scan.times + 0.001
    elif problem == "geometry":
        scan.coordinates[1, 0] += 1
    elif problem == "energy":
        scan.energy = 12
    elif problem == "counter":
        beam.file_meta["IC Profiler"]["Pulses"][0] = -1
    elif problem == "fractional_time":
        for scan in beam.measurements:
            scan.times = scan.times + 0.0000001
    else:
        scan.metadata["Dose Scale"] = "dose"
    output = tmp_path / "existing.prm"
    output.write_text("keep")
    with pytest.raises(ValueError):
        beam.save(output)
    assert output.read_text() == "keep"


@pytest.mark.parametrize("buildup_cm", [-1.0, -1e-9])
def test_prm_rejects_negative_buildup_before_replacing_output(tmp_path, buildup_cm):
    beam = load(tmp_path, ICProfiler, prm_export())
    beam.file_meta["Profiler Setup"]["Buildup"] = f"{buildup_cm:.17g}\tcm\tWaterEquiv"
    for scan in beam.measurements:
        scan.coordinates[:, 2] = buildup_cm * 10
        scan.startpoint = tuple(scan.coordinates[0])
        scan.endpoint = tuple(scan.coordinates[-1])
        scan.validate()
    output = tmp_path / "existing.prm"
    output.write_bytes(b"keep existing acquisition")
    with pytest.raises(ValueError, match="Buildup must be nonnegative"):
        beam.save(output)
    assert output.read_bytes() == b"keep existing acquisition"


def test_prm_accepts_zero_buildup(tmp_path):
    beam = load(tmp_path, ICProfiler, prm_export())
    beam.file_meta["Profiler Setup"]["Buildup"] = "0\tcm\tWaterEquiv"
    for scan in beam.measurements:
        scan.coordinates[:, 2] = 0
        scan.startpoint = tuple(scan.coordinates[0])
        scan.endpoint = tuple(scan.coordinates[-1])
    output = tmp_path / "zero_buildup.prm"
    beam.save(output)
    actual = ICProfiler(output).parse()
    for scan in actual.measurements:
        np.testing.assert_array_equal(scan.coordinates[:, 2], 0)


@pytest.mark.parametrize("make_export", [movie_export, measured_export])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
def test_srs_round_trip_preserves_all_native_records_and_regenerates_crc(
    tmp_path, make_export, encoding
):
    source = tmp_path / "source.dat"
    source.write_bytes(
        make_export().replace("Synthetic export", "Synthetic caf\u00e9 export").encode(encoding)
    )
    beam = SRSMapCheck(source).parse()
    output = tmp_path / "roundtrip.dat"
    beam.save(output)
    actual = BeamDataFactory.from_file(output)
    np.testing.assert_array_equal(actual.measurements[0].doses, beam.measurements[0].doses)
    assert actual.file_meta["Header"]["File Name"] == "Synthetic caf\u00e9 export"
    raw = output.read_bytes().splitlines(keepends=True)
    assert int(raw[-1].strip(), 16) == zlib.crc32(b"".join(raw[:-1]))
    before, after = beam.file_meta["SRS MapCHECK"], actual.file_meta["SRS MapCHECK"]
    assert after["Encoding"] == encoding
    if "Tables" in before:
        assert after["Tables"].keys() == before["Tables"].keys()
        for key, values in before["Tables"].items():
            np.testing.assert_array_equal(after["Tables"][key], values)
        for key in ("Rows", "Columns", "X", "Y"):
            np.testing.assert_array_equal(after["Grid"][key], before["Grid"][key])
    else:
        assert after["Columns"] == before["Columns"]
        np.testing.assert_array_equal(after["Data"], before["Data"])
        for name, values in before["Corrections"].items():
            np.testing.assert_array_equal(after["Corrections"][name], values)
        # Auxiliary correction cells, including blank cells, must survive.
        for name, cells in before["Correction Rows"].items():
            assert after["Correction Rows"][name][:7] == cells[:7]
            assert after["Correction Rows"][name][-1] == cells[-1]


def test_srs_measured_writer_uses_edited_measurement_and_retains_intermediate_tables(tmp_path):
    beam = load(tmp_path, SRSMapCheck, measured_export())
    original = beam.file_meta["SRS MapCHECK"]["Tables"]["Raw Counts"].copy()
    beam.measurements[0].doses = beam.measurements[0].doses * 3
    beam.measurements[0].energy = 10
    output = tmp_path / "edited.txt"
    beam.save(output)
    actual = SRSMapCheck(output).parse()
    assert actual.measurements[0].energy == 10
    np.testing.assert_array_equal(actual.measurements[0].doses, beam.measurements[0].doses)
    np.testing.assert_array_equal(
        actual.file_meta["SRS MapCHECK"]["Tables"]["Raw Counts"], original
    )


def test_srs_movie_writer_keeps_auxiliaries_and_writes_detector_edits(tmp_path):
    beam = load(tmp_path, SRSMapCheck, movie_export())
    native = beam.file_meta["SRS MapCHECK"]
    scan = beam.measurements[0]
    scan.doses[:, 2, 0] = [777, 888]  # First detector; derived gaps are rebuilt on reload.
    scan.times = scan.times + 0.002
    native["Data"][:, native["Columns"].index("T1")] = [30, 31]
    native["Corrections"]["Calibration(AP)"][0] = 1.234
    output = tmp_path / "edited.smcm"
    beam.save(output)
    actual = SRSMapCheck(output).parse()
    np.testing.assert_array_equal(actual.measurements[0].doses[:, 2, 0], [777, 888])
    np.testing.assert_allclose(actual.measurements[0].times, scan.times, atol=1e-15)
    np.testing.assert_array_equal(
        actual.file_meta["SRS MapCHECK"]["Data"][:, native["Columns"].index("T1")], [30, 31]
    )
    assert actual.file_meta["SRS MapCHECK"]["Corrections"]["Calibration(AP)"][0] == 1.234
    assert np.isfinite(actual.measurements[0].doses).all()


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_srs_movie_preserves_fractional_dose_edits_with_other_metadata_dtypes(tmp_path, dtype):
    beam = load(tmp_path, SRSMapCheck, movie_export())
    native = beam.file_meta["SRS MapCHECK"]
    native["Data"] = native["Data"].astype(dtype)
    original = native["Data"].copy()
    edited = np.array([123.123456789, -0.125])
    beam.measurements[0].doses[:, 2, 0] = edited
    output = tmp_path / "fractional.smcm"
    beam.save(output)
    actual = SRSMapCheck(output).parse()
    np.testing.assert_array_equal(actual.measurements[0].doses[:, 2, 0], edited)
    np.testing.assert_array_equal(
        actual.file_meta["SRS MapCHECK"]["Data"][:, native["Columns"].index("D1")], edited
    )
    np.testing.assert_array_equal(native["Data"], original)
    assert native["Data"].dtype == np.dtype(dtype)


def test_srs_retains_non_detector_coordinate_and_correction_cells(tmp_path):
    lines = movie_export().splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("X (mm)", "Y (mm)", "Background\t")):
            cells = line.split("\t")
            cells[1] = "auxiliary source cell"
            lines[index] = "\t".join(cells)
    beam = load(tmp_path, SRSMapCheck, "\n".join(lines))
    output = tmp_path / "all_cells.smcm"
    beam.save(output)
    native = SRSMapCheck(output).parse().file_meta["SRS MapCHECK"]
    assert native["Coordinate Rows"][0][0] == "auxiliary source cell"
    assert native["Coordinate Rows"][1][0] == "auxiliary source cell"
    assert native["Correction Rows"]["Background"][0] == "auxiliary source cell"


@pytest.mark.parametrize("value", [999.0, np.nan, np.inf])
def test_srs_movie_ignores_interpolated_pixel_edits_and_writes_detector_edits(tmp_path, value):
    beam = load(tmp_path, SRSMapCheck, movie_export())
    scan = beam.measurements[0]
    scan.doses[:, 1, 2] = value  # Interior interpolated site.
    scan.doses[:, 0, 1] = value  # Perimeter interpolated site.
    scan.doses[:, 2, 0] = [321, 654]  # Physical detector D1 is authoritative.
    output = tmp_path / "edited.smcm"
    beam.save(output)
    actual = SRSMapCheck(output).parse()
    native = actual.file_meta["SRS MapCHECK"]
    np.testing.assert_array_equal(native["Data"][:, native["Columns"].index("D1")], [321, 654])
    np.testing.assert_array_equal(actual.measurements[0].doses[:, 2, 0], [321, 654])
    # D1 does not contribute to the sole interior gap, so its original
    # interpolation returns despite the independent pixel edit.
    np.testing.assert_array_equal(actual.measurements[0].doses[:, 1, 2], [103.5, 203.5])
    assert np.isfinite(actual.measurements[0].doses).all()


@pytest.mark.parametrize(
    "kind,problem",
    [
        ("movie", "missing"),
        ("movie", "time"),
        ("movie", "counter"),
        ("measured", "table"),
        ("measured", "nan"),
        ("measured", "geometry"),
        ("measured", "header"),
        ("measured", "energy"),
    ],
)
def test_srs_preflight_preserves_existing_destination(tmp_path, kind, problem):
    beam = load(tmp_path, SRSMapCheck, movie_export() if kind == "movie" else measured_export())
    scan = beam.measurements[0]
    if problem == "missing":
        beam.file_meta["SRS MapCHECK"].pop("Data")
    elif problem == "time":
        scan.times = scan.times + 0.0001
    elif problem == "counter":
        beam.file_meta["SRS MapCHECK"]["Pulses"][0] = -1
    elif problem == "table":
        beam.file_meta["SRS MapCHECK"]["Tables"]["Raw Counts"] = np.zeros((1, 1))
    elif problem == "nan":
        scan.doses[0, 0] = np.nan
    elif problem == "geometry":
        scan.coordinates[..., 2] += 1
    elif problem == "energy":
        scan.energy = np.nan
    else:
        beam.file_meta["Header"]["File Name"] = "bad\nheader"
    output = tmp_path / "existing.dat"
    output.write_text("keep")
    with pytest.raises(ValueError):
        beam.save(output)
    assert output.read_text() == "keep"
