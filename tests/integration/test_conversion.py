from copy import deepcopy

import numpy as np
import pytest

from beamio.common import AcquisitionType, MeasurementType
from beamio.rfa import IBAAccept6, ICProfiler, PTWMCC, SNCSunScan, SRSMapCheck
from beamio.tps.raystation import Raystation
from tests.support.measurements import array_measurement, assert_scan_equal
from tests.support.icprofiler import prm_export
from tests.support.ptw import mcc_export, mcc_scan
from tests.support.snc import SNC_TEXT, xml_export
from tests.support.srsmapcheck import DOSE, measured_export, movie_export


@pytest.mark.parametrize(
    "target,suffix", [(IBAAccept6, ".asc"), (Raystation, ".csv"), (SNCSunScan, ".snctxt")]
)
def test_srsmapcheck_explicit_static_row_converts_with_local_coordinates(tmp_path, target, suffix):
    path = tmp_path / "map.txt"
    path.write_text(measured_export())
    source = SRSMapCheck(path).parse()
    plane = source.measurements[0]
    # The caller explicitly selects a 1D row from the shared plane model.
    row = deepcopy(plane)
    row.type = MeasurementType.CrosslineProfile
    row.positions = plane.axes[1].copy()
    row.doses = plane.doses[1].copy()
    row.coordinates = plane.coordinates[1].copy()
    row.startpoint, row.endpoint = tuple(row.coordinates[0]), tuple(row.coordinates[-1])
    row.validate()
    source.measurements = [row]
    converted = source.convert(target)
    converted.file_meta["Header"]["File Name"] = "changed"
    assert source.file_meta["Header"]["File Name"] == "Synthetic export"
    destination = tmp_path / ("row" + suffix)
    converted.save(destination)
    actual = target(destination).parse().measurements[0]
    assert actual.type == MeasurementType.CrosslineProfile
    assert actual.energy == 6
    tolerance = 0.11 if target is IBAAccept6 else 1e-9
    np.testing.assert_allclose(actual.positions, row.positions, atol=tolerance)
    np.testing.assert_array_equal(actual.doses, DOSE[1])
    np.testing.assert_allclose(actual.startpoint, row.startpoint, atol=tolerance)


def test_srsmapcheck_movie_requires_explicit_frame_and_row_for_curve_export(tmp_path):
    path = tmp_path / "movie.smcm"
    path.write_text(movie_export())
    source = SRSMapCheck(path).parse()
    series = source.measurements[0]
    row = deepcopy(series)
    row.type = MeasurementType.CrosslineProfile
    row.acquisition_type = AcquisitionType.Static
    row.times = None
    row.positions = series.axes[1].copy()
    row.doses = series.doses[-1, 1].copy()
    row.coordinates = series.coordinates[1].copy()
    row.startpoint, row.endpoint = tuple(row.coordinates[0]), tuple(row.coordinates[-1])
    row.validate()
    source.measurements = [row]
    converted = source.convert(SNCSunScan)
    destination = tmp_path / "row.snctxt"
    converted.save(destination)
    actual = SNCSunScan(destination).parse().measurements[0]
    np.testing.assert_array_equal(actual.doses, series.doses[-1, 1])
    np.testing.assert_array_equal(actual.coordinates, row.coordinates)


@pytest.mark.parametrize(
    "target,suffix", [(IBAAccept6, ".asc"), (SNCSunScan, ".snctxt"), (Raystation, ".csv")]
)
@pytest.mark.parametrize(
    "kind,temporal",
    [
        (MeasurementType.CrosslineProfile, True),
        (MeasurementType.DosePlane, False),
        (MeasurementType.DosePlane, True),
        (MeasurementType.DoseVolume, False),
        (MeasurementType.DoseVolume, True),
    ],
)
def test_curve_writers_reject_unsupported_layouts_before_changing_output(
    tmp_path, target, suffix, kind, temporal
):
    beam = target("")
    scan = array_measurement(kind, temporal)
    scan.validate()
    beam.measurements = [scan]
    destination = tmp_path / ("existing" + suffix)
    destination.write_text("keep this file")
    with pytest.raises(ValueError, match="only static 1D"):
        beam.save(destination)
    assert destination.read_text() == "keep this file"


@pytest.mark.parametrize(
    "target,suffix", [(IBAAccept6, ".asc"), (SNCSunScan, ".snctxt"), (Raystation, ".csv")]
)
@pytest.mark.parametrize("invalid", ["type", "shape", "times", "coordinates"])
def test_writers_validate_supported_measurements_before_replacing_output(
    tmp_path, measurement, target, suffix, invalid
):
    beam = target("")
    broken = deepcopy(measurement)
    if invalid == "type":
        broken.type = None
    elif invalid == "shape":
        broken.doses = broken.doses[:1]
    elif invalid == "times":
        broken.times = np.array([0.1])
    else:
        broken.coordinates = np.column_stack(broken.reconstruct_xyz())
        broken.coordinates[1, 2] += 1
    # A valid first measurement must not cause a partial write if the next fails.
    beam.measurements = [measurement, broken]
    destination = tmp_path / ("existing" + suffix)
    destination.write_text("keep this file")
    with pytest.raises(ValueError):
        beam.save(destination)
    assert destination.read_text() == "keep this file"


@pytest.mark.parametrize("temporal", [False, True])
def test_conversion_preserves_supported_grid_geometry_and_time_without_aliasing(temporal):
    source = SNCSunScan("")
    scan = array_measurement(MeasurementType.DosePlane, temporal)
    scan.coordinates = np.stack(scan.reconstruct_xyz(), axis=-1)
    source.measurements = [scan]
    converted = source.convert(SRSMapCheck)
    actual = converted.measurements[0]
    actual.validate()
    assert actual.type is MeasurementType.DosePlane
    assert actual.acquisition_type is scan.acquisition_type
    np.testing.assert_array_equal(actual.doses, scan.doses)
    np.testing.assert_array_equal(actual.coordinates, scan.coordinates)
    for a, b in zip(actual.axes, scan.axes):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
    if temporal:
        np.testing.assert_array_equal(actual.times, scan.times)
        assert not np.shares_memory(actual.times, scan.times)
    actual.doses[...] = -123
    actual.coordinates[...] = -456
    assert not np.any(scan.doses == -123)
    assert not np.any(scan.coordinates == -456)


@pytest.mark.parametrize(
    "target,suffix", [(IBAAccept6, ".asc"), (Raystation, ".csv"), (SNCSunScan, ".snctxt")]
)
def test_icprofiler_conversion_preserves_four_profiles(tmp_path, target, suffix):
    source_path = tmp_path / "source.prm"
    source_path.write_text(prm_export())
    source = ICProfiler(source_path).parse()
    # The caller selects static final profiles using the shared arrays.
    for measurement in source.measurements:
        measurement.doses = measurement.doses[-1].copy()
        measurement.acquisition_type = AcquisitionType.Static
        measurement.times = None
        measurement.validate()
    output_path = tmp_path / ("converted" + suffix)
    converted = source.convert(target)
    converted.save(output_path)
    actual = target(output_path).parse()
    assert len(actual.measurements) == 4
    for expected, scan in zip(source.measurements, actual.measurements):
        assert scan.type == expected.type
        assert scan.radiation_type == expected.radiation_type
        assert scan.energy == expected.energy
        assert scan.SSD == expected.SSD
        assert scan.field_size == expected.field_size
        assert scan.wedge_angle == expected.wedge_angle
        np.testing.assert_allclose(scan.positions, expected.positions, rtol=1e-5)
        np.testing.assert_allclose(scan.doses, expected.doses)
        np.testing.assert_allclose(scan.startpoint, expected.startpoint)
        if scan.type == MeasurementType.DiagonalProfile or target is not Raystation:
            np.testing.assert_allclose(
                np.column_stack(scan.reconstruct_xyz()), expected.coordinates, atol=1e-3
            )
        if target is not IBAAccept6:
            assert scan.metadata["FluenceMode"] == "FFF"
    converted.file_meta["IC Profiler"]["Calibration"][0] = 100
    converted.file_meta["Machine Data"]["Energy"] = "15 MV"
    assert source.file_meta["IC Profiler"]["Calibration"][0] == 2
    assert source.file_meta["Machine Data"]["Energy"] == "6 MV"


@pytest.mark.parametrize(
    "target,suffix", [(IBAAccept6, ".asc"), (Raystation, ".csv"), (SNCSunScan, ".snctxt")]
)
@pytest.mark.parametrize("curve", ["PDD", "CROSSPLANE_PROFILE", "INPLANE_PROFILE"])
@pytest.mark.parametrize("modality,beam_filter", [("X", "FFF"), ("EL", "FF")])
def test_mcc_conversion_preserves_geometry_and_beam_fields(
    tmp_path, target, suffix, curve, modality, beam_filter
):
    path = tmp_path / "source.mcc"
    path.write_text(
        mcc_export(
            mcc_scan(
                {
                    "SCAN_CURVETYPE": curve,
                    "MODALITY": modality,
                    "FILTER": beam_filter,
                    # Use coordinates compatible with IBA's 0.2 mm header rounding.
                    "SCAN_OFFAXIS_CROSSPLANE": "2.0",
                }
            )
        )
    )
    source = PTWMCC(path).parse()
    expected = source.measurements[0]
    converted = source.convert(target)
    converted.measurements[0].metadata["PTW Reference"][0] = 999
    assert expected.metadata["PTW Reference"][0] == 2
    destination = tmp_path / ("converted" + suffix)
    converted.save(destination)
    actual = target(destination).parse().measurements[0]
    assert_scan_equal(actual, expected)
    if target is not Raystation:
        np.testing.assert_allclose(
            np.column_stack(actual.reconstruct_xyz()), expected.coordinates, atol=1e-12
        )
    # RayStation omits a zero wedge angle; its reader leaves that field unset.
    assert (actual.wedge_angle or 0) == expected.wedge_angle
    if modality == "X" and target is not IBAAccept6:
        assert actual.metadata["FluenceMode"] == "FFF"


@pytest.mark.parametrize("curve", ["PDD", "CROSSPLANE_PROFILE", "INPLANE_PROFILE"])
@pytest.mark.parametrize("offsets", [(-110, -110), (25, -15), (0, 0), (None, None)])
def test_mcc_raystation_conversion_preserves_field_edges(tmp_path, curve, offsets):
    path = tmp_path / "source.mcc"
    path.write_text(
        mcc_export(
            mcc_scan(
                {
                    "SCAN_CURVETYPE": curve,
                    "COLL_OFFSET_CROSSPLANE": offsets[0],
                    "COLL_OFFSET_INPLANE": offsets[1],
                }
            )
        )
    )
    source = PTWMCC(path).parse()
    expected = source.measurements[0]
    destination = tmp_path / "converted.csv"
    source.convert(Raystation).save(destination)
    actual = Raystation(destination).parse().measurements[0]
    assert_scan_equal(actual, expected)
    x, y = (value or 0 for value in offsets)
    corners = [float(value) for value in actual.metadata["Fieldsize[mm]"].split(";")]
    np.testing.assert_array_equal(corners, [x - 50, y - 100, x + 50, y + 100])
    # Field offsets must not translate the exported sample positions.
    np.testing.assert_array_equal(actual.positions, [-10, 0, 10])


def test_conversion_is_independent_and_exports(tmp_path, iba_text):
    source = tmp_path / "source.asc"
    source.write_text(iba_text, encoding="utf-8")
    beam = IBAAccept6(source).parse()
    converted = beam.convert(Raystation)
    converted.measurements[0].doses[0] = 123
    converted.measurements[0].metadata["CUSTOM"] = "changed"
    converted.file_meta["SYS"] = "changed"
    assert beam.measurements[0].doses[0] == 40
    assert beam.measurements[0].metadata["CUSTOM"] == "preserved"
    assert beam.file_meta["SYS"] == "BDS 0"
    destination = tmp_path / "converted.csv"
    converted.save(destination)
    assert_scan_equal(Raystation(destination).parse().measurements[0], converted.measurements[0])


@pytest.mark.parametrize("target,suffix", [(IBAAccept6, ".asc"), (Raystation, ".csv")])
@pytest.mark.parametrize(
    "name,rows",
    [
        ("Crossline", [(0.5, 0, 1), (0, 0, 1), (-0.5, 0, 1)]),
        ("Inline", [(0, 0.5, 1), (0, 0, 1), (0, -0.5, 1)]),
        ("Depth Scan", [(0, 0, 2), (0, 0, 1), (0, 0, 0)]),
        ("Profile at Angle", [(0.5, 0.5, 1), (0, 0, 1), (-0.5, -0.5, 1)]),
        ("Profile at Angle", [(-0.5, 0.5, 1), (0, 0, 1), (0.5, -0.5, 1)]),
    ],
)
def test_snc_conversion_preserves_profiles(tmp_path, target, suffix, name, rows):
    text = SNC_TEXT.replace("Crossline", name)
    for old, xyz, dose in zip(
        ("0.5\t0\t1\t55", "0\t0\t1\t100", "-0.5\t0\t1\t65"), rows, (55, 100, 65)
    ):
        new = "\t".join(str(value) for value in (*xyz, dose))
        text = text.replace("\t" + old + "\n", "\t" + new + "\n")
    path = tmp_path / "input.snctxt"
    path.write_text(text)
    source = SNCSunScan(path).parse()
    output = tmp_path / ("converted" + suffix)
    source.convert(target).save(output)
    expected = source.measurements[0]
    actual = target(output).parse().measurements[0]
    assert actual.type == expected.type
    assert actual.radiation_type == expected.radiation_type
    assert actual.field_size == expected.field_size
    assert actual.SSD == expected.SSD
    assert actual.energy == expected.energy
    source_order, output_order = np.argsort(expected.positions), np.argsort(actual.positions)
    np.testing.assert_allclose(
        actual.positions[output_order], expected.positions[source_order], rtol=1e-6
    )
    np.testing.assert_allclose(actual.doses[output_order], expected.doses[source_order])
    if actual.type == MeasurementType.DiagonalProfile:
        np.testing.assert_allclose(
            np.column_stack(actual.reconstruct_xyz())[output_order],
            expected.coordinates[source_order],
            atol=1e-5,
        )


def test_xml_to_text_preserves_latest_layer_and_copy_isolation(tmp_path):
    path = tmp_path / "source.sncxml"
    path.write_text(xml_export())
    source = SNCSunScan(path).parse()
    converted = source.convert(SNCSunScan)
    converted.measurements[0].metadata["SNC XML"]["SelectedLayer"]["Details"] = "changed"
    assert (
        source.measurements[0].metadata["SNC XML"]["SelectedLayer"]["Details"] == "synthetic layer"
    )
    output = tmp_path / "converted.snctxt"
    converted.save(output)
    actual = SNCSunScan(output).parse().measurements[0]
    assert_scan_equal(actual, source.measurements[0])
    np.testing.assert_array_equal(actual.coordinates, source.measurements[0].coordinates)


@pytest.mark.parametrize("reverse", [False, True])
def test_snc_off_axis_diagonal_iba_round_trip(tmp_path, measurement, reverse):
    xyz = np.array([[-2, -11, 50], [4, -3, 50], [10, 5, 50]], dtype=float)
    positions, doses = np.array([-10, 0, 10], dtype=float), np.array([50, 100, 60], dtype=float)
    if reverse:
        xyz, positions, doses = xyz[::-1], positions[::-1], doses[::-1]
    measurement.type = MeasurementType.DiagonalProfile
    measurement.positions, measurement.doses = positions, doses
    measurement.coordinates = xyz.copy()
    measurement.startpoint, measurement.endpoint = tuple(xyz[0]), tuple(xyz[-1])
    beam = SNCSunScan("")
    beam.measurements = [measurement]
    snc_path = tmp_path / "input.snctxt"
    beam.save(snc_path)
    output = tmp_path / "converted.asc"
    SNCSunScan(snc_path).parse().convert(IBAAccept6).save(output)
    actual = IBAAccept6(output).parse().measurements[0]
    order = np.argsort(positions)
    np.testing.assert_allclose(actual.positions, positions[order], atol=1e-12)
    np.testing.assert_array_equal(actual.doses, doses[order])
    np.testing.assert_allclose(np.column_stack(actual.reconstruct_xyz()), xyz[order], atol=1e-12)


@pytest.mark.parametrize("source_kind", ["text", "xml"])
@pytest.mark.parametrize(
    "rate_type,mode",
    [
        ("FF", "Standard"),
        ("FFF", "FFF"),
        ("FlatteningFilterFree", "FFF"),
        ("SRS", "SRS"),
    ],
)
def test_snc_raystation_round_trip_preserves_fluence_mode(tmp_path, source_kind, rate_type, mode):
    path = tmp_path / ("input.snctxt" if source_kind == "text" else "input.sncxml")
    text = (
        SNC_TEXT.replace("Rate Type\tFF\t", f"Rate Type\t{rate_type}\t")
        if source_kind == "text"
        else xml_export(rate_type=rate_type)
    )
    path.write_text(text)
    source = SNCSunScan(path).parse()
    assert source.measurements[0].metadata["FluenceMode"] == mode
    output = tmp_path / "converted.csv"
    source.convert(Raystation).save(output)
    assert f"FluenceMode:; {mode}" in output.read_text()
    csv = Raystation(output).parse()
    destination = tmp_path / "returned.snctxt"
    csv.convert(SNCSunScan).save(destination)
    reloaded = SNCSunScan(destination).parse().measurements[0]
    assert reloaded.metadata["FluenceMode"] == mode
    assert reloaded.metadata["Rate Type"] == ("FF" if mode == "Standard" else mode)
