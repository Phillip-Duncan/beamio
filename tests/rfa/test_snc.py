from copy import deepcopy

import numpy as np
import pytest

from beamio.common.measurement import MeasurementType, RadiationType
from beamio.rfa import BeamDataFactory, SNCSunScan
from tests.support.measurements import AXIAL_SCANS, assert_scan_equal
from tests.support.snc import SNC_TEXT, xml_export


def test_read_text_processed_table_and_metadata(tmp_path):
    path = tmp_path / "scan.SNCTXT"
    path.write_text(SNC_TEXT, encoding="utf-8-sig")
    beam = BeamDataFactory.from_file(path)
    assert isinstance(beam, SNCSunScan)
    assert len(beam.measurements) == 1
    scan = beam.measurements[0]
    assert scan.type == MeasurementType.CrosslineProfile
    assert scan.radiation_type == RadiationType.PHOTON
    assert scan.energy == 6
    assert scan.SSD == 1000
    assert scan.field_size == (100, 200)
    assert scan.wedge_angle == 0
    assert scan.startpoint == (5, 0, 10)
    assert scan.endpoint == (-5, 0, 10)
    np.testing.assert_array_equal(scan.positions, [5, 0, -5])
    np.testing.assert_array_equal(scan.doses, [55, 100, 65])
    assert scan.metadata["Summary Comments"] == ""
    assert scan.metadata["Custom Metadata"] == "preserved"
    assert scan.metadata["Action"] == "Normalize"
    assert scan.metadata["GantryAngle[deg]"] == "90"
    assert scan.metadata["CollimatorAngle[deg]"] == "12"
    beam.parse()
    assert len(beam.measurements) == 1


def test_text_preserves_sample_order_duplicates_and_nan_doses(tmp_path):
    text = SNC_TEXT.replace("\t0\t0\t1\t100", "\t0.5\t0\t1\tNaN")
    path = tmp_path / "scan.snctxt"
    path.write_text(text)
    scan = SNCSunScan(path).parse().measurements[0]
    np.testing.assert_array_equal(scan.positions, [5, 5, -5])
    np.testing.assert_allclose(scan.doses, [55, np.nan, 65], equal_nan=True)


def test_text_round_trip_preserves_measured_off_axis_motion(tmp_path):
    text = SNC_TEXT.replace("\t0\t0\t1\t100", "\t0\t0.002\t1.003\t100")
    path = tmp_path / "scan.snctxt"
    path.write_text(text)
    beam = SNCSunScan(path).parse()
    expected = [[5, 0, 10], [0, 0.02, 10.03], [-5, 0, 10]]
    np.testing.assert_allclose(beam.measurements[0].coordinates, expected)
    np.testing.assert_allclose(np.column_stack(beam.measurements[0].reconstruct_xyz()), expected)
    destination = tmp_path / "saved.snctxt"
    beam.save(destination)
    np.testing.assert_allclose(
        SNCSunScan(destination).parse().measurements[0].coordinates, expected
    )


def test_text_reads_reordered_columns_and_units(tmp_path):
    text = SNC_TEXT.replace(
        "X (cm)\tY (cm)\tZ (cm)\tRelative Dose (%)", "Relative Dose (%)\tZ (mm)\tY (cm)\tX (mm)"
    )
    for old, new in [
        ("0.5\t0\t1\t55", "55\t10\t0\t5"),
        ("0\t0\t1\t100", "100\t10\t0\t0"),
        ("-0.5\t0\t1\t65", "65\t10\t0\t-5"),
    ]:
        text = text.replace("\t" + old + "\n", "\t" + new + "\n")
    path = tmp_path / "scan.snctxt"
    path.write_text(text)
    scan = SNCSunScan(path).parse().measurements[0]
    np.testing.assert_array_equal(scan.positions, [5, 0, -5])
    np.testing.assert_array_equal(scan.doses, [55, 100, 65])
    assert scan.startpoint == (5, 0, 10)


@pytest.mark.parametrize(
    "before,after",
    [
        ("END SCAN\n", ""),
        ("END DOSE TABLE\n", ""),
        ("END RAW DATA\n", ""),
        ("File Scan Count\t1", "File Scan Count\t2"),
        ("Relative Dose (%)", "Absolute Dose (Gy)"),
        ("X (cm)", "X (inch)"),
        ("\t0.5\t0\t1\t55", "\tbad\t0\t1\t55"),
        ("\t0.5\t0\t1\t55", "\t0.5\t0\t1"),
        ("\t0.5\t0\t1\t55", "\tNaN\t0\t1\t55"),
        ("Crossline", "UnsupportedScan"),
        ("Photon", "UnknownBeam"),
    ],
)
def test_text_rejects_invalid_or_incomplete_data(tmp_path, before, after):
    path = tmp_path / "bad.snctxt"
    path.write_text(SNC_TEXT.replace(before, after))
    with pytest.raises(ValueError):
        SNCSunScan(path).parse()


@pytest.mark.parametrize("namespace", [False, True])
def test_xml_defaults_to_latest_layer(tmp_path, namespace):
    path = tmp_path / "scan.sncxml"
    path.write_text(xml_export(count=2, namespace=namespace), encoding="utf-8-sig")
    beam = BeamDataFactory.from_file(path)
    assert isinstance(beam, SNCSunScan)
    assert len(beam.measurements) == 2
    scan = beam.measurements[0]
    assert scan.type == MeasurementType.CrosslineProfile
    assert scan.radiation_type == RadiationType.PHOTON
    assert scan.field_size == (100, 200)
    assert scan.SSD == 1000
    np.testing.assert_array_equal(scan.positions, [5, 0, -5])
    np.testing.assert_array_equal(scan.doses, [55, 100, 65])
    assert scan.metadata["SNC XML"]["SelectedLayer"]["LayerNumber"] == "3"
    assert len(scan.metadata["SNC XML"]["Layers"]) == 2
    beam.parse()
    assert len(beam.measurements) == 2


@pytest.mark.parametrize(
    "selection,expected",
    [("raw", [0.1, 0.2, 0.3, 0.4]), (1, [0.1, 0.2, 0.3, 0.4]), (3, [55, 100, 65])],
)
def test_xml_selects_a_processing_layer(tmp_path, selection, expected):
    path = tmp_path / "scan.sncxml"
    path.write_text(xml_export())
    beam = SNCSunScan(path, layer=selection).parse()
    np.testing.assert_array_equal(beam.measurements[0].doses, expected)


def test_xml_missing_layer_is_explicit(tmp_path):
    path = tmp_path / "scan.sncxml"
    path.write_text(xml_export())
    with pytest.raises(ValueError, match="no layer matching"):
        SNCSunScan(path, layer=2).parse()


def test_xml_respects_field_size_unit(tmp_path):
    path = tmp_path / "scan.sncxml"
    path.write_text(xml_export(field_unit="Millimeter"))
    scan = SNCSunScan(path).parse().measurements[0]
    assert scan.field_size == (10, 20)
    assert scan.SSD == 1000
    np.testing.assert_array_equal(scan.positions, [5, 0, -5])


@pytest.mark.parametrize(
    "text",
    [
        "<unrelated />",
        "<snc_fileexport>",
        xml_export().replace('count="1"', 'count="2"'),
        xml_export(field_unit="Inch"),
        xml_export().replace('name="RelativeDose"', 'name="OtherDose"'),
    ],
)
def test_xml_rejects_invalid_data(tmp_path, text):
    path = tmp_path / "bad.sncxml"
    path.write_text(text)
    with pytest.raises(ValueError):
        SNCSunScan(path).parse()


@pytest.mark.parametrize("selection", [0, -1, True, "unknown"])
def test_invalid_layer_selection(selection):
    with pytest.raises(ValueError, match="SNC layer"):
        SNCSunScan("", layer=selection)


def test_text_does_not_claim_to_provide_xml_layers(tmp_path):
    path = tmp_path / "scan.snctxt"
    path.write_text(SNC_TEXT)
    with pytest.raises(ValueError, match="use XML"):
        SNCSunScan(path, layer="raw").parse()


@pytest.mark.parametrize(
    "scan_type,start,end,positions",
    AXIAL_SCANS
    + [
        (MeasurementType.DiagonalProfile, (6, 8, 50), (-6, -8, 50), [10, 0, -10]),
        (MeasurementType.DiagonalProfile, (-6, 8, 50), (6, -8, 50), [-10, 0, 10]),
    ],
)
def test_text_round_trip(tmp_path, measurement, scan_type, start, end, positions):
    measurement.type, measurement.startpoint, measurement.endpoint = scan_type, start, end
    measurement.positions = np.asarray(positions, dtype=float)
    measurement.radiation_type = RadiationType.ELECTRON
    beam = SNCSunScan("")
    beam.measurements = [measurement, deepcopy(measurement)]
    destination = tmp_path / "saved.snctxt"
    beam.save(destination)
    loaded = SNCSunScan(destination).parse()
    assert len(loaded.measurements) == 2
    for scan in loaded.measurements:
        assert_scan_equal(scan, measurement)
        np.testing.assert_allclose(scan.endpoint, end)
        np.testing.assert_allclose(
            scan.reconstruct_xyz(), measurement.reconstruct_xyz(), atol=1e-12
        )


def test_save_updates_metadata_from_edited_shared_fields(tmp_path):
    path = tmp_path / "input.snctxt"
    path.write_text(SNC_TEXT)
    beam = SNCSunScan(path).parse()
    scan = beam.measurements[0]
    scan.field_size, scan.SSD, scan.energy = (30, 50), 950, 10
    scan.radiation_type = RadiationType.ELECTRON
    destination = tmp_path / "saved.snctxt"
    beam.save(destination)
    loaded = SNCSunScan(destination).parse().measurements[0]
    assert_scan_equal(loaded, scan)
    assert loaded.metadata["Custom Metadata"] == "preserved"
    assert loaded.metadata["Action"] == "Normalize"
    assert scan.metadata["Summary FieldSize X (cm)"] == "10"


def test_save_infers_axial_endpoint_without_mutating_input(tmp_path, measurement):
    measurement.endpoint = None
    beam = SNCSunScan("")
    beam.measurements = [measurement]
    path = tmp_path / "scan.snctxt"
    beam.save(path)
    assert measurement.endpoint is None
    assert SNCSunScan(path).parse().measurements[0].endpoint == (0, 0, 20)


def test_empty_file_round_trip(tmp_path):
    path = tmp_path / "empty.snctxt"
    SNCSunScan("").save(path)
    assert SNCSunScan(path).parse().measurements == []


@pytest.mark.parametrize(
    "text",
    [
        "Tab-Delimited Scan Output\n",
        "Tab-Delimited Scan Output\nFILE HEADER\n",
        "Tab-Delimited Scan Output\nFILE HEADER\nFile Version\t6.0\n",
        "Tab-Delimited Scan Output\nFILE HEADER\nFile Scan Count\t0\n",
        "Tab-Delimited Scan Output\nFile Version\t6.0\nFile Scan Count\t0\n",
        SNC_TEXT.replace("File Scan Count\t1\t\n", ""),
    ],
)
def test_reject_incomplete_text_header(tmp_path, text):
    path = tmp_path / "incomplete.snctxt"
    path.write_text(text)
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        SNCSunScan(path).parse()


def test_reject_xml_without_declared_scan_count(tmp_path):
    path = tmp_path / "incomplete.sncxml"
    path.write_text(xml_export(count=0).replace('count="0"', ""))
    with pytest.raises(ValueError, match="scan count"):
        SNCSunScan(path).parse()


@pytest.mark.parametrize("source_kind", ["text", "xml"])
def test_reject_unknown_photon_rate_type(tmp_path, source_kind):
    path = tmp_path / ("scan.snctxt" if source_kind == "text" else "scan.sncxml")
    text = (
        SNC_TEXT.replace("Rate Type\tFF\t", "Rate Type\tUnknownMode\t")
        if source_kind == "text"
        else xml_export(rate_type="UnknownMode")
    )
    path.write_text(text)
    with pytest.raises(ValueError, match="photon rate type"):
        SNCSunScan(path).parse()


@pytest.mark.parametrize("suffix", [".sncxml", ".csv", ".txt"])
def test_reject_unsupported_output(tmp_path, suffix):
    with pytest.raises(ValueError, match="Unsupported output format"):
        SNCSunScan("").save(tmp_path / ("output" + suffix))


def test_invalid_output_does_not_overwrite_existing_file(tmp_path, measurement):
    beam = SNCSunScan("")
    measurement.doses = measurement.doses[:1]
    beam.measurements = [measurement]
    path = tmp_path / "existing.snctxt"
    path.write_text("existing contents")
    with pytest.raises(ValueError, match="matching one-dimensional"):
        beam.save(path)
    assert path.read_text() == "existing contents"
