import numpy as np
import pytest

from beamio.common.measurement import MeasurementType, RadiationType
from beamio.rfa import BeamDataFactory, PTWMCC
from tests.support.ptw import mcc_export, mcc_scan


def load(tmp_path, text):
    path = tmp_path / "scan.mcc"
    path.write_text(text, encoding="utf-8")
    return PTWMCC(path).parse()


@pytest.mark.parametrize(
    "curve,kind,xyz",
    [
        ("PDD", MeasurementType.DepthDose, [[2.5, -5, -10], [2.5, -5, 0], [2.5, -5, 10]]),
        (
            "CROSSPLANE_PROFILE",
            MeasurementType.CrosslineProfile,
            [[-10, -5, 100], [0, -5, 100], [10, -5, 100]],
        ),
        (
            "INPLANE_PROFILE",
            MeasurementType.InlineProfile,
            [[2.5, -10, 100], [2.5, 0, 100], [2.5, 10, 100]],
        ),
    ],
)
def test_read_geometry_and_metadata(tmp_path, curve, kind, xyz):
    metadata = {"SCAN_CURVETYPE": curve, "SCAN_DEPTH": None if curve == "PDD" else "100"}
    # Field offsets describe jaw position, not a translation of the sample coordinates.
    metadata.update(COLL_OFFSET_CROSSPLANE="-110", COLL_OFFSET_INPLANE="-110")
    beam = load(tmp_path, mcc_export(mcc_scan(metadata)))
    m = beam.measurements[0]
    assert m.type == kind
    assert m.radiation_type == RadiationType.PHOTON
    assert (m.energy, m.SSD, m.wedge_angle, m.field_size) == (6, 900, 0, (100, 200))
    assert m.startpoint == tuple(xyz[0])
    assert m.endpoint == tuple(xyz[-1])
    np.testing.assert_array_equal(m.positions, [-10, 0, 10])
    np.testing.assert_allclose(m.doses, [0.0055, 0.01, 0.0065])
    np.testing.assert_array_equal(m.coordinates, xyz)
    np.testing.assert_array_equal(np.column_stack(m.reconstruct_xyz()), xyz)
    np.testing.assert_array_equal(m.metadata["PTW Reference"], [2, 2.1, 2.2])
    assert m.metadata["COMMENT"] == "Synthetic=value"
    assert m.metadata["MEAS_UNIT"] == "A.U."
    assert m.metadata["PTW Scan Number"] == 1
    assert beam.file_meta == {"FORMAT": "CC-Export V1.9"}


def test_preserve_order_duplicates_nonfinite_signals_and_optional_reference(tmp_path):
    rows = ["10\t1e-9", "0\tnan", "0\tinf", "-10\t-2e-9"]
    m = load(tmp_path, mcc_export(mcc_scan({"SCAN_DIRECTION": "NEGATIVE"}, rows))).measurements[0]
    np.testing.assert_array_equal(m.positions, [10, 0, 0, -10])
    np.testing.assert_array_equal(m.doses, [1e-9, np.nan, np.inf, -2e-9])
    assert "PTW Reference" not in m.metadata
    assert m.startpoint == (10, -5, 100)
    assert m.endpoint == (-10, -5, 100)


def test_scan_metadata_does_not_leak_and_reparse_is_idempotent(tmp_path):
    beam = load(
        tmp_path,
        mcc_export(
            mcc_scan({"FILTER": "FFF"}),
            mcc_scan(
                {
                    "MODALITY": "EL",
                    "ENERGY": None,
                    "SSD": None,
                    "WEDGE_ANGLE": None,
                    "FILTER": None,
                    "SCAN_OFFAXIS_CROSSPLANE": None,
                    "SCAN_OFFAXIS_INPLANE": None,
                },
                number=7,
            ),
        ),
    )
    assert beam.parse() is beam
    assert len(beam.measurements) == 2
    first, second = beam.measurements
    assert first.metadata["FluenceMode"] == "FFF"
    assert second.radiation_type == RadiationType.ELECTRON
    assert second.energy is None and second.SSD is None and second.wedge_angle is None
    assert "FluenceMode" not in second.metadata
    assert second.startpoint == (-10, 0, 100)
    assert second.metadata["PTW Scan Number"] == 7


@pytest.mark.parametrize("beam_filter,mode", [("FF", "Standard"), ("FFF", "FFF"), ("SRS", "SRS")])
def test_photon_filter(tmp_path, beam_filter, mode):
    m = load(tmp_path, mcc_export(mcc_scan({"FILTER": beam_filter}))).measurements[0]
    assert m.metadata["FluenceMode"] == mode


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
def test_factory_detects_content_with_supported_text_encodings(tmp_path, encoding):
    path = tmp_path / "measurements.txt"
    path.write_bytes(mcc_export(mcc_scan({"COMMENT": "München µ"})).encode(encoding))
    beam = BeamDataFactory.from_file(path)
    assert isinstance(beam, PTWMCC)
    assert beam.measurements[0].metadata["COMMENT"] == "München µ"


@pytest.mark.parametrize(
    "comment",
    ["Tab-Delimited Scan Output", "Imported # Measurement number, %FSZ preserved"],
)
@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_mcc_opening_marker_takes_priority_over_comment_signatures(tmp_path, comment, encoding):
    path = tmp_path / "source.mcc"
    path.write_bytes(("\n\t\n" + mcc_export(mcc_scan({"COMMENT": comment}))).encode(encoding))
    beam = BeamDataFactory.from_file(path)
    assert isinstance(beam, PTWMCC)
    assert beam.measurements[0].metadata["COMMENT"] == comment


@pytest.mark.parametrize(
    "metadata,match",
    [
        ({"SCAN_CURVETYPE": "UNDEFINED"}, "SCAN_CURVETYPE"),
        ({"MODALITY": "UNKNOWN"}, "MODALITY"),
        ({"FILTER": "UNKNOWN"}, "FILTER"),
        ({"SCAN_DEPTH": None}, "require SCAN_DEPTH"),
        ({"ENERGY": "invalid"}, "ENERGY"),
        ({"SSD": "nan"}, "SSD must be finite"),
        ({"COLL_OFFSET_CROSSPLANE": "nan"}, "COLL_OFFSET_CROSSPLANE must be finite"),
        ({"COLL_OFFSET_INPLANE": "invalid"}, "Invalid MCC COLL_OFFSET_INPLANE"),
        ({"SCAN_OFFAXIS_INPLANE": "inf"}, "must be finite"),
        ({"DETECTOR": "STARCHECK"}, "detector-array"),
        ({"DETECTOR": "OCTAVIUS_1500"}, "detector-array"),
        ({"SCAN_DEVICE": "STARCHECKMAXI"}, "detector-array"),
        ({"SCAN_DIAGONAL": "FIRST_DIAGONAL"}, "diagonal"),
        ({"SCAN_ANGLE": "45"}, "angled"),
        ({"INPLANE_AXIS": "X"}, "axis mapping"),
        ({"CROSSPLANE_AXIS": "Y"}, "axis mapping"),
    ],
)
def test_reject_unsupported_or_invalid_metadata(tmp_path, metadata, match):
    with pytest.raises(ValueError, match=match):
        load(tmp_path, mcc_export(mcc_scan(metadata)))


@pytest.mark.parametrize(
    "rows,match",
    [
        ([], "at least one"),
        (["0 1", "1 2 3"], "consistently"),
        (["0"], "columns"),
        (["0 1 2 3"], "columns"),
        (["invalid 1"], "numeric"),
        (["nan 1"], "positions must be finite"),
        (["inf 1"], "positions must be finite"),
        (["0 1 #123"], "detector-array channel"),
    ],
)
def test_reject_invalid_samples(tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        load(tmp_path, mcc_export(mcc_scan(rows=rows)))


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("BEGIN_SCAN_DATA", "BROKEN", "BEGIN_SCAN_DATA"),
        ("END_SCAN_DATA", "", "Incomplete"),
        ("\tEND_SCAN  1", "", "missing END_SCAN"),
        ("\tEND_SCAN  1", "\tEND_SCAN  2", "Mismatched"),
        ("\t\tEND_DATA", "", "Expected END_DATA"),
        ("\t\tBEGIN_DATA", "\t\tBEGIN_DATA\n\t\tEND_DATA\n\t\tBEGIN_DATA", "duplicate"),
        ("\t\tENERGY=6.00", "\t\tENERGY=6.00\n\t\tENERGY=10.00", "duplicate"),
        ("\t\tBEGIN_DATA", "\tBEGIN_SCAN  2\n\t\tBEGIN_DATA", "Nested"),
    ],
)
def test_reject_malformed_structure_with_line_context(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match) as error:
        load(tmp_path, mcc_export().replace(old, new))
    if match != "Incomplete":
        assert "MCC line " in str(error.value)


@pytest.mark.parametrize(
    "text,match",
    [
        ("", "Incomplete"),
        (mcc_export() + "junk\n", "after END_SCAN_DATA"),
        (mcc_export(mcc_scan(), mcc_scan()), "Duplicate MCC scan"),
        ("BEGIN_SCAN_DATA\nEND_SCAN_DATA", "no scans"),
        ("BEGIN_SCAN_DATA\nBEGIN_SCAN 1\nEND_SCAN 1\nEND_SCAN_DATA", "missing BEGIN_DATA"),
    ],
)
def test_reject_incomplete_or_ambiguous_export(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load(tmp_path, text)


def test_mcc_save_is_explicitly_unsupported_and_does_not_touch_destination(tmp_path):
    destination = tmp_path / "existing.mcc"
    destination.write_text("original")
    with pytest.raises(NotImplementedError, match="MCC writing"):
        PTWMCC("").save(destination)
    assert destination.read_text() == "original"
