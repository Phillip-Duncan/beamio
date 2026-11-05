"""ArcCHECK native tables, merge semantics, detection and edited exports."""

from copy import deepcopy

import numpy as np
import pytest

from beamio.common import AcquisitionType, MeasurementType, RadiationType
from beamio.rfa import ArcCHECK, BeamDataFactory, SRSMapCheck
from tests.support.arccheck import export, change


def load(tmp_path, data=None):
    path = tmp_path / "source.dat"
    path.write_bytes(export() if data is None else data)
    result = BeamDataFactory.from_file(path)
    assert isinstance(result, ArcCHECK)
    return result


def test_measured_uses_final_dose_and_keeps_all_native_tables(tmp_path):
    beam = load(tmp_path)
    assert set(vars(beam)) == {"file_path", "file_meta", "measurements"}
    m = beam.measurements[0]
    assert m.type is MeasurementType.DosePlane and m.acquisition_type is AcquisitionType.Static
    assert m.radiation_type is RadiationType.PHOTON and m.energy == 6
    assert m.metadata["DoseUnit"] == "cGy" and m.metadata["FluenceMode"] == "FFF"
    np.testing.assert_array_equal(m.axes[0], [5, 0, -5])
    np.testing.assert_array_equal(m.axes[1], [-10, -5, 0, 5, 10])
    native = beam.file_meta["ArcCHECK"]
    assert len(native["Tables"]) == 10 and native["Tables"]["Dose Interpolated"] is m.doses
    assert beam.file_meta["Header"]["Time"] == "1000"
    assert "Time:\t12:34:56" in beam.file_meta["Header"]["Date"]
    assert beam.file_meta["Measurement Corrections"]["Applied Out Of Field"] == "0"


@pytest.mark.parametrize("kind,rows", [("Measured File", 41), ("Merged File", 73)])
def test_native_shapes_and_selected_source_roundtrip(tmp_path, kind, rows):
    beam = load(tmp_path, export(kind=kind, rows=rows, cols=131))
    m = beam.measurements[0]
    source = "Dose Counts" if kind == "Merged File" else "Dose Interpolated"
    assert m.doses.shape == (rows, 131) and np.max(m.doses) > 0
    assert m.metadata["ArcCHECK"]["Source Table"] == source
    if kind == "Measured File":
        assert beam.file_meta["ArcCHECK"]["Tables"][source] is m.doses
    else:
        assert not np.shares_memory(beam.file_meta["ArcCHECK"]["Tables"][source], m.doses)
    if kind == "Merged File":
        assert beam.file_meta["Header"]["Dose per Count"] == "0"
        assert not np.any(beam.file_meta["ArcCHECK"]["Tables"]["Dose Interpolated"])
    original_tables = deepcopy(beam.file_meta["ArcCHECK"]["Tables"])
    m.doses[2, 2] = 123.456789
    output = tmp_path / "edited.txt"
    beam.save(output)
    reread = ArcCHECK(output).parse()
    if kind == "Merged File":
        recorded = ~beam.file_meta["ArcCHECK"]["Missing Mask"]
        np.testing.assert_array_equal(reread.measurements[0].doses[recorded], m.doses[recorded])
    else:
        np.testing.assert_array_equal(reread.measurements[0].doses, m.doses)
    for name, values in original_tables.items():
        if name != source:
            np.testing.assert_array_equal(reread.file_meta["ArcCHECK"]["Tables"][name], values)
    assert beam.file_meta["Header"] == reread.file_meta["Header"]
    if kind == "Merged File":
        assert [
            l for l in beam.file_meta["ArcCHECK"]["Header Lines"] if l.startswith("C:\\synthetic")
        ]
        original = [
            l for l in beam.file_meta["ArcCHECK"]["Header Lines"] if l.startswith("C:\\synthetic")
        ]
        assert original == [
            l for l in reread.file_meta["ArcCHECK"]["Header Lines"] if l.startswith("C:\\synthetic")
        ]


@pytest.mark.parametrize("orientation,y_shift", [(0, -80), (180, 80)])
def test_shifted_coordinates_are_used_without_reapplying_orientation(
    tmp_path, orientation, y_shift
):
    beam = load(tmp_path, export(y_shift=y_shift, orientation=orientation))
    np.testing.assert_array_equal(beam.measurements[0].axes[0], np.array([5, 0, -5]) + y_shift)
    assert beam.file_meta["Header"]["Orientation"] == str(orientation)


@pytest.mark.parametrize("model", ["1220-MR ArcCHECK-MR", "1220 ArcCHECK-MR", "1220MR ArcCHECK-MR"])
def test_mr_schema_uses_file_dimensions_with_validation_warning(tmp_path, model):
    with pytest.warns(UserWarning, match="model.*native samples"):
        beam = load(tmp_path, export(model=model, rows=7, cols=9))
    assert beam.measurements[0].doses.shape == (7, 9)
    beam.save(tmp_path / "mr.txt")
    with pytest.warns(UserWarning, match="model"):
        reread = ArcCHECK(tmp_path / "mr.txt").parse()
    np.testing.assert_array_equal(reread.measurements[0].doses, beam.measurements[0].doses)


@pytest.mark.parametrize(
    "energy,value,mode",
    [("6MV", 6, ""), ("10X", 10, ""), ("6 MV FFF", 6, "FFF"), ("10FFF", 10, "FFF")],
)
def test_energy_labels(tmp_path, energy, value, mode):
    beam = load(tmp_path, export(energy=energy))
    m = beam.measurements[0]
    assert m.energy == value and m.metadata.get("FluenceMode", "") == mode
    beam.save(tmp_path / "copy.txt")
    assert (
        ArcCHECK(tmp_path / "copy.txt").parse().file_meta["Measurement Corrections"]["Energy"]
        == energy
    )


def test_unknown_energy_is_preserved_without_guessing(tmp_path):
    with pytest.warns(UserWarning, match="energy label"):
        beam = load(tmp_path, export(energy="Co60"))
    assert beam.measurements[0].energy is None
    beam.save(tmp_path / "copy.txt")
    with pytest.warns(UserWarning, match="energy label"):
        assert (
            ArcCHECK(tmp_path / "copy.txt").parse().file_meta["Measurement Corrections"]["Energy"]
            == "Co60"
        )


def test_energy_and_existing_header_edits_are_exported(tmp_path):
    beam = load(tmp_path)
    beam.measurements[0].energy = 10
    beam.file_meta["Header"]["Cal File"] = "edited.cal"
    beam.save(tmp_path / "edited.txt")
    reread = ArcCHECK(tmp_path / "edited.txt").parse()
    assert reread.measurements[0].energy == 10
    assert reread.file_meta["Header"]["Cal File"] == "edited.cal"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
def test_native_encoding_and_metadata_retained(tmp_path, encoding):
    data = export(encoding=encoding).decode(encoding).replace("Keep this", "Caf\u00e9")
    # Build a correct encoded checksum after editing the header.
    import zlib

    payload = data.rsplit("\r\n", 2)[0] + "\r\n"
    raw = payload.encode(encoding)
    beam = load(tmp_path, raw + f"{zlib.crc32(raw):08X}\r\n".encode())
    beam.save(tmp_path / "encoded.txt")
    reread = ArcCHECK(tmp_path / "encoded.txt").parse()
    assert reread.file_meta["Header"]["Unknown Field"] == "Caf\u00e9"
    assert reread.file_meta["ArcCHECK"]["Encoding"] == encoding


@pytest.mark.parametrize(
    "old,new",
    [
        ("Rows:\t3", "Rows:\t4"),
        ("Cols:\t5", "Cols:\t6"),
        ("CAX X:\t3", "CAX X:\t6"),
        ("CAX Y:\t2", "CAX Y:\tnan"),
        ("0.25\t1.25", "nan\t1.25"),
        ("\tCOL\t1\t2\t3", "\tCOL\t1\t2\t2"),
        ("Unknown Field:\tKeep this", "Rows:\t3"),
    ],
)
def test_invalid_native_records_rejected(tmp_path, old, new):
    with pytest.raises(ValueError):
        load(tmp_path, change(export(), old, new))


def test_checksum_detects_changes_and_legacy_missing_footer_is_allowed(tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        load(tmp_path, export().replace(b"Keep this", b"corrupted"))
    assert load(tmp_path, export(checksum=False)).file_meta["ArcCHECK"]["Checksum"] is None


def test_unknown_revision_attempts_compatible_schema(tmp_path):
    with pytest.warns(UserWarning, match="revision 'Z'"):
        assert load(tmp_path, export(revision="Z")).measurements


@pytest.mark.parametrize(
    "mutation", ["nan", "geometry", "flags", "time", "radiation", "metadata", "units"]
)
def test_invalid_export_does_not_change_destination(tmp_path, mutation):
    beam = load(tmp_path)
    m = beam.measurements[0]
    if mutation == "nan":
        m.doses[0, 0] = np.nan
    elif mutation == "geometry":
        m.axes = (m.axes[0] + 1, m.axes[1])
    elif mutation == "flags":
        beam.file_meta["ArcCHECK"]["Tables"]["Data Flags"] = np.zeros((1, 1))
    elif mutation == "time":
        m.doses = m.doses[None]
        m.times = np.array([0])
        m.acquisition_type = AcquisitionType.TimeSeries
    elif mutation == "radiation":
        m.radiation_type = RadiationType.ELECTRON
    elif mutation == "metadata":
        beam.file_meta = {}
    elif mutation == "units":
        m.metadata["DoseUnit"] = "Gy"
    output = tmp_path / "existing.txt"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError):
        beam.save(output)
    assert output.read_bytes() == b"keep"


def test_detection_requires_real_preamble_and_distinguishes_srs(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, b"Notes about ArcCHECK\n1220 ArcCHECK\n")
    from tests.support.srsmapcheck import measured_export

    path = tmp_path / "srs.txt"
    path.write_text(measured_export())
    assert isinstance(BeamDataFactory.from_file(path), SRSMapCheck)


def test_native_conversion_isolated_and_merged_signal_is_writable(tmp_path):
    beam = load(tmp_path, export(kind="Merged File"))
    copied = beam.convert(ArcCHECK)
    copied.measurements[0].doses[0, 0] = 432.1
    assert not np.shares_memory(
        copied.file_meta["ArcCHECK"]["Tables"]["Dose Counts"], copied.measurements[0].doses
    )
    assert beam.measurements[0].doses[0, 0] != 432.1
    copied.save(tmp_path / "copy.txt")
    assert ArcCHECK(tmp_path / "copy.txt").parse().measurements[0].doses[0, 0] == 432.1
