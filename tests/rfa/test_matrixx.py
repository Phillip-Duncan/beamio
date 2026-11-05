"""IBA grid orientation, native preservation, editing and malformed input."""

from copy import deepcopy
import struct

import numpy as np
import pytest

from beamio.common import AcquisitionType, Measurement, MeasurementType, RadiationType
from beamio.rfa import BeamDataFactory, IBAMatriXX
from tests.support.matrixx import opd_export, opg_export, with_crc

KEY = "IBA MatriXX"


def load(tmp_path, data, name="source.dat"):
    path = tmp_path / name
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    result = BeamDataFactory.from_file(path)
    assert isinstance(result, IBAMatriXX)
    return result


def test_opg_scaling_and_primary_array_edits(tmp_path):
    beam = load(tmp_path, opg_export())
    assert set(vars(beam)) == {"file_path", "measurements", "file_meta"}
    m = beam.measurements[0]
    assert m.type is MeasurementType.DosePlane and m.acquisition_type is AcquisitionType.Static
    assert m.radiation_type is RadiationType.PHOTON and m.energy == 6
    assert m.SSD == 900 and m.field_size == (100, 80)
    np.testing.assert_array_equal(m.axes[0], [-20, 20])
    np.testing.assert_array_equal(m.axes[1], [-10, 0, 10])
    np.testing.assert_allclose(m.doses, [[0.02, 0.04, 0.06], [0.08, 0.10, 0.12]])
    np.testing.assert_array_equal(m.reconstruct_xyz()[2], 15)
    m.doses[0, 1] = 0.123456789
    output = tmp_path / "edited.opg"
    beam.save(output)
    reloaded = IBAMatriXX(output).parse()
    np.testing.assert_array_equal(reloaded.measurements[0].doses, m.doses)
    assert reloaded.file_meta[KEY]["Header"]["Custom Header"] == "retain this"
    assert reloaded.measurements[0].metadata[KEY]["Body"]["Custom Body"] == "keep this too"


@pytest.mark.parametrize("unit,factor", [("Gy", 1), ("cGy", 0.01), ("mGy", 0.001), ("%", 1)])
def test_opg_dose_units(tmp_path, unit, factor):
    m = load(tmp_path, opg_export(**{"Data Unit": unit})).measurements[0]
    np.testing.assert_allclose(m.doses, np.arange(1, 7).reshape(2, 3) * 20 * factor)
    assert m.metadata["DoseUnit"] == ("%" if unit == "%" else "Gy")


@pytest.mark.parametrize("plane", ["XY", "XZ", "YX", "YZ", "ZX", "ZY"])
def test_opg_orientations_and_multiple_bodies(tmp_path, plane):
    text = opg_export(Plane=plane, **{"Number of Bodies": "2"})
    text = text.replace("X[cm]", f"{plane[0]}[cm]").replace("Y[cm]", f"{plane[1]}[cm]")
    # Avoid label replacement collisions for YX/ZX.
    lines = text.splitlines()
    x_index = next(i for i, line in enumerate(lines) if "\t-1\t0\t1" in line)
    lines[x_index] = f"{plane[0]}[cm]\t-1\t0\t1"
    lines[x_index + 1] = f"{plane[1]}[cm]"
    text = "\n".join(lines)
    body = text[text.index("<asciibody>") : text.index("</asciibody>") + len("</asciibody>")]
    text = text.replace("</opimrtascii>", body.replace("1.5 cm", "2 cm") + "\n</opimrtascii>")
    beam = load(tmp_path, text)
    assert len(beam.measurements) == 2
    m = beam.measurements[1]
    xyz = m.reconstruct_xyz()
    np.testing.assert_array_equal(xyz["XYZ".index(plane[0])], [[-10, 0, 10], [-10, 0, 10]])
    normal = next(i for i in range(3) if "XYZ"[i] not in plane)
    np.testing.assert_array_equal(xyz[normal], 20)
    beam.save(tmp_path / "planes.opg")
    reread = IBAMatriXX(tmp_path / "planes.opg").parse()
    np.testing.assert_array_equal(reread.measurements[1].reconstruct_xyz(), xyz)


@pytest.mark.parametrize(
    "changes",
    [
        {"Data Factor": "0"},
        {"Data Factor": "nan"},
        {"Data Unit": "unknown"},
        {"Length Unit": "inches"},
        {"No. of Columns": "4"},
        {"No. of Rows": "3"},
        {"Number of Bodies": "2"},
        {"No. of Rows": "0"},
        {"Plane": "XX"},
        {"Separator": "[COMMA]"},
    ],
)
def test_opg_bad_headers(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, opg_export(**changes))


@pytest.mark.parametrize(
    "old,new",
    [
        ("Data Factor: 2\n", ""),
        ("10\t20\t30", "10\t20"),
        ("-1\t0\t1", "-1\t0\t0"),
        ("Y[cm]", "Z[cm]"),
        ("</asciibody>", ""),
        ("</opimrtascii>", "junk</opimrtascii>"),
    ],
)
def test_opg_malformed_grids(tmp_path, old, new):
    with pytest.raises(ValueError):
        load(tmp_path, opg_export().replace(old, new))


def test_opg_version_and_rotation_warnings(tmp_path):
    with pytest.warns(UserWarning, match="version 9"):
        load(tmp_path, opg_export(**{"File Version": "9"}))
    with pytest.warns(UserWarning, match="rotation metadata"):
        beam = load(tmp_path, opg_export(**{"In-phantom Rotation": "30 deg"}))
    assert beam.file_meta[KEY]["Header"]["In-phantom Rotation"] == "30 deg"


def test_opd_roundtrip_preserves_records_and_flags(tmp_path):
    pixels = np.array([[1, 2, 0x80000003], [4, 5, 6]], dtype=np.uint32)
    raw = opd_export([("Integral 0001", 1, 0, 2, pixels)])
    beam = load(tmp_path, raw)
    m = beam.measurements[0]
    assert m.energy == 6 and m.SSD == 900 and m.field_size == (100, 80)
    assert m.radiation_type is RadiationType.PHOTON
    np.testing.assert_array_equal(m.axes[0], [20, -20])
    np.testing.assert_allclose(m.doses, np.arange(1, 7).reshape(2, 3) * 0.002)
    image = beam.file_meta[KEY]["Images"][0]
    assert "Pixels" not in image
    assert image["PixelFlags"][0, 2]
    output = tmp_path / "copy.opd"
    beam.save(output)
    assert output.read_bytes() == raw
    m.doses[0, 2] = 0.02
    m.energy = 10
    m.SSD = 950
    m.field_size = (80, 60)
    beam.save(output)
    reloaded = IBAMatriXX(output).parse()
    np.testing.assert_array_equal(reloaded.measurements[0].doses, m.doses)
    assert reloaded.measurements[0].energy == 10
    assert reloaded.measurements[0].SSD == 950
    assert reloaded.measurements[0].field_size == (80, 60)
    assert reloaded.file_meta[KEY]["Images"][0]["PixelFlags"][0, 2]
    assert (99, b"opaque record") in reloaded.file_meta[KEY]["Images"][0]["Records"]


def test_opd_snapshots_form_timeseries_and_integrals_stay_separate(tmp_path):
    pixels = np.arange(1, 7).reshape(2, 3)
    raw = opd_export(
        [
            ("Snap 0001", 1, 10, 2, pixels),
            ("Snap 0002", 1, 12.5, 4, pixels + 1),
            ("Integral 0003", 1, 15, 2, pixels * 3),
            ("Snap 0004", 1, 20, 2, pixels),
        ]
    )
    beam = load(tmp_path, raw)
    assert [m.doses.shape for m in beam.measurements] == [(2, 2, 3), (2, 3), (1, 2, 3)]
    movie = beam.measurements[0]
    assert movie.acquisition_type is AcquisitionType.TimeSeries
    np.testing.assert_allclose(movie.times, [0, 2.5], atol=1e-6)
    np.testing.assert_allclose(movie.doses[1], (pixels + 1) * 0.004)
    movie.doses[1, 0, 0] = 0.128
    movie.times[1] = 4.25
    output = tmp_path / "movie.opd"
    beam.save(output)
    reloaded = IBAMatriXX(output).parse()
    np.testing.assert_array_equal(reloaded.measurements[0].doses, movie.doses)
    np.testing.assert_allclose(reloaded.measurements[0].times, movie.times, atol=1e-6)


def test_opd_geometry_changes_split_snapshot_groups(tmp_path):
    beam = load(
        tmp_path,
        opd_export(
            [
                ("Snap 0001", 1, 0, 1, np.ones((2, 3))),
                ("Snap 0002", 1, 1, 1, np.ones((3, 3))),
            ]
        ),
    )
    assert len(beam.measurements) == 2


def test_opd_non_dose_images_retained_without_dose_interpretation(tmp_path):
    pixels = np.array([[0, 0x800186A0], [2, 200000]], dtype=np.uint32)
    raw = opd_export([("Gamma analysis", 3, 0, 0.01, pixels)])
    with pytest.warns(UserWarning, match="no dose images"):
        beam = load(tmp_path, raw)
    assert not beam.measurements
    np.testing.assert_array_equal(beam.file_meta[KEY]["Images"][0]["Pixels"], pixels)
    beam.save(tmp_path / "analysis.opd")
    assert (tmp_path / "analysis.opd").read_bytes() == raw


def test_opd_unknown_version_attempts_read(tmp_path):
    with pytest.warns(UserWarning, match="version 8"):
        beam = load(tmp_path, opd_export(version=8))
    assert len(beam.measurements) == 1


@pytest.mark.parametrize(
    "corruption", ["crc", "count", "truncated_record", "truncated_grid", "pixel_layout"]
)
def test_opd_rejects_corrupt_records(tmp_path, corruption):
    data = bytearray(opd_export()[:-4])
    if corruption == "crc":
        raw = with_crc(data)[:-1] + b"x"
    else:
        if corruption == "count":
            struct.pack_into("<H", data, 16, 2)
        elif corruption == "truncated_record":
            struct.pack_into("<I", data, 20, len(data))
        elif corruption == "truncated_grid":
            del data[-4:]
        elif corruption == "pixel_layout":
            grid = data.index(struct.pack("<HI", 6, 151)) + 6
            struct.pack_into("<I", data, grid + 87, 8)
        raw = with_crc(data)
    with pytest.raises(ValueError):
        load(tmp_path, raw)


@pytest.mark.parametrize(
    "change",
    ["negative", "overflow", "nan", "shape", "geometry", "missing", "duplicate", "radiation"],
)
def test_opd_preflight_preserves_existing_file(tmp_path, change):
    beam = load(tmp_path, opd_export())
    m = beam.measurements[0]
    if change == "negative":
        m.doses[0, 0] = -1
    elif change == "overflow":
        m.doses[0, 0] = 1e10
    elif change == "nan":
        m.doses[0, 0] = np.nan
    elif change == "shape":
        m.doses = m.doses[:-1]
    elif change == "geometry":
        m.axes = (m.axes[0] + 1, m.axes[1])
    elif change == "missing":
        beam.measurements = []
    elif change == "duplicate":
        beam.measurements.append(deepcopy(m))
    elif change == "radiation":
        m.radiation_type = RadiationType.ELECTRON
    output = tmp_path / "existing.opd"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError):
        beam.save(output)
    assert output.read_bytes() == b"keep"


def test_opd_quantization_is_native_precision(tmp_path):
    beam = load(tmp_path, opd_export())
    beam.measurements[0].doses[0, 0] = 0.0031
    beam.save(tmp_path / "rounded.opd")
    assert IBAMatriXX(tmp_path / "rounded.opd").parse().measurements[0].doses[0, 0] == 0.004


@pytest.mark.parametrize("size", [32, 39])
def test_opd_and_opg_grid_sizes_are_file_defined(tmp_path, size):
    pixels = np.arange(size * size).reshape(size, size)
    beam = load(tmp_path, opd_export([("Integral 0001", 1, 0, 1, pixels)]))
    assert beam.measurements[0].doses.shape == (size, size)
    output = tmp_path / "plane.opg"
    beam.save(output)
    reread = IBAMatriXX(output).parse().measurements[0]
    np.testing.assert_allclose(reread.doses, pixels * 0.001)
    np.testing.assert_array_equal(reread.axes, beam.measurements[0].axes)


@pytest.mark.parametrize("encoding", ["cp1252", "utf-8-sig", "utf-8"])
def test_opg_preserves_encoding_and_non_ascii_metadata(tmp_path, encoding):
    text = opg_export().replace("Synthetic export", "Caf\u00e9 export")
    beam = load(tmp_path, text.encode(encoding))
    beam.save(tmp_path / "encoded.opg")
    encoded = (tmp_path / "encoded.opg").read_bytes()
    assert encoded.startswith(b"\xef\xbb\xbf") == (encoding == "utf-8-sig")
    assert "Caf\u00e9 export".encode(encoding).lstrip(b"\xef\xbb\xbf") in encoded
    assert (
        "Caf\u00e9 export"
        in IBAMatriXX(tmp_path / "encoded.opg").parse().file_meta[KEY]["Header"]["Operators Note"]
    )


def test_opd_sub_count_negative_edits_are_rejected(tmp_path):
    beam = load(tmp_path, opd_export())
    beam.measurements[0].doses[0, 0] = -1e-9
    with pytest.raises(ValueError, match="nonnegative"):
        beam.save(tmp_path / "negative.opd")


def test_opd_elapsed_time_origin_cannot_silently_change(tmp_path):
    beam = load(tmp_path, opd_export([("Snap 0001", 1, 0, 1, np.ones((2, 3)))]))
    beam.measurements[0].times[0] = 1
    with pytest.raises(ValueError, match="start at zero"):
        beam.save(tmp_path / "shifted.opd")


def test_opd_native_acquisition_type_cannot_silently_change(tmp_path):
    beam = load(tmp_path, opd_export([("Snap 0001", 1, 0, 1, np.ones((2, 3)))]))
    m = beam.measurements[0]
    m.doses = m.doses[0]
    m.times = None
    m.acquisition_type = AcquisitionType.Static
    with pytest.raises(ValueError, match="acquisition type"):
        beam.save(tmp_path / "changed.opd")


def test_opg_body_header_edit_cannot_inject_extra_rows(tmp_path):
    beam = load(tmp_path, opg_export())
    beam.measurements[0].metadata[KEY]["Body"]["Custom Body"] = "value\nother line"
    with pytest.raises(ValueError, match="newlines"):
        beam.save(tmp_path / "bad.opg")


def test_opw_renamed_dependencies_roundtrip_and_primary_edits(tmp_path):
    (tmp_path / "workspace_1.opd").write_bytes(opd_export())
    workspace = (
        "[FileVersion]\nVersion=1\n[DataSet1]\nPath=old_1.opd\n[General]\nComment=retain me\n"
    )
    with pytest.warns(UserWarning, match="using 'workspace_1.opd'"):
        beam = load(tmp_path, workspace, "workspace.opw")
    m = beam.measurements[0]
    m.doses[0, 0] = 0.1
    beam.save(tmp_path / "edited.opw")
    reread = IBAMatriXX(tmp_path / "edited.opw").parse()
    np.testing.assert_array_equal(reread.measurements[0].doses, m.doses)
    assert reread.file_meta[KEY]["Sections"]["General"]["Comment"] == "retain me"
    assert reread.file_meta[KEY]["Sections"]["DataSet1"]["Path"] == "edited_1.opd"


def test_opw_missing_reference_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing dataset"):
        load(tmp_path, "[FileVersion]\nVersion=1\n[DataSet1]\nPath=absent.opd\n")


@pytest.mark.parametrize("path_style", ["relative-slashes", "relative-backslashes", "absolute"])
def test_opw_stored_path_wins_over_neighbour_with_same_name(tmp_path, path_style):
    nested = tmp_path / "nested"
    nested.mkdir()
    reference = nested / "reference.opd"
    reference.write_bytes(opd_export([("Integral 0001", 1, 0, 2, np.full((2, 3), 100))]))
    (tmp_path / "reference.opd").write_bytes(opd_export())
    if path_style == "absolute":
        stored = str(reference)
    elif path_style == "relative-backslashes":
        stored = "nested\\reference.opd"
    else:
        stored = "nested/reference.opd"
    beam = load(tmp_path, f"[FileVersion]\nVersion=1\n[DataSet1]\nPath={stored}\n", "workspace.opw")
    np.testing.assert_array_equal(beam.measurements[0].doses, 0.2)


@pytest.mark.parametrize("stored", ["missing\\reference.opd", "Z:\\missing\\reference.opd"])
def test_opw_relocated_basename_fallback_warns(tmp_path, stored):
    (tmp_path / "reference.opd").write_bytes(opd_export())
    with pytest.warns(UserWarning, match="using 'reference.opd'"):
        beam = load(
            tmp_path, f"[FileVersion]\nVersion=1\n[DataSet1]\nPath={stored}\n", "workspace.opw"
        )
    assert beam.measurements[0].doses[0, 0] == 0.002


@pytest.mark.parametrize("reference_kind", ["self", "cycle", "opg"])
def test_opw_rejects_non_opd_dependencies_before_dispatch(tmp_path, reference_kind):
    if reference_kind == "self":
        stored = "workspace.opw"
    elif reference_kind == "cycle":
        stored = "other.opw"
        (tmp_path / stored).write_text("[FileVersion]\nVersion=1\n[DataSet1]\nPath=workspace.opw\n")
    else:
        stored = "reference.opd"
        (tmp_path / stored).write_text(opg_export())
    # An existing explicit reference must not be replaced with a valid fallback.
    (tmp_path / "workspace_1.opd").write_bytes(opd_export())
    with pytest.raises(ValueError, match="DataSet1.*must be an OPD file"):
        load(tmp_path, f"[FileVersion]\nVersion=1\n[DataSet1]\nPath={stored}\n", "workspace.opw")


@pytest.mark.parametrize(
    "radiation,energy",
    [("", "6 MV"), ("", "6 MeV"), ("Photons", "1.25 MeV"), ("Electrons", "9 MeV")],
)
@pytest.mark.parametrize("edit_energy", [False, True])
def test_opg_preserves_source_energy_units(tmp_path, radiation, energy, edit_energy):
    beam = load(tmp_path, opg_export(**{"Radiation Type": radiation, "Energy": energy}))
    measurement = beam.measurements[0]
    measurement.doses[0, 0] *= 1.1
    if edit_energy:
        measurement.energy += 1.5
    output = tmp_path / "energy.opg"
    beam.save(output)
    reloaded = IBAMatriXX(output).parse()
    expected = f"{measurement.energy:.17g} {energy.split()[1]}"
    assert reloaded.file_meta[KEY]["Header"]["Energy"] == expected
    assert reloaded.measurements[0].energy == measurement.energy


def test_opg_explicit_radiation_change_sets_appropriate_energy_unit(tmp_path):
    beam = load(tmp_path, opg_export())
    beam.measurements[0].radiation_type = RadiationType.ELECTRON
    beam.save(tmp_path / "electron.opg")
    reloaded = IBAMatriXX(tmp_path / "electron.opg").parse()
    assert reloaded.file_meta[KEY]["Header"]["Energy"] == "6 MeV"
    assert reloaded.measurements[0].radiation_type is RadiationType.ELECTRON


def test_opg_unknown_energy_unit_rejects_before_touching_destination(tmp_path):
    beam = load(tmp_path, opg_export(**{"Radiation Type": "", "Energy": "6"}))
    output = tmp_path / "existing.opg"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="requires a retained source energy unit"):
        beam.save(output)
    assert output.read_bytes() == b"keep"


def test_opw_preflights_all_children(tmp_path):
    for number in (1, 2):
        (tmp_path / f"source_{number}.opd").write_bytes(opd_export())
    beam = load(
        tmp_path,
        "[FileVersion]\nVersion=1\n[DataSet1]\nPath=source_1.opd\n"
        "[DataSet2]\nPath=source_2.opd\n",
        "source.opw",
    )
    beam.measurements[-1].doses[0, 0] = -1
    with pytest.raises(ValueError):
        beam.save(tmp_path / "edited.opw")
    assert not list(tmp_path.glob("edited*"))


def test_generic_static_plane_can_write_opg(tmp_path):
    m = Measurement()
    m.type = MeasurementType.DosePlane
    m.doses = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m.axes = (np.array([-1.0, 1.0]), np.array([-2.0, 0.0, 2.0]))
    m.metadata["DoseUnit"] = "Gy"
    beam = IBAMatriXX("")
    beam.measurements = [m]
    beam.save(tmp_path / "generic.opg")
    np.testing.assert_array_equal(
        IBAMatriXX(tmp_path / "generic.opg").parse().measurements[0].doses, m.doses
    )
    with pytest.raises(ValueError, match="native OPD metadata"):
        beam.save(tmp_path / "generic.opd")


def test_movie_to_opg_requires_explicit_frame_selection(tmp_path):
    beam = load(tmp_path, opd_export([("Snap 0001", 1, 0, 1, np.ones((2, 3)))]))
    with pytest.raises(ValueError, match="static dose planes"):
        beam.save(tmp_path / "movie.opg")


def test_opg_requires_explicit_units_for_other_readers_signals(tmp_path):
    beam = load(tmp_path, opg_export())
    del beam.measurements[0].metadata["DoseUnit"]
    with pytest.raises(ValueError, match="explicit DoseUnit"):
        beam.save(tmp_path / "unscaled.opg")


def test_opg_unknown_version_is_preserved_on_export(tmp_path):
    with pytest.warns(UserWarning, match="version 9"):
        beam = load(tmp_path, opg_export(**{"File Version": "9"}))
    beam.save(tmp_path / "future.opg")
    with pytest.warns(UserWarning, match="version 9"):
        reread = IBAMatriXX(tmp_path / "future.opg").parse()
    assert reread.file_meta[KEY]["Header"]["File Version"] == "9"


def test_opd_undecoded_radiation_remains_native_and_roundtrips(tmp_path):
    data = bytearray(opd_export()[:-4])
    setup = data.index(struct.pack("<HI", 3, 86)) + 6
    struct.pack_into("<H", data, setup + 41, 123)
    raw = with_crc(data)
    beam = load(tmp_path, raw)
    assert beam.measurements[0].radiation_type is None
    beam.save(tmp_path / "unknown.opd")
    assert (tmp_path / "unknown.opd").read_bytes() == raw


def test_native_conversion_isolated_and_writable(tmp_path):
    beam = load(tmp_path, opd_export())
    converted = beam.convert(IBAMatriXX)
    converted.measurements[0].doses[0, 0] = 0.1
    assert beam.measurements[0].doses[0, 0] == 0.002
    converted.save(tmp_path / "converted.opd")
    assert IBAMatriXX(tmp_path / "converted.opd").parse().measurements[0].doses[0, 0] == 0.1


def test_matrixx_conversion_drops_incompatible_measurements(tmp_path):
    beam = load(tmp_path, opd_export())
    profile = Measurement()
    profile.type = MeasurementType.CrosslineProfile
    profile.doses = np.array([1.0, 2.0, 3.0])
    profile.positions = np.array([-1.0, 0.0, 1.0])
    beam.measurements.append(profile)
    with pytest.warns(UserWarning, match="type=CrosslineProfile"):
        converted = beam.convert(IBAMatriXX)
    assert len(converted.measurements) == 1
