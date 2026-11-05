"""Native movie arrays, geometry dependencies and primary signal exports."""

import struct
import zlib

import numpy as np
import pytest

from beamio.common import AcquisitionType
from beamio.rfa import ArcCHECK, BeamDataFactory
from tests.support.arccheck_movie import acm, acml


def pair(tmp_path):
    (tmp_path / "movie.acm").write_bytes(acm())
    (tmp_path / "movie.acml").write_bytes(acml())
    return tmp_path / "movie.acm", tmp_path / "movie.acml"


def test_acm_native_frames_and_telemetry(tmp_path):
    path, _ = pair(tmp_path)
    beam = BeamDataFactory.from_file(path)
    scan = beam.measurements[0]
    assert scan.acquisition_type is AcquisitionType.TimeSeries
    assert scan.doses.shape == (3, 2, 4) and scan.coordinates.shape == (2, 4, 3)
    np.testing.assert_array_equal(scan.doses[0].ravel(), np.arange(8) - 5)
    np.testing.assert_allclose(scan.times, [3.649966, 3.699966, 3.749966], atol=1e-12)
    assert scan.metadata["DoseUnit"] == "counts" and scan.metadata["Cumulative"]
    native = beam.file_meta["ArcCHECK"]
    assert native["Auxiliary Data"].shape == (3, 10)
    assert not np.shares_memory(scan.doses, native["Auxiliary Data"])
    np.testing.assert_array_equal(scan.axes[0], [-5, 5])
    np.testing.assert_array_equal(scan.axes[1], [-15, -5, 5, 15])
    assert scan.energy == 6


def test_acm_primary_edits_and_times_survive_export(tmp_path):
    path, _ = pair(tmp_path)
    beam = ArcCHECK(path).parse()
    scan = beam.measurements[0]
    original_aux = beam.file_meta["ArcCHECK"]["Auxiliary Data"].copy()
    scan.doses[1, 0, 1] = 1234.25
    scan.times[1] += 0.001
    output = tmp_path / "edited.acm"
    beam.save(output)
    reread = ArcCHECK(output).parse()
    np.testing.assert_array_equal(reread.measurements[0].doses, scan.doses)
    np.testing.assert_allclose(reread.measurements[0].times, scan.times, rtol=0, atol=1e-12)
    native = reread.file_meta["ArcCHECK"]
    for i, name in enumerate(native["Auxiliary Columns"]):
        if name != "TIMETIC1":
            np.testing.assert_array_equal(native["Auxiliary Data"][:, i], original_aux[:, i])
    assert beam.file_meta["Sections"] == reread.file_meta["Sections"]
    assert "Data" not in native and "Raw Frames" not in native


def test_acm_float32_time_edits_are_not_multiplied_at_lower_precision(tmp_path):
    path, _ = pair(tmp_path)
    beam = ArcCHECK(path).parse()
    scan = beam.measurements[0]
    scan.times = scan.times.astype(np.float32)
    beam.save(tmp_path / "float32.acm")
    reread = ArcCHECK(tmp_path / "float32.acm").parse()
    np.testing.assert_allclose(reread.measurements[0].times, scan.times, rtol=0, atol=1e-12)


def test_acml_processed_frames_geometry_and_binary_roundtrip(tmp_path):
    _, path = pair(tmp_path)
    beam = BeamDataFactory.from_file(path)
    scan = beam.measurements[0]
    assert scan.doses.shape == (2, 2, 4) and scan.doses.dtype == np.dtype("float32")
    assert scan.metadata["DoseUnit"] == "cGy" and scan.energy is None
    np.testing.assert_array_equal(scan.doses[0].ravel(), np.arange(8) + 0.125)
    output = tmp_path / "copy.acml"
    beam.save(output)
    assert output.read_bytes() == path.read_bytes()
    assert (tmp_path / "copy.acm").read_bytes() == (tmp_path / "movie.acm").read_bytes()
    reread = ArcCHECK(output).parse()
    np.testing.assert_array_equal(reread.measurements[0].coordinates, scan.coordinates)


def test_acml_edits_use_processed_doses_and_float32_times(tmp_path):
    _, path = pair(tmp_path)
    beam = ArcCHECK(path).parse()
    scan = beam.measurements[0]
    scan.doses = scan.doses.astype(float)
    scan.doses[1, 0, 0] = 0.123456789
    scan.times[1] += 0.01
    beam.save(tmp_path / "edited.acml")
    reread = ArcCHECK(tmp_path / "edited.acml").parse()
    assert reread.measurements[0].doses[1, 0, 0] == np.float32(0.123456789)
    np.testing.assert_allclose(reread.measurements[0].times, scan.times, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(reread.file_meta["ArcCHECK"]["Auxiliary Data"]["STATUS1"], [4, 4])


def test_lite_reuses_existing_compatible_acm_without_overwriting_it(tmp_path):
    source, lite = pair(tmp_path)
    raw = ArcCHECK(source).parse()
    raw.measurements[0].doses[0, 0, 0] = 99
    raw.save(tmp_path / "copy.acm")
    before = (tmp_path / "copy.acm").read_bytes()
    ArcCHECK(lite).parse().save(tmp_path / "copy.acml")
    assert (tmp_path / "copy.acm").read_bytes() == before
    assert ArcCHECK(tmp_path / "copy.acml").parse().measurements


def test_lite_does_not_overwrite_incompatible_companion(tmp_path):
    _, lite = pair(tmp_path)
    beam = ArcCHECK(lite).parse()
    (tmp_path / "copy.acm").write_bytes(acm(serial="different"))
    target = tmp_path / "copy.acml"
    target.write_bytes(b"keep")
    with pytest.raises(ValueError, match="incompatible ACM companion"):
        beam.save(target)
    assert target.read_bytes() == b"keep"
    assert (tmp_path / "copy.acm").read_bytes() == acm(serial="different")


def test_lite_missing_geometry_dependency_is_explicit(tmp_path):
    path = tmp_path / "movie.acml"
    path.write_bytes(acml())
    with pytest.raises(FileNotFoundError, match="matching same-stem ACM"):
        BeamDataFactory.from_file(path)


@pytest.mark.parametrize("mismatch", ["serial", "count"])
def test_lite_mismatched_geometry_is_rejected(tmp_path, mismatch):
    _, path = pair(tmp_path)
    path.write_bytes(acml(serial="different") if mismatch == "serial" else acml(n=4))
    with pytest.raises(ValueError, match="different serials or detector counts"):
        ArcCHECK(path).parse()


@pytest.mark.parametrize("suffix", ["acm", "acml"])
@pytest.mark.parametrize(
    "mutation", ["nonfinite", "geometry", "frame_count", "units", "cross_format"]
)
def test_movie_preflight_preserves_existing_file(tmp_path, suffix, mutation):
    pair(tmp_path)
    beam = ArcCHECK(tmp_path / f"movie.{suffix}").parse()
    scan = beam.measurements[0]
    if mutation == "nonfinite":
        scan.doses[0, 0, 0] = np.nan
    elif mutation == "geometry":
        scan.coordinates[0, 0, 0] += 1
    elif mutation == "frame_count":
        scan.doses = scan.doses[:-1]
        scan.times = scan.times[:-1]
    elif mutation == "units":
        scan.metadata["DoseUnit"] = "Gy"
    output = tmp_path / ("existing.txt" if mutation == "cross_format" else f"existing.{suffix}")
    output.write_bytes(b"keep")
    with pytest.raises(ValueError):
        beam.save(output)
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("kind", ["checksum", "count", "width", "coordinates"])
def test_corrupt_acm_rejected(tmp_path, kind):
    raw = acm()
    if kind == "checksum":
        raw = raw.replace(b"Synthetic", b"Corrupted")
    else:
        payload = raw.rsplit(b"\r\n", 2)[0] + b"\r\n"
        if kind == "count":
            payload = payload.replace(b"Number of Updates:\t3", b"Number of Updates:\t4")
        elif kind == "width":
            payload = payload.replace(b"\t1000\t-5\t-4", b"\t1000\t-4")
        elif kind == "coordinates":
            payload = payload.replace(b"X(n)", b"Q(n)")
        raw = payload + f"{zlib.crc32(payload):08X}\r\n".encode()
    path = tmp_path / "movie.acm"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        ArcCHECK(path).parse()


@pytest.mark.parametrize("kind", ["truncated", "trailing", "header", "count", "nonfinite"])
def test_corrupt_lite_rejected(tmp_path, kind):
    _, path = pair(tmp_path)
    raw = bytearray(path.read_bytes())
    if kind == "truncated":
        del raw[-1:]
    elif kind == "trailing":
        raw.extend(b"x")
    elif kind == "header":
        raw[100:104] = b"oops"
    elif kind == "count":
        struct.pack_into("<I", raw, 1324, 3)
    elif kind == "nonfinite":
        struct.pack_into("<f", raw, len(raw) - 4, float("nan"))
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        ArcCHECK(path).parse()


@pytest.mark.parametrize("suffix,revision", [("acm", "Z"), ("acml", "Z")])
def test_unknown_movie_revisions_attempt_known_layout(tmp_path, suffix, revision):
    pair(tmp_path)
    path = tmp_path / f"movie.{suffix}"
    path.write_bytes(acm(revision=revision) if suffix == "acm" else acml(revision=revision))
    with pytest.warns(UserWarning, match="revision 'Z'"):
        assert ArcCHECK(path).parse().measurements


def test_movie_conversion_preserves_independent_signal_and_geometry(tmp_path):
    source, _ = pair(tmp_path)
    beam = ArcCHECK(source).parse()
    copied = beam.convert(ArcCHECK)
    copied.measurements[0].doses[0, 0, 0] = 123
    assert beam.measurements[0].doses[0, 0, 0] != 123
    copied.save(tmp_path / "converted.acm")
    assert ArcCHECK(tmp_path / "converted.acm").parse().measurements[0].doses[0, 0, 0] == 123
