import gc
import weakref
import warnings

import beamio.rfa.srsmapcheck as srsmapcheck_module

import numpy as np
import pytest

from beamio.common import AcquisitionType, MeasurementType, RadiationType
from beamio.rfa import BeamDataFactory, SRSMapCheck
from tests.support.srsmapcheck import DETECTOR_XY, DOSE, X, Y, measured_export, movie_export, table


def read_export(tmp_path, text=None, *, encoding="utf-8"):
    path = tmp_path / "measurement.dat"
    path.write_bytes((measured_export() if text is None else text).encode(encoding))
    return SRSMapCheck(path).parse()


def test_final_dose_is_one_shared_static_plane_without_recalibration(tmp_path):
    beam = read_export(tmp_path)
    assert len(beam.measurements) == 1
    plane = beam.measurements[0]
    assert plane.type is MeasurementType.DosePlane
    assert plane.acquisition_type is AcquisitionType.Static
    assert plane.times is None
    np.testing.assert_array_equal(plane.axes[1], X)
    np.testing.assert_array_equal(plane.axes[0], Y)
    np.testing.assert_array_equal(plane.doses, DOSE)
    assert "SRS MapCHECK" not in beam.file_meta
    assert plane.radiation_type is RadiationType.PHOTON
    assert plane.energy == 6
    assert plane.metadata["FluenceMode"] == "FFF"
    assert plane.SSD is None and plane.field_size is None and plane.wedge_angle is None
    assert plane.coordinates.shape == (3, 5, 3)
    np.testing.assert_array_equal(plane.coordinates[..., 0], np.broadcast_to(X, (3, 5)))
    np.testing.assert_array_equal(plane.coordinates[..., 1], np.broadcast_to(Y[:, None], (3, 5)))
    np.testing.assert_array_equal(plane.coordinates[..., 2], 0)
    np.testing.assert_array_equal(np.stack(plane.reconstruct_xyz(), axis=-1), plane.coordinates)
    plane.validate()


def test_headers_preserve_sections_units_and_literal_free_text(tmp_path):
    beam = read_export(tmp_path)
    assert beam.file_meta["Header"]["Date"] == "01/02/26"
    assert beam.file_meta["Header"]["Time"] == "12:34:56"
    assert beam.file_meta["Dose Calibration"]["Temperature"] == "22\tC"
    assert beam.file_meta["Measurement Info"]["Temperature"] == "23"
    assert beam.file_meta["Data Corrections Applied"]["Temperature"] == "1"
    assert beam.file_meta["Dose Calibration"]["Comments"] == "Keep this label:\tliterally"
    assert beam.file_meta["Device Setup"]["StereoPHAN Angle"] == "270"
    assert beam.file_meta["Device Setup"]["Shift From Iso X"] == "2\tmm"
    assert beam.file_meta["Checksum"] == "A1B2C3"
    plane = beam.measurements[0]
    assert beam.file_meta["Header"]["File Name"] == "Synthetic export"


def test_raw_movie_exposes_plane_time_series_and_acquisition_metadata(tmp_path):
    beam = read_export(tmp_path, movie_export())
    assert len(beam.measurements) == 1
    plane = beam.measurements[0]
    assert plane.type is MeasurementType.DosePlane
    assert plane.acquisition_type is AcquisitionType.TimeSeries
    assert plane.doses.shape == (2, 3, 5)
    assert plane.coordinates.shape == (3, 5, 3)
    assert plane.metadata["Dose Scale"] == "raw counts"
    assert plane.metadata["Cumulative"] is True
    assert beam.file_meta["SRS MapCHECK"]["Detector IDs"] == tuple(f"D{i}" for i in range(1, 9))
    np.testing.assert_array_equal(
        beam.file_meta["SRS MapCHECK"]["Detector Coordinates"], DETECTOR_XY
    )
    np.testing.assert_array_equal(beam.file_meta["SRS MapCHECK"]["Pulses"], [0, 4])
    np.testing.assert_array_equal(beam.file_meta["SRS MapCHECK"]["Updates"], [11, 12])
    np.testing.assert_array_equal(beam.file_meta["SRS MapCHECK"]["Statuses"], [0, 0])
    np.testing.assert_allclose(plane.times, [0.1, 0.15])
    np.testing.assert_array_equal(
        beam.file_meta["SRS MapCHECK"]["Corrections"]["Calibration(AP)"], 2
    )
    np.testing.assert_array_equal(
        beam.file_meta["SRS MapCHECK"]["Corrections"]["Calibration(PA)"], 3
    )
    np.testing.assert_array_equal(
        beam.file_meta["SRS MapCHECK"]["Corrections"]["Background"], -0.02
    )
    plane.validate()


@pytest.mark.parametrize("frame,expected", [(-1, 200), (0, 100), (-2, 100), (1, 200)])
def test_movie_frames_are_selected_with_numpy_indexing(tmp_path, frame, expected):
    beam = read_export(tmp_path, movie_export())
    plane = beam.measurements[0]
    selected = plane.doses[frame]
    assert plane.acquisition_type is AcquisitionType.TimeSeries
    assert selected.shape == (3, 5)
    assert np.isfinite(selected).all()
    for detector, (x, y) in enumerate(DETECTOR_XY):
        row, col = np.flatnonzero(Y == y)[0], np.flatnonzero(X == x)[0]
        assert selected[row, col] == expected + detector
    assert plane.times.shape == (2,)


def test_cumulative_movie_is_interpolated_after_detector_accumulation(tmp_path):
    # Two nonnegative detector increments make different interpolation
    # directions preferable. Interpolating the increments and summing would
    # give 10 at the missing centre; interpolating their cumulative total gives
    # 55. The movie contains cumulative detector totals, so 55 is required.
    first = np.zeros(len(DETECTOR_XY))
    increments = np.zeros_like(first)
    first[[1, 6]] = 10  # Below/above the missing centre.
    first[4] = 100  # Right neighbour.
    first[5] = 100  # A measured corner sets the frame peak.
    increments[6] = 100
    cumulative = (first, first + increments)
    lines = movie_export().splitlines()
    frame = 0
    for index, line in enumerate(lines):
        if line.startswith("Data:"):
            cells = line.split("\t")
            cells[8:16] = map(str, cumulative[frame])
            lines[index] = "\t".join(cells)
            frame += 1
    movie = read_export(tmp_path, "\n".join(lines)).measurements[0]
    np.testing.assert_array_equal(movie.doses[:, 1, 2], [10, 55])
    assert movie.metadata["Cumulative"] is True
    for detector, (x, y) in enumerate(DETECTOR_XY):
        row, column = np.flatnonzero(Y == y)[0], np.flatnonzero(X == x)[0]
        assert movie.doses[-1, row, column] == (first + increments)[detector]


@pytest.mark.parametrize("factory", [False, True])
@pytest.mark.parametrize("kind", ["measured", "movie"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
def test_encodings_and_content_detection(tmp_path, factory, kind, encoding):
    path = tmp_path / "export.unrelated"
    text = movie_export() if kind == "movie" else measured_export()
    text = "\n\n" + text.replace("Synthetic export", "Synthetic caf\u00e9 export")
    path.write_bytes(text.replace("\n", "\r\n").encode(encoding))
    beam = BeamDataFactory.from_file(path) if factory else SRSMapCheck(path).parse()
    assert isinstance(beam, SRSMapCheck)
    assert beam.file_meta["Header"]["File Name"] == "Synthetic caf\u00e9 export"


@pytest.mark.parametrize("make_export", [measured_export, movie_export])
def test_unknown_revision_warns_and_attempts_compatible_layout(tmp_path, make_export):
    text = make_export().replace("File Revision:\tA", "File Revision:\tB")
    with pytest.warns(UserWarning, match="revision 'B' is unvalidated") as caught:
        beam = read_export(tmp_path, text)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert beam.file_meta["Header"]["File Revision"] == "B"


def test_validated_revision_is_quiet(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_export(tmp_path)
        read_export(tmp_path, movie_export())


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("Measured File", "Planned File", "file kind"),
        ("1179 SRS MapCHECK", "Other Device", "Model"),
        ("File Revision:\tA", "File Revision:\t", "File Revision"),
        ("Row Counts in 2D:\t3", "Row Counts in 2D:\t0", "Row Counts"),
        ("Col Counts in 2D:\t5", "Col Counts in 2D:\t4", "row width"),
        ("Col Counts in 2D:\t5", "Col Counts in 2D:\tNaN", "finite"),
        ("Beam Energy:\t6\tFFF", "Beam Energy:\t-6\tFFF", "positive"),
        ("Dose Interpolated", "Missing Dose Table", "Dose Interpolated"),
    ],
)
def test_rejects_invalid_measured_headers(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        read_export(tmp_path, measured_export().replace(old, new))


def test_missing_beam_energy_remains_unknown(tmp_path):
    beam = read_export(
        tmp_path, measured_export().replace("Beam Energy:\t6\tFFF", "Beam Energy:\t")
    )
    assert beam.measurements[0].energy is None
    assert beam.measurements[0].radiation_type is None


@pytest.mark.parametrize("name", ["Raw Counts", "Dose Interpolated"])
def test_rejects_truncated_tables(tmp_path, name):
    text = measured_export().split(name + "\nYcm")[0] + name + "\nYcm\tROW\n0\t1\t2\n"
    with pytest.raises(ValueError, match="incomplete"):
        read_export(tmp_path, text)


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda text: text.replace("A1B2C3", table("Dose Interpolated", DOSE)),
            "Duplicate",
        ),
        (lambda text: text.replace("\tCOL\t1\t2\t3\t4\t5", "\tCOL\t1\t2\t2\t4\t5"), "identifiers"),
        (lambda text: text.replace("\tXcm\t-0.35\t-0.175", "\tXcm\t-0.35\t-0.35"), "monotonic"),
        (lambda text: text.replace("0.25\t1.25", "nan\t1.25"), "finite"),
        (lambda text: text.replace("0.25\t1.25", "0.25\t"), "Invalid|row width"),
        (
            lambda text: text.replace("Dose Interpolated\nYcm\tROW", "Dose Interpolated\nBad\tROW"),
            "heading",
        ),
    ],
)
def test_rejects_malformed_measured_tables(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        read_export(tmp_path, change(measured_export()))


def test_rejects_geometry_shift_in_one_table(tmp_path):
    text = measured_export().replace(
        table("Dose Interpolated", DOSE), table("Dose Interpolated", DOSE, x=X + 1)
    )
    with pytest.raises(ValueError, match="geometry differs"):
        read_export(tmp_path, text)


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("X (mm)", "X (cm)", "headings"),
        ("D8\tEIB", "D7\tEIB", "duplicate TYPE"),
        ("Data:\t12\t150", "Data:\t13\t150", "missing or duplicate updates"),
        ("Data:\t12\t150\t4", "Data:\t12\t150\t-1", "PULSES"),
        ("Data:\t12\t150\t4", "Data:\t12\t150\t3", "Total Pulses"),
        ("Data:\t12\t150", "Data:\t12\t149", "Total Time"),
        ("Total Time:\t150\tms", "Total Time:\t150\ts", "ms units"),
        ("Update Count:\t2", "Update Count:\t3", "Update Count"),
        ("200\t201\t202", "200\t201", "row width"),
        ("200\t201\t202", "200\tnan\t202", "finite"),
        ("200\t201\t202", "200\ttext\t202", "numeric"),
        ("Col Counts in 2D:\t5", "Col Counts in 2D:\t4", "detector identifiers"),
        ("D8\tEIB", "D9\tEIB", "detector identifiers"),
        ("Background\t", "UnknownRow\t", "Unexpected"),
    ],
)
def test_rejects_invalid_raw_movie(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        read_export(tmp_path, movie_export().replace(old, new))


def test_truncated_movie_is_rejected(tmp_path):
    text = "\n".join(
        line for line in movie_export().splitlines() if not line.startswith("Data:\t12")
    )
    with pytest.raises(ValueError, match="Update Count"):
        read_export(tmp_path, text)


def test_movie_detector_order_follows_labels_and_coordinates_together(tmp_path):
    lines = movie_export().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("X (mm)"))
    for index in range(start, len(lines) - 1):
        cells = lines[index].split("\t")
        cells[8:16] = reversed(cells[8:16])
        lines[index] = "\t".join(cells)
    beam = read_export(tmp_path, "\n".join(lines))
    assert beam.file_meta["SRS MapCHECK"]["Detector IDs"] == tuple(f"D{i}" for i in range(8, 0, -1))
    np.testing.assert_array_equal(
        beam.file_meta["SRS MapCHECK"]["Detector Coordinates"], DETECTOR_XY[::-1]
    )
    reference = read_export(tmp_path, movie_export())
    np.testing.assert_array_equal(beam.measurements[0].doses, reference.measurements[0].doses)


def test_movie_frames_are_directly_indexable_arrays_preserving_measured_counts(tmp_path):
    beam = read_export(tmp_path, movie_export())
    assert isinstance(beam.measurements[0].doses, np.ndarray)
    assert beam.measurements[0].doses.shape == (2, 3, 5)
    for detector, (x, y) in enumerate(DETECTOR_XY):
        row, column = np.flatnonzero(Y == y)[0], np.flatnonzero(X == x)[0]
        np.testing.assert_array_equal(
            beam.measurements[0].doses[:, row, column],
            [100 + detector, 200 + detector],
        )
    assert beam.measurements[0].doses[-1].shape == (3, 5)
    assert beam.measurements[0].doses[0:1].shape == (1, 3, 5)
    assert beam.measurements[0].doses[5:9].shape == (0, 3, 5)
    assert len(beam.measurements[0].doses[::-1]) == len(beam.measurements[0].times[::-1])
    beam.measurements[0].doses[0, :, :] = 999
    np.testing.assert_array_equal(beam.measurements[0].doses[1, 2, ::2], [200, 201, 202])


def test_movie_frames_fill_interior_and_bracketed_perimeter_gaps(tmp_path):
    beam = read_export(tmp_path, movie_export())
    # The sole unmeasured interior site is surrounded by D2, D4, D5, D7.
    np.testing.assert_array_equal(beam.measurements[0].doses[:, 1, 2], [103.5, 203.5])
    assert np.isfinite(beam.measurements[0].doses).all()
    np.testing.assert_array_equal(beam.measurements[0].doses[:, 0, 1], [105.5, 205.5])
    np.testing.assert_array_equal(beam.measurements[0].doses[:, -1, 1], [100.5, 200.5])
    np.testing.assert_array_equal(beam.measurements[0].doses[:, 1, 0], [102.5, 202.5])
    np.testing.assert_array_equal(beam.measurements[0].doses[:, 1, -1], [104.5, 204.5])


@pytest.mark.parametrize("direction", ["x", "y"])
def test_movie_interpolation_avoids_blurring_a_steep_penumbra(tmp_path, direction):
    xy = DETECTOR_XY
    axis = X if direction == "x" else Y[::-1]
    profile = np.array([0, 1, 20, 80, 100]) if direction == "x" else np.array([0, 20, 100])
    counts = np.interp(xy[:, 0 if direction == "x" else 1], axis, profile)
    lines = movie_export().splitlines()
    frame = 0
    for index, line in enumerate(lines):
        if line.startswith("Data:"):
            cells = line.split("\t")
            cells[8:16] = map(str, counts * (frame + 1))
            lines[index] = "\t".join(cells)
            frame += 1
    movie = read_export(tmp_path, "\n".join(lines)).measurements[0]
    # Across the edge, equal X/Y averaging gives 30.25 (X edge) or 35
    # (Y edge). Along the edge both bracketing diodes measure exactly 20.
    np.testing.assert_array_equal(movie.doses[:, 1, 2], [20, 40])
    for detector, (x, y) in enumerate(xy):
        row, col = np.flatnonzero(Y == y)[0], np.flatnonzero(X == x)[0]
        np.testing.assert_array_equal(movie.doses[:, row, col], counts[detector] * np.array([1, 2]))


@pytest.mark.parametrize("difference", [9.0, 10.0, 11.0])
def test_directional_interpolation_uses_ten_percent_contrast_difference(difference):
    frames = np.full((1, 3, 5), np.nan)
    frames[:, ::2, ::2] = 0
    frames[:, 1, 1::2] = 0
    frames[0, 0, 0] = 100  # A measured peak sets the contrast threshold to 10.
    frames[0, 1, 1] = 10
    frames[0, 1, 3] = 20 + difference
    frames[0, 0, 2], frames[0, 2, 2] = 10, 20
    along_x, along_y = 15 + difference / 2, 15
    srsmapcheck_module._interpolate_movie(frames, X, Y)
    expected = along_y if difference > 10 else (along_x + along_y) / 2
    assert frames[0, 1, 2] == expected


def test_interpolation_is_scale_independent_and_does_not_mix_frames():
    measured = np.array(
        [
            [100, np.nan, 10, np.nan, 50],
            [np.nan, 10, np.nan, 90, np.nan],
            [0, np.nan, 10, np.nan, 100],
        ],
        dtype=float,
    )
    scales = np.array([0, 1, 100, -1])
    frames = scales[:, None, None] * measured
    original = frames.copy()
    srsmapcheck_module._interpolate_movie(frames, X, Y)
    np.testing.assert_array_equal(frames, scales[:, None, None] * frames[1])
    samples = np.isfinite(original)
    np.testing.assert_array_equal(frames[samples], original[samples])
    assert np.isfinite(frames).all()


def test_interpolation_does_not_extrapolate_or_propagate_filled_samples():
    frames = np.full((1, 5, 5), np.nan)
    frames[0, 0, 0] = 0
    frames[0, 0, 2] = 2
    frames[0, 2, 1] = 20
    srsmapcheck_module._interpolate_movie(frames, np.arange(5), np.arange(5))
    assert frames[0, 0, 1] == 1  # A measured horizontal bracket.
    assert np.isnan(frames[0, 0, 3])  # No measured sample to the right.
    assert np.isnan(frames[0, 1, 1])  # The top bracket was only just interpolated.
    assert np.isnan(frames[0, -1]).all()


def test_interpolation_stays_within_measured_neighbour_bounds():
    rng = np.random.default_rng(17)
    frames = np.full((3, 9, 9), np.nan)
    sampled = np.indices((9, 9)).sum(axis=0) % 2 == 0
    frames[:, sampled] = rng.normal(size=(3, sampled.sum()))
    original = frames.copy()
    srsmapcheck_module._interpolate_movie(frames, np.arange(9), np.arange(9))
    for row, column in np.argwhere(~sampled):
        neighbours = [
            original[:, r, c]
            for r, c in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
            if 0 <= r < 9 and 0 <= c < 9
        ]
        lower, upper = np.min(neighbours, axis=0), np.max(neighbours, axis=0)
        assert np.all(frames[:, row, column] >= lower)
        assert np.all(frames[:, row, column] <= upper)


def test_movie_interpolation_uses_physical_spacing(tmp_path):
    lines = movie_export().splitlines()
    xy = DETECTOR_XY.copy()
    xy[xy[:, 0] == 0, 0] = 0.5
    xy[xy[:, 1] == 0, 1] = 0.25
    frame = 0
    for index, line in enumerate(lines):
        if line.startswith("X (mm)"):
            cells = line.split("\t")
            cells[8:16] = map(str, xy[:, 0])
            lines[index] = "\t".join(cells)
        elif line.startswith("Y (mm)"):
            cells = line.split("\t")
            cells[8:16] = map(str, xy[:, 1])
            lines[index] = "\t".join(cells)
        elif line.startswith("Data:"):
            cells = line.split("\t")
            cells[8:16] = map(str, 100 + 10 * frame + 2 * xy[:, 0] + 3 * xy[:, 1])
            lines[index] = "\t".join(cells)
            frame += 1
    beam = read_export(tmp_path, "\n".join(lines))
    # A planar signal has a known value at every coordinate, even on this
    # nonuniform grid. Linear interpolation must recover it at missing sites.
    expected = (
        100
        + 10 * np.arange(2)[:, None, None]
        + 2 * beam.measurements[0].axes[1][None, None, :]
        + 3 * beam.measurements[0].axes[0][None, :, None]
    )
    np.testing.assert_allclose(beam.measurements[0].doses, expected)


def test_negative_raw_counts_and_final_dose_zeros_are_preserved(tmp_path):
    beam = read_export(tmp_path, movie_export().replace("200\t201\t202", "-2\t0\t202"))
    np.testing.assert_array_equal(beam.measurements[0].doses[-1, 2, :3:2], [-2, 0])
    beam = read_export(tmp_path, measured_export().replace("0.25\t1.25", "0\t-1.25"))
    np.testing.assert_array_equal(beam.measurements[0].doses[0, :2], [0, -1.25])


@pytest.mark.parametrize("make_export", [measured_export, movie_export])
def test_unknown_revision_still_rejects_incomplete_data(tmp_path, make_export):
    text = make_export().replace("File Revision:\tA", "File Revision:\tB")
    if make_export is movie_export:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("Data:\t12"))
    else:
        text = text.split("Dose Interpolated")[0]
    with pytest.warns(UserWarning, match="unvalidated"):
        with pytest.raises(ValueError):
            read_export(tmp_path, text)


@pytest.mark.parametrize("make_export", [measured_export, movie_export])
def test_factory_signature_takes_precedence_over_quoted_format_markers(tmp_path, make_export):
    path = tmp_path / "export.dat"
    text = make_export().replace(
        "Synthetic export", "# Measurement number 1 :MSR Tab-Delimited Scan Output"
    )
    path.write_text(text)
    assert isinstance(BeamDataFactory.from_file(path), SRSMapCheck)


def test_factory_rejects_a_quoted_signature_in_unrelated_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Notes about an export\nSRS MapCHECK Measurement File:\tMeasured File\n")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(path)


@pytest.mark.parametrize("text", ["", "\n\n", "unrelated content"])
def test_empty_or_unrecognized_files_raise_value_error(tmp_path, text):
    with pytest.raises(ValueError, match="empty|signature"):
        read_export(tmp_path, text)


def test_duplicate_header_fields_are_rejected(tmp_path):
    text = measured_export().replace("File Revision:\tA", "File Revision:\tA\nFile Revision:\tA")
    with pytest.raises(ValueError, match="Duplicate.*header"):
        read_export(tmp_path, text)


@pytest.mark.parametrize("time_first", [False, True])
def test_duplicate_compound_time_header_is_rejected(tmp_path, time_first):
    date = "Date:\t01/02/26\tTime:\t12:34:56\t"
    duplicate = "Time:\t00:00:00"
    replacement = f"{duplicate}\n{date}" if time_first else f"{date}\n{duplicate}"
    text = measured_export().replace(date, replacement)
    with pytest.raises(ValueError, match="Duplicate.*header field: Time"):
        read_export(tmp_path, text)


def test_reparse_replaces_data_and_failed_reparse_preserves_previous_acquisition(tmp_path):
    beam = read_export(tmp_path, movie_export())
    path = beam.file_path
    path.write_text(measured_export())
    beam.parse()
    assert len(beam.measurements) == 1
    assert "SRS MapCHECK" not in beam.file_meta
    assert beam.measurements[0].doses.shape == (3, 5)
    path.write_text("invalid data")
    with pytest.raises(ValueError):
        beam.parse()
    np.testing.assert_array_equal(beam.measurements[0].doses, DOSE)


def test_save_is_explicitly_unsupported(tmp_path):
    beam = read_export(tmp_path)
    with pytest.raises(NotImplementedError, match="writing"):
        beam.save(tmp_path / "output.txt")


def test_movie_raw_table_is_released_after_parsing(tmp_path, monkeypatch):
    references = []
    original_movie = srsmapcheck_module._movie

    def observe_movie(*args):
        result = original_movie(*args)
        references.append(weakref.ref(result[3]))
        return result

    monkeypatch.setattr(srsmapcheck_module, "_movie", observe_movie)
    beam = read_export(tmp_path, movie_export())
    gc.collect()
    assert references[0]() is None
    assert set(vars(beam)) == {"file_path", "measurements", "file_meta"}
    np.testing.assert_array_equal(beam.file_meta["SRS MapCHECK"]["Pulses"], [0, 4])
    assert beam.measurements[0].doses.shape == (2, 3, 5)


def test_unused_exported_tables_are_released_after_parsing(tmp_path, monkeypatch):
    references = []
    original_grids = srsmapcheck_module._grids

    def observe_grids(*args):
        result = original_grids(*args)
        references.extend(
            weakref.ref(value) for name, value in result[1].items() if name != "Dose Interpolated"
        )
        return result

    monkeypatch.setattr(srsmapcheck_module, "_grids", observe_grids)
    beam = read_export(tmp_path)
    gc.collect()
    assert references and all(reference() is None for reference in references)
    assert set(vars(beam)) == {"file_path", "measurements", "file_meta"}
    np.testing.assert_array_equal(beam.measurements[0].doses, DOSE)
