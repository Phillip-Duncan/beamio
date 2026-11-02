from copy import deepcopy

import numpy as np
import pytest

from beamio.rfa import BeamDataFactory, IBAAccept6
from tests.support.measurements import AXIAL_SCANS, assert_scan_equal


@pytest.mark.parametrize("end_marker", [True, False])
def test_parse_and_cleanup(tmp_path, iba_text, measurement, end_marker):
    source = tmp_path / "scan.asc"
    if not end_marker:
        iba_text = iba_text.replace(":EOM\n", "")
    source.write_text(iba_text, encoding="utf-8")
    beam = BeamDataFactory.from_file(source)
    assert isinstance(beam, IBAAccept6)
    assert len(beam.measurements) == 1
    assert_scan_equal(beam.measurements[0], measurement)
    assert beam.measurements[0].endpoint == measurement.endpoint
    assert beam.measurements[0].metadata["CUSTOM"] == "preserved"
    beam.parse()
    assert len(beam.measurements) == 1


@pytest.mark.parametrize("scan_type,start,end,positions", AXIAL_SCANS)
def test_round_trip(tmp_path, measurement, scan_type, start, end, positions):
    measurement.type = scan_type
    measurement.startpoint = start
    measurement.endpoint = end
    measurement.positions = np.array(positions, dtype=float)
    path = tmp_path / "roundtrip.asc"
    beam = IBAAccept6("")
    beam.measurements = [measurement, deepcopy(measurement)]
    beam.save(path)
    loaded = IBAAccept6(path).parse()
    assert len(loaded.measurements) == 2
    for actual in loaded.measurements:
        assert_scan_equal(actual, measurement)


def test_reject_unsupported_output(tmp_path):
    with pytest.raises(ValueError, match="Unsupported output format"):
        IBAAccept6("").save(tmp_path / "output.txt")


def test_empty_file_round_trip(tmp_path):
    path = tmp_path / "empty.asc"
    IBAAccept6("").save(path)
    assert IBAAccept6(path).parse().measurements == []


def test_diagonal_cleanup_keeps_coordinates_aligned(tmp_path):
    path = tmp_path / "diagonal.asc"
    path.write_text(""":MSR 1
# Measurement number 1
%SCN DIA
= 10 5 50 60
= 4 -3 50 100
= -2 -11 50 50
= 4 -3 50 99
= 16 13 50 NaN
:EOM
:EOF
""")
    measurement = IBAAccept6(path).parse().measurements[0]
    np.testing.assert_allclose(measurement.positions, [-10, 0, 10], atol=1e-12)
    np.testing.assert_array_equal(measurement.doses, [50, 100, 60])
    expected = [[-2, -11, 50], [4, -3, 50], [10, 5, 50]]
    np.testing.assert_array_equal(measurement.coordinates, expected)
    np.testing.assert_allclose(np.column_stack(measurement.reconstruct_xyz()), expected)
