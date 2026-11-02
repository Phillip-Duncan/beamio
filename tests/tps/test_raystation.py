from copy import deepcopy

import numpy as np
import pytest

from beamio.common.measurement import RadiationType
from beamio.tps.raystation import Raystation
from tests.support.measurements import AXIAL_SCANS, assert_scan_equal


@pytest.mark.parametrize("scan_type,start,end,positions", AXIAL_SCANS)
def test_round_trip(tmp_path, measurement, scan_type, start, end, positions):
    measurement.type = scan_type
    measurement.startpoint = start
    measurement.endpoint = end
    measurement.positions = np.array(positions, dtype=float)
    path = tmp_path / "roundtrip.csv"
    beam = Raystation("")
    beam.measurements = [measurement, deepcopy(measurement)]
    beam.save(path)
    loaded = Raystation(path).parse()
    assert len(loaded.measurements) == 2
    for actual in loaded.measurements:
        assert_scan_equal(actual, measurement)


def test_wedge_and_field_size(tmp_path, measurement):
    measurement.wedge_angle = 30.0
    measurement.radiation_type = RadiationType.ELECTRON
    beam = Raystation("")
    beam.measurements = [measurement]
    destination = tmp_path / "wedged.csv"
    beam.save(destination)
    text = destination.read_text(encoding="utf-8")
    assert "WedgeAngle[deg]:; 30" in text
    assert "Fieldsize[mm]:; -50; -50; 50; 50" in text
    parsed = Raystation(destination).parse().measurements[0]
    assert parsed.wedge_angle == 30.0
    assert_scan_equal(parsed, measurement)


def test_reject_unsupported_output(tmp_path):
    with pytest.raises(ValueError, match="Unsupported output format"):
        Raystation("").save(tmp_path / "output.txt")


def test_empty_file_round_trip(tmp_path):
    path = tmp_path / "empty.csv"
    Raystation("").save(path)
    assert Raystation(path).parse().measurements == []
