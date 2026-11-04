import numpy as np
import pytest

from beamio.common.measurement import Measurement, MeasurementType, RadiationType
from beamio.profile import beam_center, field_edges, field_width
from beamio.rfa import IBAAccept6, PTWMCC, SNCSunScan
from beamio.tps.raystation import Raystation
from tests.support.measurements import profile_samples
from tests.support.ptw import mcc_export, mcc_scan


@pytest.mark.parametrize(
    "reader,suffix",
    [(SNCSunScan, ".snctxt"), (IBAAccept6, ".asc"), (PTWMCC, ".mcc"), (Raystation, ".csv")],
)
@pytest.mark.parametrize("method", ["InflectionPoint", "FWHM"])
def test_public_profile_analysis_on_profiles_loaded_from_beamdata(tmp_path, reader, suffix, method):
    path = tmp_path / ("profile" + suffix)
    measurement = Measurement()
    measurement.type = MeasurementType.CrosslineProfile
    measurement.radiation_type = RadiationType.PHOTON
    measurement.energy = 6
    measurement.SSD = 900
    measurement.field_size = (200, 250)
    positions, doses = profile_samples()
    # Exercise readers that retain reversed acquisition order and IBA's sorting.
    measurement.positions, measurement.doses = positions[::-1], doses[::-1]
    measurement.startpoint, measurement.endpoint = (160, 0, 100), (-160, 0, 100)
    if reader is PTWMCC:
        rows = [f"{p:.12g} {d:.12g}" for p, d in zip(measurement.positions, measurement.doses)]
        path.write_text(mcc_export(mcc_scan(rows=rows)))
    else:
        source = reader("")
        source.measurements = [measurement]
        source.save(path)
    loaded = reader(path).parse().measurements[0]
    nominal_size = loaded.field_size
    assert beam_center(loaded.positions, loaded.doses, method=method) == pytest.approx(
        2.5, abs=0.01
    )
    np.testing.assert_allclose(
        field_edges(loaded.positions, loaded.doses, method=method), [-47.5, 52.5], atol=0.01
    )
    assert field_width(loaded.positions, loaded.doses, method=method) == pytest.approx(
        100, abs=0.01
    )
    assert loaded.field_size == nominal_size
