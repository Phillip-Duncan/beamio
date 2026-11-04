"""Conversion checks target capabilities without changing acquired samples."""

from copy import deepcopy
import warnings

import numpy as np
import pytest

from beamio.common import AcquisitionType, MeasurementType, RadiationType
from beamio.common.beamdata import BeamData
from beamio.rfa import IBAAccept6, ICProfiler, PTWMCC, SNCSunScan, SRSMapCheck
from beamio.tps.raystation import Raystation
from tests.support.measurements import array_measurement
from tests.support.srsmapcheck import measured_export


class ArrayFormat(BeamData):
    """An in-memory test target that can represent every shared array layout."""

    supported_acquisition_types = frozenset(AcquisitionType)
    supported_measurement_types = frozenset(MeasurementType)
    supported_radiation_types = frozenset({None, *RadiationType})

    def parse(self):
        return self

    def save(self, output_path):
        raise NotImplementedError


class UndeclaredFormat(BeamData):
    def parse(self):
        return self

    def save(self, output_path):
        raise NotImplementedError


def test_mixed_conversion_warns_once_per_drop_preserves_order_and_source(measurement):
    source = ArrayFormat("")
    first = deepcopy(measurement)
    first.doses = np.array([1, 2, 3])
    unsupported = array_measurement(MeasurementType.DosePlane, temporal=True)
    unsupported.radiation_type = RadiationType.PROTON
    last = deepcopy(measurement)
    last.radiation_type = RadiationType.ELECTRON
    last.doses = np.array([4, 5, 6])
    source.measurements = [first, unsupported, last]
    source.file_meta = {"native": {"values": np.array([7, 8])}}
    with pytest.warns(UserWarning, match="Skipping measurement 2 converting to PTWMCC") as caught:
        result = source.convert(PTWMCC, "result.mcc")
    assert len(caught) == 1 and caught[0].filename == __file__
    for reason in ("acquisition_type=TimeSeries", "type=DosePlane", "radiation_type=Proton"):
        assert reason in str(caught[0].message)
    assert result.file_path == "result.mcc"
    assert len(result.measurements) == 2 and len(source.measurements) == 3
    np.testing.assert_array_equal(result.measurements[0].doses, [1, 2, 3])
    np.testing.assert_array_equal(result.measurements[1].doses, [4, 5, 6])
    result.measurements[0].doses[0] = 999
    result.file_meta["native"]["values"][0] = 999
    assert source.measurements[0].doses[0] == 1
    assert source.file_meta["native"]["values"][0] == 7


@pytest.mark.parametrize(
    "target,kind,temporal,radiation,accepted",
    [
        (IBAAccept6, MeasurementType.DiagonalProfile, False, RadiationType.PROTON, True),
        (IBAAccept6, MeasurementType.CrosslineProfile, True, RadiationType.PHOTON, False),
        (SNCSunScan, MeasurementType.DepthDose, False, RadiationType.ELECTRON, True),
        (SNCSunScan, MeasurementType.DosePlane, False, RadiationType.PHOTON, False),
        (Raystation, MeasurementType.CrosslineProfile, False, RadiationType.PROTON, True),
        (Raystation, MeasurementType.DoseVolume, False, RadiationType.PHOTON, False),
        (PTWMCC, MeasurementType.InlineProfile, False, RadiationType.ELECTRON, True),
        (PTWMCC, MeasurementType.DiagonalProfile, False, RadiationType.PHOTON, False),
        (PTWMCC, MeasurementType.DepthDose, False, RadiationType.PROTON, False),
        (ICProfiler, MeasurementType.DiagonalProfile, True, RadiationType.PHOTON, True),
        (ICProfiler, MeasurementType.CrosslineProfile, False, RadiationType.PHOTON, False),
        (ICProfiler, MeasurementType.DepthDose, True, RadiationType.PHOTON, False),
        (SRSMapCheck, MeasurementType.DosePlane, False, RadiationType.PHOTON, True),
        (SRSMapCheck, MeasurementType.DosePlane, True, RadiationType.PHOTON, True),
        (SRSMapCheck, MeasurementType.DosePlane, True, RadiationType.ELECTRON, False),
        (SRSMapCheck, MeasurementType.CrosslineProfile, False, RadiationType.PHOTON, False),
    ],
)
def test_concrete_format_capabilities_control_conversion(
    target, kind, temporal, radiation, accepted
):
    source = ArrayFormat("")
    scan = array_measurement(kind, temporal)
    scan.radiation_type = radiation
    source.measurements = [scan]
    if accepted:
        with warnings.catch_warnings(record=True) as caught:
            result = source.convert(target)
        assert not caught and len(result.measurements) == 1
    else:
        with pytest.warns(UserWarning, match=f"converting to {target.__name__}"):
            result = source.convert(target)
        assert result.measurements == []
    assert source.measurements == [scan]


@pytest.mark.parametrize(
    "target", [IBAAccept6, SNCSunScan, PTWMCC, ICProfiler, SRSMapCheck, Raystation]
)
def test_unknown_radiation_is_explicitly_supported(target):
    source = ArrayFormat("")
    kind = next(iter(target.supported_measurement_types))
    temporal = AcquisitionType.Static not in target.supported_acquisition_types
    scan = array_measurement(kind, temporal)
    scan.radiation_type = None
    source.measurements = [scan]
    with warnings.catch_warnings(record=True) as caught:
        result = source.convert(target)
    assert not caught
    assert result.measurements[0].radiation_type is None


@pytest.mark.parametrize("attribute", ["type", "acquisition_type", "radiation_type"])
@pytest.mark.parametrize("invalid", ["unknown", []])
def test_invalid_type_values_are_warned_and_omitted(measurement, attribute, invalid):
    source = ArrayFormat("")
    setattr(measurement, attribute, invalid)
    source.measurements = [measurement]
    with pytest.warns(UserWarning, match=attribute):
        result = source.convert(ArrayFormat)
    assert result.measurements == []


def test_undeclared_capabilities_do_not_silently_accept_measurements(measurement):
    source = ArrayFormat("")
    source.measurements = [measurement]
    with pytest.warns(UserWarning, match="UndeclaredFormat"):
        result = source.convert(UndeclaredFormat)
    assert result.measurements == []


def test_empty_conversion_is_quiet_and_retains_independent_metadata():
    source = ArrayFormat("")
    source.file_meta = {"headers": {"source": "original"}}
    with warnings.catch_warnings(record=True) as caught:
        result = source.convert(SRSMapCheck)
    assert not caught and result.measurements == []
    result.file_meta["headers"]["source"] = "changed"
    assert source.file_meta["headers"]["source"] == "original"


def test_every_unsupported_measurement_warns_when_the_result_is_empty():
    source = ArrayFormat("")
    source.measurements = [
        array_measurement(MeasurementType.DoseVolume),
        array_measurement(MeasurementType.DosePlane, temporal=True),
    ]
    source.file_meta = {"source": "retained"}
    with pytest.warns(UserWarning) as caught:
        result = source.convert(IBAAccept6)
    assert len(caught) == 2
    assert "measurement 1" in str(caught[0].message)
    assert "measurement 2" in str(caught[1].message)
    assert result.measurements == []
    assert result.file_meta == source.file_meta
    assert len(source.measurements) == 2


@pytest.mark.parametrize("temporal", [False, True])
def test_capable_custom_target_preserves_volume_geometry_and_time(temporal):
    source = ArrayFormat("")
    scan = array_measurement(MeasurementType.DoseVolume, temporal)
    scan.coordinates = np.stack(scan.reconstruct_xyz(), axis=-1)
    source.measurements = [scan]
    result = source.convert(ArrayFormat)
    actual = result.measurements[0]
    actual.validate()
    np.testing.assert_array_equal(actual.doses, scan.doses)
    np.testing.assert_array_equal(actual.coordinates, scan.coordinates)
    if temporal:
        np.testing.assert_array_equal(actual.times, scan.times)
        assert not np.shares_memory(actual.times, scan.times)
    for a, b in zip(actual.axes, scan.axes):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
    assert not np.shares_memory(actual.doses, scan.doses)
    assert not np.shares_memory(actual.coordinates, scan.coordinates)


def test_conversion_preserves_native_aliases_without_aliasing_the_source(tmp_path):
    path = tmp_path / "map.txt"
    path.write_text(measured_export())
    source = SRSMapCheck(path).parse()
    result = source.convert(SRSMapCheck)
    assert (
        result.file_meta["SRS MapCHECK"]["Tables"]["Dose Interpolated"]
        is result.measurements[0].doses
    )
    assert not np.shares_memory(result.measurements[0].doses, source.measurements[0].doses)
    result.measurements[0].doses[0, 0] = 999
    assert source.measurements[0].doses[0, 0] != 999


@pytest.mark.parametrize("target", [None, object, SRSMapCheck("")])
def test_convert_requires_a_beamdata_class(target):
    with pytest.raises(TypeError, match="BeamData subclass"):
        ArrayFormat("").convert(target)
