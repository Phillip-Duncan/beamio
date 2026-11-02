"""Small, synthetic scans; no acquired beam data is needed by the tests."""

import numpy as np
import pytest

from beamio.common.measurement import Measurement, MeasurementType, RadiationType


def pytest_collection_modifyitems(items):
    for item in items:
        for marker in ("integration", "e2e"):
            if marker in item.path.parts:
                item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture
def measurement():
    scan = Measurement()
    scan.type = MeasurementType.DepthDose
    scan.radiation_type = RadiationType.PHOTON
    scan.energy = 6.0
    scan.SSD = 900.0
    scan.field_size = (100.0, 100.0)
    scan.startpoint = (0.0, 0.0, 0.0)
    scan.endpoint = (0.0, 0.0, 20.0)
    scan.positions = np.array([0.0, 10.0, 20.0])
    scan.doses = np.array([40.0, 100.0, 85.0])
    return scan


@pytest.fixture
def iba_text():
    return """:MSR 1
:SYS BDS 0
# Measurement number 1
%SCN DPT
%BMT PHO 6
%SSD 900
%FSZ 100 100
%STS 0.1 -0.1 0
%EDS 0 0 20
%CUSTOM preserved
= 0 0 0 40
= 0 0 10 100
= 0 0 10 99
= 0 0 15 NaN
= 0 0 20 85
:EOM
:EOF
"""
