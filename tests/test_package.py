from importlib.metadata import version

import beamio


def test_installed_version():
    assert beamio.__version__ == version("beamio")
