import pytest

from beamio.rfa import BeamDataFactory


@pytest.mark.parametrize("name", ["scan.asc", "scan.ASC", "folder/scan.Asc"])
def test_supported_extensions(name):
    assert BeamDataFactory.is_supported_file(name)
    assert BeamDataFactory.get_supported_extensions() == ["asc"]


@pytest.mark.parametrize("name", ["scan.csv", "scan.mcc", "scan", "scan.asc.bak"])
def test_unsupported_extensions(name):
    assert not BeamDataFactory.is_supported_file(name)


def test_factory_rejects_unrecognized_content(tmp_path):
    source = tmp_path / "unknown.asc"
    source.write_text("Unknown format", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(source)
