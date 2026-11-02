import codecs

import pytest

from beamio.rfa import BeamDataFactory, SNCSunScan
from tests.support.snc import xml_export


@pytest.mark.parametrize(
    "name", ["scan.asc", "scan.ASC", "folder/scan.Asc", "scan.snctxt", "scan.SNCXML"]
)
def test_supported_extensions(name):
    assert BeamDataFactory.is_supported_file(name)
    assert BeamDataFactory.get_supported_extensions() == ["asc", "snctxt", "sncxml"]


@pytest.mark.parametrize("name", ["scan.csv", "scan.mcc", "scan", "scan.asc.bak"])
def test_unsupported_extensions(name):
    assert not BeamDataFactory.is_supported_file(name)


def test_factory_rejects_unrecognized_content(tmp_path):
    source = tmp_path / "unknown.asc"
    source.write_text("Unknown format", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(source)


@pytest.mark.parametrize("encoding", ["utf8", "utf8-bom", "utf16-le", "utf16-be"])
@pytest.mark.parametrize("namespace", ["plain", "default", "prefixed"])
def test_detect_xml_encoding_and_namespace(tmp_path, encoding, namespace):
    text = xml_export(namespace=namespace == "default")
    if namespace == "prefixed":
        text = text.replace("<snc_fileexport ", '<snc:snc_fileexport xmlns:snc="urn:snc-test" ')
        text = text.replace("</snc_fileexport>", "</snc:snc_fileexport>")
    declaration = "UTF-16" if encoding.startswith("utf16") else "UTF-8"
    text = f'<?xml version="1.0" encoding="{declaration}"?>\n' + text
    if encoding == "utf16-le":
        data = codecs.BOM_UTF16_LE + text.encode("utf-16-le")
    elif encoding == "utf16-be":
        data = codecs.BOM_UTF16_BE + text.encode("utf-16-be")
    else:
        data = text.encode("utf-8-sig" if encoding == "utf8-bom" else "utf-8")
    path = tmp_path / "scan.sncxml"
    path.write_bytes(data)
    assert len(SNCSunScan(path).parse().measurements) == 1
    result = BeamDataFactory.from_file(path)
    assert isinstance(result, SNCSunScan)
    assert len(result.measurements) == 1


def test_unrelated_xml_with_snc_child_is_not_detected(tmp_path):
    path = tmp_path / "other.sncxml"
    path.write_text("<unrelated><snc_fileexport /></unrelated>")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(path)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_detect_xml_after_long_prolog(tmp_path, encoding):
    path = tmp_path / "scan.sncxml"
    text = f'<?xml version="1.0" encoding="{encoding}"?>\n'
    text += "<!--" + "export note " * 1500 + "-->\n" + xml_export()
    path.write_text(text, encoding=encoding)
    assert len(BeamDataFactory.from_file(path).measurements) == 1
