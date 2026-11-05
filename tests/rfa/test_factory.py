import codecs

import pytest

from beamio.rfa import BeamDataFactory, ICProfiler, SNCSunScan
from tests.support.icprofiler import prm_export
from tests.support.snc import xml_export


@pytest.mark.parametrize(
    "name",
    [
        "scan.asc",
        "scan.ASC",
        "folder/scan.Asc",
        "scan.snctxt",
        "scan.SNCXML",
        "scan.MCC",
        "scan.PRM",
        "scan.TXT",
        "scan.SMCM",
        "scan.OPG",
        "scan.OPD",
        "scan.OPW",
    ],
)
def test_supported_extensions(name):
    assert BeamDataFactory.is_supported_file(name)
    assert BeamDataFactory.get_supported_extensions() == [
        "asc",
        "mcc",
        "opd",
        "opg",
        "opw",
        "prm",
        "smcm",
        "snctxt",
        "sncxml",
        "txt",
    ]


@pytest.mark.parametrize("name", ["scan.csv", "scan", "scan.asc.bak"])
def test_unsupported_extensions(name):
    assert not BeamDataFactory.is_supported_file(name)


def test_factory_rejects_unrecognized_content(tmp_path):
    source = tmp_path / "unknown.asc"
    source.write_text("Unknown format", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(source)


def test_mcc_marker_in_unrelated_text_is_not_a_signature(tmp_path):
    path = tmp_path / "notes.mcc"
    path.write_text("Notes about MCC exports\nBEGIN_SCAN_DATA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(path)


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


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_icprofiler_detection_reads_past_long_free_text_header(tmp_path, encoding):
    path = tmp_path / "measurement.dat"
    text = prm_export().replace(
        "Description:\tSynthetic acquisition", "Description:\t" + "Long note " * 2000
    )
    path.write_bytes(text.encode(encoding))
    assert isinstance(BeamDataFactory.from_file(path), ICProfiler)


def test_icprofiler_detection_attempts_unvalidated_versions_with_a_warning(tmp_path):
    path = tmp_path / "measurement.prm"
    path.write_text(prm_export().replace("Version:\t 25", "Version:\t 26"))
    with pytest.warns(UserWarning, match="version '26'.*is unvalidated"):
        acquisition = BeamDataFactory.from_file(path)
    assert isinstance(acquisition, ICProfiler)
    assert acquisition.file_meta["Version"] == "26"
    assert len(acquisition.measurements) == 4


def test_quoted_profiler_signature_in_unrelated_text_is_not_a_match(tmp_path):
    path = tmp_path / "notes.prm"
    path.write_text("Version:\t25\nDescription:\tCollector Model:\tIC PROFILER\n")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(path)


def test_profiler_detection_does_not_search_acquisition_data_for_a_model(tmp_path):
    path = tmp_path / "notes.prm"
    path.write_text("Version:\t25\nTYPE\tData\nCollector Model:\tIC PROFILER\n")
    with pytest.raises(ValueError, match="Unsupported or unrecognized"):
        BeamDataFactory.from_file(path)
