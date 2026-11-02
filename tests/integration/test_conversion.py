from beamio.rfa import IBAAccept6
from beamio.tps.raystation import Raystation
from tests.support.measurements import assert_scan_equal


def test_conversion_is_independent_and_exports(tmp_path, iba_text):
    source = tmp_path / "source.asc"
    source.write_text(iba_text, encoding="utf-8")
    beam = IBAAccept6(source).parse()
    converted = beam.convert(Raystation)
    converted.measurements[0].doses[0] = 123
    converted.measurements[0].metadata["CUSTOM"] = "changed"
    converted.file_meta["SYS"] = "changed"
    assert beam.measurements[0].doses[0] == 40
    assert beam.measurements[0].metadata["CUSTOM"] == "preserved"
    assert beam.file_meta["SYS"] == "BDS 0"
    destination = tmp_path / "converted.csv"
    converted.save(destination)
    assert_scan_equal(Raystation(destination).parse().measurements[0], converted.measurements[0])
