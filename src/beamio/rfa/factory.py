import os
import xml.etree.ElementTree as ET

from beamio.rfa.iba import IBAAccept6
from beamio.rfa.icprofiler import ICProfiler
from beamio.rfa.ptw import PTWMCC
from beamio.rfa.snc import SNCSunScan
from beamio.common.beamdata import BeamData

# --------------------------------------------------------------------------
# Each key is a BeamData subclass
# Each value is a dict with:
#   'first_line' -> allowed first nonblank lines; these signatures take precedence
#   'all' → list of keywords that must ALL be present
#   'any' → list of keywords where at least one must be present
#   'file_extensions' → list of valid file extensions (without the leading dot e.g., 'asc', 'txt')
# --------------------------------------------------------------------------
BEAMDATA_KEYWORDS: dict[type[BeamData], dict[str, list[str]]] = {
    PTWMCC: {
        "first_line": ["BEGIN_SCAN_DATA"],
        "file_extensions": ["mcc"],
    },
    ICProfiler: {
        "file_extensions": ["prm"],
    },
    IBAAccept6: {
        "all": [r"# Measurement number"],
        "any": [r":MSR", r"%FSZ", r"%STS", r"%EDS"],
        "file_extensions": ["asc"],
    },
    SNCSunScan: {
        "any": ["Tab-Delimited Scan Output"],
        "file_extensions": ["snctxt", "sncxml"],
    },
}


def _icprofiler_header(file_path: str) -> bool:
    """Inspect complete PRM header lines without scanning its acquisition data."""
    with open(file_path, "rb") as source:
        for raw in source:
            cells = [cell.strip() for cell in raw.rstrip(b"\r\n").split(b"\t")]
            if cells[0] in (b"Definitions:", b"TYPE", b"Data:"):
                break
            if cells[0] == b"Collector Model:":
                return len(cells) > 1 and cells[1] == b"IC PROFILER"
    return False


class BeamDataFactory:
    """Identify beam data by text signatures or its XML root."""

    @staticmethod
    def from_file(file_path: str) -> BeamData:
        """Identify and load the correct BeamData subclass based on file contents."""
        with open(file_path, "rb") as f:
            prefix = f.read(8192)
            # Feed bytes so XML declarations/BOMs and namespaces are respected.
            # Continue through a long XML prolog instead of assuming a fixed offset.
            parser = ET.XMLPullParser(events=("start",))
            xml_root = None
            chunk = prefix
            try:
                while chunk:
                    parser.feed(chunk)
                    event = next(parser.read_events(), None)
                    if event is not None:
                        xml_root = event[1].tag.rsplit("}", 1)[-1]
                        break
                    chunk = f.read(8192)
                if xml_root is None:
                    # Expat may defer a large prolog's events until end-of-input.
                    parser.close()
                    event = next(parser.read_events(), None)
                    if event is not None:
                        xml_root = event[1].tag.rsplit("}", 1)[-1]
            except ET.ParseError:
                pass
        if xml_root == "snc_fileexport":
            return SNCSunScan(file_path).parse()
        head = prefix.decode("utf-8-sig", errors="ignore")
        first_line = next((line.strip() for line in head.splitlines() if line.strip()), "")

        for cls, rules in BEAMDATA_KEYWORDS.items():
            if cls is ICProfiler:
                # Free-text header fields can extend beyond the initial prefix.
                # Inspect labelled lines rather than matching quoted signatures.
                if first_line.partition("\t")[0].strip() == "Version:" and _icprofiler_header(
                    file_path
                ):
                    return cls(file_path).parse()
                continue
            first_lines = rules.get("first_line", [])
            if first_lines and first_line not in first_lines:
                continue
            all_keys = rules.get("all", [])
            any_keys = rules.get("any", [])

            # Check "all" keywords
            if all_keys and not all(k in head for k in all_keys):
                continue

            # Check "any" keywords
            if any_keys and not any(k in head for k in any_keys):
                continue

            # Match found
            return cls(file_path).parse()

        raise ValueError(f"Unsupported or unrecognized beam data file: {file_path}")

    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        """Check if the file is a supported BeamData file based on its extension."""
        ext = os.path.splitext(file_path)[1].lstrip(".").lower()
        for rules in BEAMDATA_KEYWORDS.values():
            if "file_extensions" in rules and ext in rules["file_extensions"]:
                return True
        return False

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Get a list of all supported file extensions for BeamData files."""
        extensions = set()
        for rules in BEAMDATA_KEYWORDS.values():
            exts = rules.get("file_extensions", [])
            extensions.update(exts)
        return sorted(extensions)
