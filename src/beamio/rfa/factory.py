import os
from iba import IBAAccept6
from beamio.common.beamdata import BeamData

# --------------------------------------------------------------------------
# Each key is a BeamData subclass
# Each value is a dict with:
#   'all' → list of keywords that must ALL be present
#   'any' → list of keywords where at least one must be present
#   'file_extensions' → list of valid file extensions (without the leading dot e.g., 'asc', 'txt')
# --------------------------------------------------------------------------
BEAMDATA_KEYWORDS: dict[type[BeamData], dict[str, list[str]]] = {
    IBAAccept6: {
        "all": [r"# Measurement number"],
        "any": [r":MSR", r"%FSZ", r"%STS", r"%EDS"],
        "file_extensions": ["asc"],
    },
    # PTWBeamScan: {
    #     "all": ["BEGIN_SCAN_DATA"],
    #     "any": ["BeamScan"],
    # },
}


class BeamDataFactory:
    """Keyword-based BeamData detector and loader."""

    @staticmethod
    def from_file(file_path: str) -> BeamData:
        """Identify and load the correct BeamData subclass based on file contents."""
        with open(file_path, "r", errors="ignore") as f:
            head = f.read(4096)

        for cls, rules in BEAMDATA_KEYWORDS.items():
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
        _, ext = os.path.splitext(file_path)
        for rules in BEAMDATA_KEYWORDS.values():
            if "file_extensions" in rules and ext.lower() in rules["file_extensions"]:
                return True
        return False
    
    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Get a list of all supported file extensions for BeamData files."""
        extensions = set()
        for rules in BEAMDATA_KEYWORDS.values():
            exts = rules.get("file_extensions", [])
            extensions.update(exts)
        return list(extensions)
    
