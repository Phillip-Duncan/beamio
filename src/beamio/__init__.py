"""Input, output, and conversion of radiotherapy beam measurements."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("beamio")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__author__ = "Phillip Duncan-Gelder"
__all__ = ["__version__", "__author__"]
