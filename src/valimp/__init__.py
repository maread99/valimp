"""valimp."""

from . import valimp
from .valimp import *

__all__ = [valimp]

__copyright__ = "Copyright (c) 2023 Marcus Read"

# Resolve version
__version__ = None

from importlib.metadata import version

try:
    # get version from installed package
    __version__ = version("valimp")
except ImportError:
    pass

if __version__ is None:
    try:
        # if package not installed, get version as set when package built
        from ._version import version
    except Exception:
        # If package not installed and not built, leave __version__ as None
        pass
    else:
        __version__ = version

del version
