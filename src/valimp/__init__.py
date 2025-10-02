"""valimp."""

import contextlib

from . import valimp
from .valimp import *  # noqa: F403

__all__ = [valimp]

__copyright__ = "Copyright (c) 2023 Marcus Read"

# Resolve version
__version__ = None

from importlib.metadata import version

with contextlib.suppress(ImportError):
    # get version from installed package
    __version__ = version("valimp")

if __version__ is None:
    try:
        # if package not installed, get version as set when package built
        from ._version import version
    except Exception:  # noqa: BLE001, S110
        # If package not installed and not built, leave __version__ as None
        pass
    else:
        __version__ = version

del version
