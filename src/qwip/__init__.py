"""QWiP - Quantum Workflows in Python

A python library for running experiments with superconducting quantum devices.
"""

try:
    from importlib.metadata import PackageNotFoundError, version  # type:ignore
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = ""

import sys

from loguru import logger

from qwip._cattr import converter
from qwip.flatdict import FlatDict

logger.remove()
logger.add(sys.stdout, level="WARNING")

from qwip._qsettings import default_qsettings

qsettings = default_qsettings()
"""The global QWiP Settings object"""

__all__ = [
    "converter",
    "FlatDict",
    "qsettings",
]
