"""QWiP - Quantum Workflows in Python

A python library for running experiments with superconducting quantum devices.
"""
try:
    from importlib.metadata import version, PackageNotFoundError # type:ignore
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = ''