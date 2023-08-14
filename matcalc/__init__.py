"""Calculators for materials properties."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matcalc")
except PackageNotFoundError:
    pass  # package not installed
