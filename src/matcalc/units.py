"""Useful constants for unit conversions."""

from __future__ import annotations

from scipy import constants

eVA3ToGPa = constants.e / (constants.angstrom) ** 3 / constants.giga  # noqa: N816
