"""Phonon properties."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import PropCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class PhononCalc(PropCalc):
    """Calculator for phonon properties."""

    def __init__(self, calculator: Calculator):
        """
        Args:
            calculator: ASE Calculator to use.
        """
        self.calculator = calculator

    def calc(self, structure) -> dict:
        """
        All PropCalc should implement a calc method that takes in a pymatgen structure and returns a dict. Note that
        the method can return more than one property.

        Args:
            structure: Pymatgen structure.

        Returns: {"prop name": value}
        """
        return {}
