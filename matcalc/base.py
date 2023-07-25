"""Define basic API."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure


class PropCalc(abc.ABCMeta):
    """API for a property calculator."""

    @abc.abstractmethod
    def calc(self, structure: Structure) -> dict:
        """
        All PropCalc should implement a calc method that takes in a pymatgen structure and returns a dict. Note that
        the method can return more than one property.

        Args:
            structure: Pymatgen structure.

        Returns: {"prop name": value}
        """
