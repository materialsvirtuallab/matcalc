"""
Define basic API.
"""
import abc

from pymatgen.core import Structure

class PropCalc(abc.ABCMeta):
    """
    API for a property calculator.
    """

    @abc.abstractmethod
    def calc(self, structure: Structure) -> dict:
        pass