"""
Phonon properties
"""

from ase.calculators.calculator import Calculator

from .base import PropCalc

class PhononCalc(PropCalc):
    """
    Calculator for phonon properties.
    """

    def __init__(self, calculator: Calculator):
        self.calculator = calculator

    def calc(self, structure) -> dict:
        pass