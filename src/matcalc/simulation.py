"""
This module implements convenience functions to perform various atomic simulation types, specifically relaxation and
static, using either ASE or LAMMPS. This enables the code to use either for the computation of various properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .utils import to_ase_atoms

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure


class PESResult(NamedTuple):
    """Container for results from PES calculators."""

    energy: float
    forces: np.ndarray
    stress: np.ndarray


def run_ase_static(structure: Structure | Atoms, calculator: Calculator) -> PESResult:
    """
    Run ASE static calculation using the given structure and calculator.

    Parameters:
    structure (Structure|Atoms): The input structure to calculate potential energy, forces, and stress.
    calculator (Calculator): The calculator object to use for the calculation.

    Returns:
    PESResult: Object containing potential energy, forces, and stress of the input structure.
    """
    atoms = to_ase_atoms(structure)
    atoms.calc = calculator
    return PESResult(atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress(voigt=False))
