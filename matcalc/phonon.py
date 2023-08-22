"""Calculator for phonon properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import phonopy
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


class PhononCalc(PropCalc):
    """Calculator for phonon properties."""

    def __init__(
        self,
        calculator: Calculator,
        atom_disp: float = 0.015,
        supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 0.1,
        relax_structure: bool = True,
    ) -> None:
        """
        Args:
            calculator: ASE Calculator to use.
            fmax: Max forces. This criterion is more stringent than for simple relaxation.
            atom_disp: Atomic displacement
            supercell_matrix: Supercell matrix to use. Defaults to 2x2x2 supercell.
            t_step: Temperature step.
            t_max: Max temperature.
            t_min: Min temperature.
            relax_structure: Whether to first relax the structure. Set to False if structures
                provided are pre-relaxed with the same calculator.
        """
        self.calculator = calculator
        self.atom_disp = atom_disp
        self.supercell_matrix = supercell_matrix
        self.fmax = fmax
        self.relax_structure = relax_structure
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min

    def calc(self, structure: Structure) -> dict:
        """
        Calculates thermal properties of Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.

        Returns: {
            temperatures: list of temperatures in Kelvin,
            free_energy: list of Helmholtz free energies at corresponding temperatures in eV,
            entropy: list of entropies at corresponding temperatures in eV/K,
            heat_capacity: list of heat capacities at constant volume at corresponding temperatures in eV/K^2,
        }
        """
        if self.relax_structure:
            relaxer = RelaxCalc(self.calculator, fmax=self.fmax)
            structure = relaxer.calc(structure)["final_structure"]
        cell = get_phonopy_structure(structure)
        phonon = phonopy.Phonopy(cell, self.supercell_matrix)
        phonon.generate_displacements(distance=self.atom_disp)
        disp_supercells = phonon.supercells_with_displacements
        phonon.forces = [
            _calc_forces(self.calculator, supercell) for supercell in disp_supercells if supercell is not None
        ]
        phonon.produce_force_constants()
        phonon.run_mesh()
        phonon.run_thermal_properties(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        return phonon.get_thermal_properties_dict()


def _calc_forces(calculator: Calculator, supercell: PhonopyAtoms) -> ArrayLike:
    """
    Helper to compute forces on a structure.

    Args:
        calculator: ASE Calculator
        supercell: Supercell from phonopy.

    Return:
        forces
    """
    struct = get_pmg_structure(supercell)
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator
    return atoms.get_forces()
