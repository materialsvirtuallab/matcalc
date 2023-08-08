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


class PhononCalc(PropCalc):
    """Calculator for phonon properties."""

    def __init__(
        self,
        calculator: Calculator,
        atom_disp=0.015,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.01,
        relax_structure=True,
        t_step=10,
        t_max=1000,
        t_min=0,
    ):
        """
        Args:
            calculator: ASE Calculator to use.
            atom_disp: Atomic displacement
            supercell_matrix: Supercell matrix to use. Defaults to 2x2x2 supercell.
            fmax: Max forces. This criterion is more stringent than for simple relaxation.
            relax_structure: Whether to first relax the structure. Set to False if structures provided are pre-relaxed
                with the same calculator.
            t_step: Temperature step.
            t_max: Max temperature.
            t_min: Min temperature.
        """
        self.calculator = calculator
        self.atom_disp = atom_disp
        self.supercell_matrix = supercell_matrix
        self.fmax = fmax
        self.relax_structure = relax_structure
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min

    def calc(self, structure) -> dict:
        """
        Calculates thermal properties of Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.

        Returns: {
            "temperature": list of temperatures,
            "free_energy": list of Hemlholtz free energies at corresponding temperatures,
            "entropy": list of entropies at corresponding temperatures,
            "heat_capacities": list of heat capacities at constant volume at corresponding temperatures,
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


def _calc_forces(calculator, supercell):
    """
    Helper to compute forces on a structure.

    Args:
        calculator: Calculator
        supercell: Supercell from phonopy.

    Return:
        forces
    """
    s = get_pmg_structure(supercell)
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(s)
    atoms.calc = calculator
    return atoms.get_forces()
