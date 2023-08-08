"""Calculator for phonon properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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
        fmax=0.1,
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
            fmax: Max forces.
            relax_structure: Whether to first relax the structure.
                Set to False if structures provided are pre-relaxed
                with the same calculator.
            t_step: increment of temperature in returned thermal property values
            t_max: max temperature for returned thermal property values
            t_min: min temperature for returned thermal porperty values.
        """
        self.calculator = calculator
        self.phonon = None
        self.atom_disp = atom_disp
        self.supercell_matrix = supercell_matrix
        self.fmax = fmax
        self.relax_structure = relax_structure
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min

    def calc(self, structure) -> dict:
        """
        Calculates thermal properties of
        input Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.
        Returns: {"thermal_properties":
                    {
                    "temp": list of temperatures,
                    "free_energy": list of Hemlholtz free energies at corresponding temperatures,
                    "entropy": list of entropies at corresponding temperatures,
                    "C_v": list of heat capacities at constant volume at corresponding temperatures,
                    }}.
        """
        phonon = self.get_phonon_from_calc(structure)
        thermal_property = _ThermalProperty(phonon)
        thermal_property.run(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        properties = thermal_property.get_thermal_properties()

        return {
            "thermal_properties": {
                "temp": np.array(properties[0]),
                "free_energy": np.array(properties[1]),
                "entropy": np.array(properties[2]),
                "C_v": np.array(properties[3]),
            },
        }

    def get_phonon_from_calc(self, structure):
        """
        Relaxes and processes the files given an MP ID.
        Returns a phonopy Phonon object with force constants produced.
        """
        if self.relax_structure:
            relaxer = RelaxCalc(self.calculator, fmax=self.fmax)
            structure = relaxer.calc(structure)["final_structure"]
        cell = get_phonopy_structure(structure)
        phonon = phonopy.Phonopy(cell, self.supercell_matrix)
        phonon.generate_displacements(distance=self.atom_disp)
        disp_supercells = phonon.supercells_with_displacements
        forces = [
            _calc_forces(self.calculator, supercell)
            for supercell in [phonon.supercell, *disp_supercells]
            if supercell is not None
        ]
        # parallel = Parallel(n_jobs=1)
        # forces = parallel(delayed(_calc_forces)(self.calculator, s) for s in structure_list)
        phonon.set_forces(forces[1:])
        phonon.produce_force_constants()
        self.phonon = phonon

        return phonon


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


class _ThermalProperty:
    """From phonondb script. Wrapper to call phonon functions."""

    def __init__(self, phonon):
        self._phonon = phonon  # Phonopy object
        self._thermal_properties = None

    def run(
        self,
        t_step=10,
        t_max=1000,
        t_min=0,
    ):
        """Runs thermal properties."""
        self._phonon.run_mesh()
        self._run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)
        return True

    def get_thermal_properties(self):
        """(temps(K), fe(kJ/mol), entropy(J/K/mol), cv(J/K/mol))."""
        return self._thermal_properties

    def _run_thermal_properties(self, t_step=10, t_max=1000, t_min=0):
        self._phonon.set_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)
        self._thermal_properties = self._phonon.get_thermal_properties()
