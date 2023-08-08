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




DEFAULT_SUPERCELL = ((2, 0, 0), (0, 2, 0), (0, 0, 2))


class PhononCalc(PropCalc):
    """Calculator for phonon properties."""

    def __init__(
                 self,
                 calculator: Calculator,
                 atom_disp=0.015,
                 supercell_matrix=DEFAULT_SUPERCELL,
                 fmax=0.1
                 ):
        """
        Args:
            calculator: ASE Calculator to use.
        """
        self.calculator = calculator
        self.phonon = None
        self.atom_disp = atom_disp
        self.supercell_matrix = supercell_matrix
        self.fmax = fmax

    def calc(self, structure) -> dict:
        """
        All PropCalc should implement a calc method that takes in a pymatgen structure
        and returns a dict.
        Note that the method can return more than one property.

        Args:
            structure: Pymatgen structure.

        Returns: {"prop name": value}
        """
        phonon = self.get_phonon_from_calc(structure)
        thermal_property = _ThermalProperty(phonon)
        thermal_property.run()
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
        model = _Model(self.calculator)
        relaxer = RelaxCalc(self.calculator, fmax=self.fmax)
        structure = relaxer.calc(structure)["final_structure"]
        cell = get_phonopy_structure(structure)
        phonon = phonopy.Phonopy(cell, self.supercell_matrix)
        phonon.generate_displacements(distance=self.atom_disp)
        disp_supercells = phonon.supercells_with_displacements
        structure_list = [get_pmg_structure(phonon.supercell)]
        for supercell in disp_supercells:
            if supercell is not None:
                structure_list.append(get_pmg_structure(supercell))
        forces = model.calculate_forces(structure_list)
        phonon.set_forces(forces[1:])
        phonon.produce_force_constants()
        self.phonon = phonon

        return phonon


class _Model:
    """Wrapper to call calculator on batches of pymatgen structures."""
    def __init__(self, calc):
        self.calc = calc
        self.adaptor = AseAtomsAdaptor()

    def calculate_forces(self, structures):
        """Calculate forces of list of pymatgen structures with calculator"""
        forces = []
        for i in range(len(structures)):
            atoms = self.adaptor.get_atoms(structures[i])
            atoms.calc = self.calc
            forces.append(atoms.get_forces())
        return forces


class _ThermalProperty:
    """From phonondb script. Wrapper to call phonon functions."""

    def __init__(self, phonon):
        self._phonon = phonon  # Phonopy object
        self._lattice = np.array(phonon.get_unitcell().get_cell().T, dtype="double")
        self._mesh = None
        self._thermal_properties = None

    def run(self, distance=100):
        """Runs thermal properties."""
        self._set_mesh(distance=distance)
        self._run_mesh_sampling()
        self._run_thermal_properties()
        return True

    def get_lattice(self):
        return self._lattice

    def get_mesh(self):
        return self._mesh

    def get_thermal_properties(self):
        """(temps(K), fe(kJ/mol), entropy(J/K/mol), cv(J/K/mol))."""
        return self._thermal_properties

    def _set_mesh(self, distance=100):
        self._mesh = k_len_to_mesh(distance, self._lattice)

    def _run_mesh_sampling(self):
        return self._phonon.set_mesh(self._mesh)

    def _run_thermal_properties(self, t_step=2, t_max=1500):
        self._phonon.set_thermal_properties(t_step=t_step, t_max=t_max)
        self._thermal_properties = self._phonon.get_thermal_properties()


def k_len_to_mesh(k_length, lattice):
    """
    From phonondb script:

    Convert length to mesh in k-point sampling.
    This conversion follows VASP manual.
    """
    rec_lattice = np.linalg.inv(lattice).T
    rec_lat_lengths = np.sqrt(np.diagonal(np.dot(rec_lattice.T, rec_lattice)))
    k_mesh = (rec_lat_lengths * k_length + 0.5).astype(int)
    return np.maximum(k_mesh, [1, 1, 1])
