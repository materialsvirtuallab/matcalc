"""Calculator for phonon properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS as write_force_constants
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


@dataclass
class PhononCalc(PropCalc):
    """Calculator for phonon properties.

    Args:
        calculator (Calculator): ASE Calculator to use.
        fmax (float): Max forces. This criterion is more stringent than for simple relaxation.
            Defaults to 0.1 (in eV/Angstrom)
        optimizer (str): Optimizer used for RelaxCalc.
        atom_disp (float): Atomic displacement (in Angstrom).
        supercell_matrix (ArrayLike): Supercell matrix to use. Defaults to 2x2x2 supercell.
        t_step (float): Temperature step (in Kelvin).
        t_max (float): Max temperature (in Kelvin).
        t_min (float): Min temperature (in Kelvin).
        relax_structure (bool): Whether to first relax the structure. Set to False if structures
            provided are pre-relaxed with the same calculator.
        relax_calc_kwargs (dict): Arguments to be passed to the RelaxCalc, if relax_structure is True.
        write_force_constants (bool | str | Path): Whether to save force constants. Pass string or Path
            for custom filename. Set to False for storage conservation. This file can be very large, be
            careful when doing high-throughput. Defaults to False.
        calculations.
        write_band_structure (bool | str | Path): Whether to calculate and save band structure
            (in yaml format). Defaults to False. Pass string or Path for custom filename.
        write_total_dos (bool | str | Path): Whether to calculate and save density of states
            (in dat format). Defaults to False. Pass string or Path for custom filename.
        write_phonon (bool | str | Path): Whether to save phonon object. Set to True to save
            necessary phonon calculation results. Band structure, density of states, thermal properties,
            etc. can be rebuilt from this file using the phonopy API via phonopy.load("phonon.yaml").
            Defaults to True. Pass string or Path for custom filename.
    """

    calculator: Calculator
    atom_disp: float = 0.015
    supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    t_step: float = 10
    t_max: float = 1000
    t_min: float = 0
    fmax: float = 0.1
    optimizer: str = "FIRE"
    relax_structure: bool = True
    relax_calc_kwargs: dict | None = None
    write_force_constants: bool | str | Path = False
    write_band_structure: bool | str | Path = False
    write_total_dos: bool | str | Path = False
    write_phonon: bool | str | Path = True

    def __post_init__(self) -> None:
        """Set default paths for where to save output files."""
        # map True to canonical default path, False to "" and Path to str
        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            phonon: Phonopy object with force constants produced
            thermal_properties:
                {
                    temperatures: list of temperatures in Kelvin,
                    free_energy: list of Helmholtz free energies at corresponding temperatures in kJ/mol,
                    entropy: list of entropies at corresponding temperatures in J/K/mol,
                    heat_capacity: list of heat capacities at constant volume at corresponding temperatures in J/K/mol,
                    The units are originally documented in phonopy.
                    See phonopy.Phonopy.run_thermal_properties()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/api_phonopy.py#L2591)
                    -> phonopy.phonon.thermal_properties.ThermalProperties.run()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L498)
                    -> phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_free_energy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L217)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_entropy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L233)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_heat_capacity()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L225)
                }
        }
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator, fmax=self.fmax, optimizer=self.optimizer, **(self.relax_calc_kwargs or {})
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]
        cell = get_phonopy_structure(structure_in)
        phonon = phonopy.Phonopy(cell, self.supercell_matrix)  # type: ignore[arg-type]
        phonon.generate_displacements(distance=self.atom_disp)
        disp_supercells = phonon.supercells_with_displacements
        phonon.forces = [  # type: ignore[assignment]
            _calc_forces(self.calculator, supercell)
            for supercell in disp_supercells  # type:ignore[union-attr]
            if supercell is not None
        ]
        phonon.produce_force_constants()
        phonon.run_mesh()
        phonon.run_thermal_properties(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        if self.write_force_constants:
            write_force_constants(phonon.force_constants, filename=self.write_force_constants)
        if self.write_band_structure:
            phonon.auto_band_structure(write_yaml=True, filename=self.write_band_structure)
        if self.write_total_dos:
            phonon.auto_total_dos(write_dat=True, filename=self.write_total_dos)
        if self.write_phonon:
            phonon.save(filename=self.write_phonon)
        return result | {"phonon": phonon, "thermal_properties": phonon.get_thermal_properties_dict()}


def _calc_forces(calculator: Calculator, supercell: PhonopyAtoms) -> ArrayLike:
    """Helper to compute forces on a structure.

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
