"""Calculator for phonon-phonon interaction and related properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from phono3py import Phono3py
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


@dataclass
class Phonon3Calc(PropCalc):
    """Calculator for phonon-phonon interaction and related properties.

    Args:
        calculator (Calculator): ASE Calculator to use.
        fmax (float): Max forces. This criterion is more stringent than for simple relaxation.
            Defaults to 0.1 (in eV/Angstrom)
        optimizer (str): Optimizer used for RelaxCalc.
        supercell_matrix (ArrayLike): Supercell matrix to use. Defaults to 2x2x2 supercell.
        mesh_numbers (ArrayLike): Numbers of sampling mesh along reciprocal axes. Defaults to (11, 11, 11).
        t_step (float): Temperature step (in Kelvin).
        t_max (float): Max temperature (in Kelvin).
        t_min (float): Min temperature (in Kelvin).
        relax_structure (bool): Whether to first relax the structure. Set to False if structures
            provided are pre-relaxed with the same calculator.
        relax_calc_kwargs (dict): Arguments to be passed to the RelaxCalc, if relax_structure is True.
        write_phonon3 (bool | str | Path): Whether to save Phono3py object. Set to True to save
            necessary phonon calculation results. Thermal conductivity, etc. can be rebuilt from this
            file using the phono3py API via phono3py.load("phonon3.yaml").
            Defaults to True. Pass string or Path for custom filename.
        write_kappa (bool): Whether to save thermal conductivity related properties. Defaults to True.
    """

    calculator: Calculator
    supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    mesh_numbers: ArrayLike = (11, 11, 11)
    t_step: float = 10
    t_max: float = 1000
    t_min: float = 0
    fmax: float = 0.1
    optimizer: str = "FIRE"
    relax_structure: bool = True
    relax_calc_kwargs: dict | None = None
    write_phonon3: bool | str | Path = True
    write_kappa: bool = True

    def __post_init__(self) -> None:
        """Set default paths for where to save output files."""
        # map True to canonical default path, False to "" and Path to str
        for key, val, default_path in (("write_phonon3", self.write_phonon3, "phonon3.yaml"),):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure) -> dict:
        """Calculates phonon-phonon interaction and related properties.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            "phonon3": Phono3py object with force constants produced,
            "temperatures": list of temperatures in ascending order (in Kelvin),
            "thermal_conductivity_tensor": list of thermal conductivity tensor at corresponding temperatures
            (in Watts/meter/Kelvin),
        }
        """
        temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator, fmax=self.fmax, optimizer=self.optimizer, **(self.relax_calc_kwargs or {})
            )
            structure = relaxer.calc(structure)["final_structure"]
        cell = get_phonopy_structure(structure)
        ph3 = Phono3py(cell, supercell_matrix=self.supercell_matrix)
        ph3.generate_displacements()
        disp_supercells = ph3.supercells_with_displacements
        static_calc = RelaxCalc(self.calculator, relax_atoms=False, relax_cell=False)
        ph3.forces = [
            static_calc.calc(get_pmg_structure(supercell))["forces"]
            for supercell in disp_supercells
            if supercell is not None
        ]
        ph3.produce_fc3()
        ph3.mesh_numbers = self.mesh_numbers
        ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(temperatures=temperatures, write_kappa=self.write_kappa)
        if self.write_phonon3:
            ph3.save(filename=self.write_phonon3)
        return {
            "phonon3": ph3,
            "temperatures": temperatures,
            "thermal_conductivity_tensor": ph3.thermal_conductivity.kappa,
        }
