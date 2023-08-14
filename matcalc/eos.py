"""Calculators for EOS and associated properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.eos import BirchMurnaghan

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class EOSCalc(PropCalc):
    """Equation of state calculator."""

    def __init__(
        self,
        calculator: Calculator,
        optimizer: Optimizer | str = "FIRE",
        steps: int = 500,
        max_abs_strain: float = 0.1,
        n_points: int = 11,
        fmax: float = 0.1,
        relax_structure: bool = True,
    ) -> None:
        """
        Args:
            calculator: ASE Calculator to use.
            optimizer (str or ase Optimizer): The optimization algorithm. Defaults to "FIRE".
            steps (int): Max number of steps for relaxation.
            max_abs_strain (float): The maximum absolute strain applied to the structure. Defaults to 0.1, i.e.,
                10% strain.
            n_points (int): Number of points in which to compute the EOS. Defaults to 11.

            fmax (float): Max force for relaxation (of structure as well as atoms).
            relax_structure: Whether to first relax the structure. Set to False if structures provided are pre-relaxed
                with the same calculator.
        """
        self.calculator = calculator
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.n_points = n_points
        self.max_abs_strain = max_abs_strain
        self.fmax = fmax
        self.steps = steps

    def calc(self, structure: Structure) -> dict:
        """Fit the Birch-Murnaghan equation of state.

        Args:
            structure: pymatgen Structure object.

        Returns:
            {
            "EOS": {
                "volumes": volumes,
                "energies": energies
            },
            "bulk_modulus": bm.b0_GPa,
        }
        """
        if self.relax_structure:
            relaxer = RelaxCalc(self.calculator, optimizer=self.optimizer, fmax=self.fmax, steps=self.steps)
            structure = relaxer.calc(structure)["final_structure"]

        volumes, energies = [], []
        relaxer = RelaxCalc(
            self.calculator, optimizer=self.optimizer, fmax=self.fmax, steps=self.steps, relax_cell=False
        )
        for idx in np.linspace(-self.max_abs_strain, self.max_abs_strain, self.n_points):
            structure_strained = structure.copy()
            structure_strained.apply_strain([idx, idx, idx])
            result = relaxer.calc(structure_strained)
            volumes.append(result["final_structure"].volume)
            energies.append(result["energy"])
        bm = BirchMurnaghan(volumes=volumes, energies=energies)
        bm.fit()

        return {
            "EOS": {"volumes": volumes, "energies": energies},
            "bulk_modulus": bm.b0_GPa,
        }
