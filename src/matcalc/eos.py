"""Calculators for EOS and associated properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.eos import BirchMurnaghan
from sklearn.metrics import r2_score

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from typing import Any

    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class EOSCalc(PropCalc):
    """Equation of state calculator."""

    def __init__(
        self,
        calculator: Calculator,
        *,
        optimizer: Optimizer | str = "FIRE",
        max_steps: int = 500,
        max_abs_strain: float = 0.1,
        n_points: int = 11,
        fmax: float = 0.1,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE Calculator to use.
            optimizer (str | ase Optimizer): The optimization algorithm. Defaults to "FIRE".
            max_steps (int): Max number of steps for relaxation. Defaults to 500.
            max_abs_strain (float): The maximum absolute strain applied to the structure. Defaults to 0.1 (10% strain).
            n_points (int): Number of points in which to compute the EOS. Defaults to 11.
            fmax (float): Max force for relaxation (of structure as well as atoms).
            relax_structure: Whether to first relax the structure. Set to False if structures provided are pre-relaxed
                with the same calculator. Defaults to True.
            relax_calc_kwargs: Arguments to be passed to the RelaxCalc, if relax_structure is True.
        """
        self.calculator = calculator
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.n_points = n_points
        self.max_abs_strain = max_abs_strain
        self.fmax = fmax
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

    def calc(self, structure: Structure | dict[str, Any]) -> dict:
        """Fit the Birch-Murnaghan equation of state.

        Args:
            structure: pymatgen Structure object.

        Returns: {
            eos: {
                volumes: tuple[float] in Angstrom^3,
                energies: tuple[float] in eV,
            },
            bulk_modulus_bm: Birch-Murnaghan bulk modulus in GPa.
            r2_score_bm: R squared of Birch-Murnaghan fit of energies predicted by model to help detect erroneous
            calculations. This value should be at least around 1 - 1e-4 to 1 - 1e-5.
        }
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator,
                optimizer=self.optimizer,
                fmax=self.fmax,
                max_steps=self.max_steps,
                **(self.relax_calc_kwargs or {}),
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        volumes, energies = [], []
        relaxer = RelaxCalc(
            self.calculator,
            optimizer=self.optimizer,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=False,
            **(self.relax_calc_kwargs or {}),
        )

        temp_structure = structure_in.copy()
        for idx in np.linspace(-self.max_abs_strain, self.max_abs_strain, self.n_points)[self.n_points // 2 :]:
            structure_strained = temp_structure.copy()
            structure_strained.apply_strain(
                (((1 + idx) ** 3 * structure_in.volume) / (structure_strained.volume)) ** (1 / 3) - 1
            )
            result = relaxer.calc(structure_strained)
            volumes.append(result["final_structure"].volume)
            energies.append(result["energy"])
            temp_structure = result["final_structure"]

        for idx in np.flip(np.linspace(-self.max_abs_strain, self.max_abs_strain, self.n_points)[: self.n_points // 2]):
            structure_strained = structure_in.copy()
            structure_strained.apply_strain(
                (((1 + idx) ** 3 * structure_in.volume) / (structure_strained.volume)) ** (1 / 3) - 1
            )
            result = relaxer.calc(structure_strained)
            volumes.append(result["final_structure"].volume)
            energies.append(result["energy"])
            temp_structure = result["final_structure"]

        bm = BirchMurnaghan(volumes=volumes, energies=energies)
        bm.fit()

        volumes, energies = map(
            list, zip(*sorted(zip(volumes, energies, strict=True), key=lambda i: i[0]), strict=False)
        )

        return result | {
            "eos": {"volumes": volumes, "energies": energies},
            "bulk_modulus_bm": bm.b0_GPa,
            "r2_score_bm": r2_score(energies, bm.func(volumes)),
        }
