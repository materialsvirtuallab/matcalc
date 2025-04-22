"""Calculators for EOS and associated properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.eos import BirchMurnaghan
from sklearn.metrics import r2_score

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class EOSCalc(PropCalc):
    """
    Performs equation of state (EOS) calculations using a specified ASE calculator.

    This class is intended to fit the Birch-Murnaghan equation of state, determine the
    bulk modulus, and provide other relevant physical properties of a given structure.
    The EOS calculation includes applying volumetric strain to the structure, optional
    initial relaxation of the structure, and evaluation of energies and volumes
    corresponding to the applied strain.

    :ivar calculator: The ASE Calculator used for the calculations.
    :type calculator: Calculator
    :ivar optimizer: Optimization algorithm. Defaults to "FIRE".
    :type optimizer: Optimizer | str
    :ivar relax_structure: Indicates if the structure should be relaxed initially. Defaults to True.
    :type relax_structure: bool
    :ivar n_points: Number of strain points for the EOS calculation. Defaults to 11.
    :type n_points: int
    :ivar max_abs_strain: Maximum absolute volumetric strain. Defaults to 0.1 (10% strain).
    :type max_abs_strain: float
    :ivar fmax: Maximum force tolerance for relaxation. Defaults to 0.1 eV/Å.
    :type fmax: float
    :ivar max_steps: Maximum number of optimization steps during relaxation. Defaults to 500.
    :type max_steps: int
    :ivar relax_calc_kwargs: Additional keyword arguments for relaxation calculations. Defaults to None.
    :type relax_calc_kwargs: dict | None
    """

    def __init__(
        self,
        calculator: Calculator | str,
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
        Constructor for initializing the data and configurations necessary for a
        calculation and optimization process. This class enables the setup of
        simulation parameters, structural relaxation options, and optimizations
        with specified constraints and tolerances.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param optimizer: The optimization algorithm used for structural relaxations
            or energy minimizations. Can be an optimizer object or the string name
            of the algorithm. Default is "FIRE".
        :type optimizer: Optimizer | str, optional
        :param max_steps: The maximum number of steps allowed during the optimization
            or relaxation process. Default is 500.
        :type max_steps: int, optional
        :param max_abs_strain: The maximum allowable absolute strain for relaxation
            processes. Default is 0.1.
        :type max_abs_strain: float, optional
        :param n_points: The number of points or configurations evaluated during
            the simulation or calculation process. Default is 11.
        :type n_points: int, optional
        :param fmax: The force convergence criterion, specifying the maximum force
            threshold (per atom) for stopping relaxations. Default is 0.1.
        :type fmax: float, optional
        :param relax_structure: A flag indicating whether structural relaxation
            should be performed before proceeding with further steps. Default is True.
        :type relax_structure: bool, optional
        :param relax_calc_kwargs: Additional keyword arguments to customize the
            relaxation calculation process. Default is None.
        :type relax_calc_kwargs: dict | None, optional
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.n_points = n_points
        self.max_abs_strain = max_abs_strain
        self.fmax = fmax
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """
        Performs energy-strain calculations using Birch-Murnaghan equations of state to extract
        equation of state properties such as bulk modulus and R-squared score of the fit.

        This function calculates properties of a material system under strain, specifically
        its volumetric energy response produced by applying incremental strain, then fits
        the Birch-Murnaghan equation of state to the calculated energy and volume data.
        Optionally, a relaxation is applied to the structure between calculations of its
        strained configurations.

        :param structure: Input structure for calculations. Can be a `Structure` object or
            a dictionary representation of its atomic configuration and parameters.
        :return: A dictionary containing results of the calculations, including relaxed
            structures under conditions of strain, energy-volume data, Birch-Murnaghan
            bulk modulus (in GPa), and R-squared fit of the Birch-Murnaghan model to the
            data.
        """
        result = super().calc(structure)
        structure_in: Structure = to_pmg_structure(result["final_structure"])

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

        temp_structure = to_pmg_structure(structure_in).copy()
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
