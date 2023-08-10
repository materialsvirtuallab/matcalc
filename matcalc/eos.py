"""Calculators for EOS and associated properties."""

from __future__ import annotations

import numpy as np
from pymatgen.analysis.eos import BirchMurnaghan

from .base import PropCalc
from .relaxation import RelaxCalc


class EOSCalc(PropCalc):
    """Equation of state calculator."""

    def __init__(
        self,
        calculator,
        relax_structure: bool = True,
        fmax: float = 0.1,
        steps: int = 500,
        n_points: int = 11,
    ):
        """
        Args:
            calculator: ASE Calculator to use.
            relax_structure: Whether to first relax the structure. Set to False if structures provided are pre-relaxed
                with the same calculator.
            fmax (float): Max force for relaxation (of structure as well as atoms).
            steps (int): Max number of steps for relaxation.
            n_points (int): Number of points in which to compute the EOS. Defaults to 11.
        """
        self.calculator = calculator
        self.relax_structure = relax_structure
        self.n_points = n_points
        self.fmax = fmax
        self.steps = steps

    def calc(self, structure):
        """Fit the Birch-Murnaghan equation of state.

        Args:
            structure: A Structure object.

        Returns:
            {
            "EOS": {
                "volumes": volumes,
                "energies": energies
            },
            "K (GPa)": bm.b0_GPa,
        }
        """
        if self.relax_structure:
            relaxer = RelaxCalc(self.calculator, fmax=self.fmax, steps=self.steps)
            structure = relaxer.calc(structure)["final_structure"]

        volumes, energies = [], []
        relaxer = RelaxCalc(self.calculator, fmax=self.fmax, steps=self.steps, relax_cell=False)
        for idx in np.linspace(-0.1, 0.1, self.n_points):
            structure_strained = structure.copy()
            structure_strained.apply_strain([idx, idx, idx])
            result = relaxer.calc(structure_strained)
            volumes.append(result["final_structure"].volume)
            energies.append(result["energy"])
        bm = BirchMurnaghan(volumes=volumes, energies=energies)
        bm.fit()

        return {
            "EOS": {"volumes": volumes, "energies": energies},
            "K (GPa)": bm.b0_GPa,
        }
