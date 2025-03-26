"""Calculator for elastic properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.analysis.elasticity.elastic import get_strain_state_dict

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class ElasticityCalc(PropCalc):
    """Calculator for elastic properties."""

    def __init__(
        self,
        calculator: Calculator,
        *,
        norm_strains: Sequence[float] | float = (-0.01, -0.005, 0.005, 0.01),
        shear_strains: Sequence[float] | float = (-0.06, -0.03, 0.03, 0.06),
        fmax: float = 0.1,
        symmetry: bool = False,
        relax_structure: bool = True,
        relax_deformed_structures: bool = False,
        use_equilibrium: bool = True,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE Calculator to use.
            norm_strains: single or multiple strain values to apply to each normal mode.
                Defaults to (-0.01, -0.005, 0.005, 0.01).
            shear_strains: single or multiple strain values to apply to each shear mode.
                Defaults to (-0.06, -0.03, 0.03, 0.06).
            fmax: maximum force in the relaxed structure (if relax_structure). Defaults to 0.1.
            symmetry: whether or not to use symmetry reduction. Defaults to False.
            relax_structure: whether to relax the provided structure with the given calculator.
                Defaults to True.
            relax_deformed_structures: whether to relax the atomic positions of the deformed/strained structures
                with the given calculator. Defaults to False.
            use_equilibrium: whether to use the equilibrium stress and strain. Ignored and set
                to True if either norm_strains or shear_strains has length 1 or is a float.
                Defaults to True.
            relax_calc_kwargs: Arguments to be passed to the RelaxCalc, if relax_structure is True.

        """
        self.calculator = calculator
        self.norm_strains = tuple(np.array([1]) * np.asarray(norm_strains))
        self.shear_strains = tuple(np.array([1]) * np.asarray(shear_strains))
        if len(self.norm_strains) == 0:
            raise ValueError("norm_strains is empty")
        if len(self.shear_strains) == 0:
            raise ValueError("shear_strains is empty")
        if 0 in self.norm_strains or 0 in self.shear_strains:
            raise ValueError("strains must be non-zero")
        self.relax_structure = relax_structure
        self.relax_deformed_structures = relax_deformed_structures
        self.fmax = fmax
        self.symmetry = symmetry
        if len(self.norm_strains) > 1 and len(self.shear_strains) > 1:
            self.use_equilibrium = use_equilibrium
        else:
            self.use_equilibrium = True
        self.relax_calc_kwargs = relax_calc_kwargs

    def calc(self, structure: Structure | dict[str, Any]) -> dict[str, Any]:
        """Calculates elastic properties of Pymatgen structure with units determined by the calculator,
        (often the stress_weight).

        Args:
            structure: Pymatgen structure.

        Returns: {
            elastic_tensor: Elastic tensor as a pymatgen ElasticTensor object (in eV/A^3),
            shear_modulus_vrh: Voigt-Reuss-Hill shear modulus based on elastic tensor (in eV/A^3),
            bulk_modulus_vrh: Voigt-Reuss-Hill bulk modulus based on elastic tensor (in eV/A^3),
            youngs_modulus: Young's modulus based on elastic tensor (in eV/A^3),
            residuals_sum: Sum of squares of all residuals in the linear fits of the
            calculation of the elastic tensor,
            structure: The equilibrium structure used for the computation.
        }
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure or self.relax_deformed_structures:
            relax_calc = RelaxCalc(self.calculator, fmax=self.fmax, **(self.relax_calc_kwargs or {}))
            if self.relax_structure:
                result |= relax_calc.calc(structure_in)
                structure_in = result["final_structure"]
            if self.relax_deformed_structures:
                relax_calc.relax_cell = False

        deformed_structure_set = DeformedStructureSet(
            structure_in,
            self.norm_strains,
            self.shear_strains,
            self.symmetry,
        )
        stresses = []
        for deformed_structure in deformed_structure_set:
            if self.relax_deformed_structures:
                deformed_structure_relaxed = relax_calc.calc(deformed_structure)["final_structure"]  # pyright:ignore (reportPossiblyUnboundVariable)
                atoms = deformed_structure_relaxed.to_ase_atoms()
            else:
                atoms = deformed_structure.to_ase_atoms()

            atoms.calc = self.calculator
            stresses.append(atoms.get_stress(voigt=False))

        strains = [Strain.from_deformation(deformation) for deformation in deformed_structure_set.deformations]
        atoms = structure_in.to_ase_atoms()
        atoms.calc = self.calculator
        elastic_tensor, residuals_sum = self._elastic_tensor_from_strains(
            strains,
            stresses,
            eq_stress=atoms.get_stress(voigt=False) if self.use_equilibrium else None,
        )
        return result | {
            "elastic_tensor": elastic_tensor,
            "shear_modulus_vrh": elastic_tensor.g_vrh,
            "bulk_modulus_vrh": elastic_tensor.k_vrh,
            "youngs_modulus": elastic_tensor.y_mod,
            "residuals_sum": residuals_sum,
            "structure": structure_in,
        }

    def _elastic_tensor_from_strains(
        self,
        strains: ArrayLike,
        stresses: ArrayLike,
        eq_stress: ArrayLike = None,
        tol: float = 1e-7,
    ) -> tuple[ElasticTensor, float]:
        """Slightly modified version of Pymatgen function
        pymatgen.analysis.elasticity.elastic.ElasticTensor.from_independent_strains;
        this is to give option to discard eq_stress,
        which (if the structure is relaxed) tends to sometimes be
        much lower than neighboring points.
        Also has option to return the sum of the squares of the residuals
        for all of the linear fits done to compute the entries of the tensor.
        """
        strain_states = [tuple(ss) for ss in np.eye(6)]
        ss_dict = get_strain_state_dict(strains, stresses, eq_stress=eq_stress, add_eq=self.use_equilibrium)
        c_ij = np.zeros((6, 6))
        residuals_sum = 0.0
        for ii in range(6):
            strain = ss_dict[strain_states[ii]]["strains"]
            stress = ss_dict[strain_states[ii]]["stresses"]
            for jj in range(6):
                fit = np.polyfit(strain[:, ii], stress[:, jj], 1, full=True)
                c_ij[ii, jj] = fit[0][0]
                residuals_sum += fit[1][0] if len(fit[1]) > 0 else 0.0
        elastic_tensor = ElasticTensor.from_voigt(c_ij)
        return elastic_tensor.zeroed(tol), residuals_sum
