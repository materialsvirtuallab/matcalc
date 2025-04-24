"""Calculator for elastic properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.analysis.elasticity.elastic import get_strain_state_dict

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class ElasticityCalc(PropCalc):
    """
    Class for calculating elastic properties of a material. This includes creating
    an elastic tensor, shear modulus, bulk modulus, and other related properties with
    the help of strain and stress analyses. It leverages the provided ASE Calculator
    for computations and supports relaxation of structures when necessary.

    :ivar calculator: The ASE Calculator used for performing computations.
    :type calculator: Calculator
    :ivar norm_strains: Sequence of normal strain values to be applied.
    :type norm_strains: Sequence[float] | float
    :ivar shear_strains: Sequence of shear strain values to be applied.
    :type shear_strains: Sequence[float] | float
    :ivar fmax: Maximum force tolerated for structure relaxation.
    :type fmax: float
    :ivar symmetry: Whether to apply symmetry reduction techniques during calculations.
    :type symmetry: bool
    :ivar relax_structure: Whether the initial structure should be relaxed before applying strains.
    :type relax_structure: bool
    :ivar relax_deformed_structures: Whether to relax atomic positions in deformed/strained structures.
    :type relax_deformed_structures: bool
    :ivar use_equilibrium: Whether to use equilibrium stress and strain in calculations.
    :type use_equilibrium: bool
    :ivar relax_calc_kwargs: Additional arguments for relaxation calculations.
    :type relax_calc_kwargs: dict | None
    """

    def __init__(
        self,
        calculator: Calculator | str,
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
        Initializes the class with parameters to construct normalized and shear strain values
        and control relaxation behavior for structures. Validates input parameters to ensure
        appropriate constraints are maintained.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param norm_strains: Sequence of normalized strain values applied during deformation.
            Can also be a single float. Must not be empty or contain zero.
        :param shear_strains: Sequence of shear strain values applied during deformation.
            Can also be a single float. Must not be empty or contain zero.
        :param fmax: Maximum force magnitude tolerance for relaxation. Default is 0.1.
        :param symmetry: Boolean flag to enforce symmetry in deformation. Default is False.
        :param relax_structure: Boolean flag indicating if the structure should be relaxed before
            applying strains. Default is True.
        :param relax_deformed_structures: Boolean flag indicating if the deformed structures
            should be relaxed. Default is False.
        :param use_equilibrium: Boolean flag indicating if equilibrium conditions should be used for
            calculations. Automatically enabled if multiple normal and shear strains are provided.
        :param relax_calc_kwargs: Optional dictionary containing keyword arguments for structure
            relaxation calculations.
        """
        self.calculator = calculator  # type: ignore[assignment]
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

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Performs a calculation to determine the elastic tensor and related elastic
        properties. It involves multiple steps such as optionally relaxing the input
        structure, generating deformed structures, calculating stresses, and evaluating
        elastic properties. The method supports equilibrium stress computation and various
        relaxations depending on configuration.

        :param structure:
            The input structure which can either be an instance of `Structure` or
            a dictionary containing structural data.
        :return:
            A dictionary containing the calculation results that include:
            - `elastic_tensor`: The computed elastic tensor of the material.
            - `shear_modulus_vrh`: Shear modulus obtained from the elastic tensor
              using the Voigt-Reuss-Hill approximation.
            - `bulk_modulus_vrh`: Bulk modulus calculated using the Voigt-Reuss-Hill
              approximation.
            - `youngs_modulus`: Young's modulus derived from the elastic tensor.
            - `residuals_sum`: The residual sum from the elastic tensor fitting.
            - `structure`: The (potentially relaxed) final structure after calculations.
        """
        result = super().calc(structure)
        structure_in: Structure | Atoms = result["final_structure"]

        if self.relax_structure or self.relax_deformed_structures:
            relax_calc = RelaxCalc(self.calculator, fmax=self.fmax, **(self.relax_calc_kwargs or {}))
            if self.relax_structure:
                result |= relax_calc.calc(structure_in)
                structure_in = result["final_structure"]
            if self.relax_deformed_structures:
                relax_calc.relax_cell = False

        deformed_structure_set = DeformedStructureSet(
            to_pmg_structure(structure_in),
            self.norm_strains,
            self.shear_strains,
            self.symmetry,
        )
        stresses = []
        for deformed_structure in deformed_structure_set:
            if self.relax_deformed_structures:
                deformed_relaxed = relax_calc.calc(deformed_structure)["final_structure"]  # pyright:ignore (reportPossiblyUnboundVariable)
                sim = run_pes_calc(deformed_relaxed, self.calculator)
            else:
                sim = run_pes_calc(deformed_structure, self.calculator)
            stresses.append(sim.stress)

        strains = [Strain.from_deformation(deformation) for deformation in deformed_structure_set.deformations]
        sim = run_pes_calc(structure_in, self.calculator)
        elastic_tensor, residuals_sum = self._elastic_tensor_from_strains(
            strains,
            stresses,
            eq_stress=sim.stress if self.use_equilibrium else None,
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
        """
        Compute the elastic tensor from given strain and stress data using least-squares
        fitting.

        This function calculates the elastic constants from strain-stress relations,
        using a least-squares fitting procedure for each independent component of stress
        and strain tensor pairs. An optional equivalent stress array can be supplied.
        Residuals from the fitting process are accumulated and returned alongside the
        elastic tensor. The elastic tensor is zeroed according to the given tolerance.

        :param strains:
            Strain data array-like, representing different strain states.
        :param stresses:
            Stress data array-like corresponding to the given strain states.
        :param eq_stress:
            Optional array-like, equivalent stress values for equilibrium stress states.
            Defaults to None.
        :param tol:
            A float representing the tolerance threshold used for zeroing the elastic
            tensor. Defaults to 1e-7.
        :return:
            A tuple consisting of:
              - ElasticTensor object: The computed and zeroed elastic tensor in Voigt
                notation.
              - float: The summed residuals from least-squares fittings across all
                tensor components.
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
