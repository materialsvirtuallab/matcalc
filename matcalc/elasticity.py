"""Calculator for phonon properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.io.ase import AseAtomsAdaptor

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


class ElasticityCalc(PropCalc):
    """Calculator for elastic properties."""

    def __init__(
        self,
        calculator: Calculator,
        norm_strains: float = 0.01,
        shear_strains: float = 0.01,
        fmax: float = 0.1,
        relax_structure: bool = True,
    ) -> None:
        """
        Args:
            calculator: ASE Calculator to use.

            fmax: maximum force in the relaxed structure (if relax_structure).
            norm_strains: strain value to apply to each normal mode.
            shear_strains: strain value to apply to each shear mode.
            relax_structure: whether to relax the provided structure with the given calculator.
        """
        self.calculator = calculator
        self.norm_strains = norm_strains
        self.shear_strains = shear_strains
        self.relax_structure = relax_structure
        self.fmax = fmax

    def calc(self, structure: Structure) -> dict:
        """
        Calculates elastic properties of Pymatgen structure with units determined by the calculator.

        Args:
            structure: Pymatgen structure.

        Returns: {
            elastic_tensor: Elastic tensor as a pymatgen ElasticTensor object,
            shear_modulus_vrh: Voigt-Reuss-Hill shear modulus based on elastic tensor,
            bulk_modulus_vrh: Voigt-Reuss-Hill bulk modulus based on elastic tensor,
            youngs_modulus: Young's modulus based on elastic tensor,
        }
        """
        if self.relax_structure:
            rcalc = RelaxCalc(self.calculator, fmax=self.fmax)
            structure = rcalc.calc(structure)["final_structure"]

        adaptor = AseAtomsAdaptor()
        deformed_structure_set = DeformedStructureSet(
            structure,
            [self.norm_strains],
            [self.shear_strains],
        )
        stresses = []
        for deformed_structure in deformed_structure_set:
            atoms = adaptor.get_atoms(deformed_structure)
            atoms.calc = self.calculator
            stresses.append(atoms.get_stress(voigt=False))

        strains = [Strain.from_deformation(deformation) for deformation in deformed_structure_set.deformations]
        atoms = adaptor.get_atoms(structure)
        atoms.calc = self.calculator
        elastic_tensor = ElasticTensor.from_independent_strains(
            strains, stresses, eq_stress=atoms.get_stress(voigt=False)
        )
        return {
            "elastic_tensor": elastic_tensor,
            "shear_modulus_vrh": elastic_tensor.g_vrh,
            "bulk_modulus_vrh": elastic_tensor.k_vrh,
            "youngs_modulus": elastic_tensor.y_mod,
        }
