"""Calculator for stability related properties."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from monty.serialization import loadfn
from pymatgen.io.ase import AseAtomsAdaptor

from .base import PropCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from typing import Any

    from ase.calculators.calculator import Calculator
    from pymatgen.core import Element, Species, Structure

ELEMENTAL_REFS_DIR = Path(__file__).parent / "elemental_refs"


class EnergeticsCalc(PropCalc):
    """Calculator for energetic properties."""

    def __init__(
        self,
        calculator: Calculator,
        *,
        elemental_refs: Literal["MatPES-PBE", "MatPES-r2SCAN"] | dict = "MatPES-PBE",
        use_dft_gs_reference: bool = False,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Initialize the class with the required computational parameters to set up properties
        and configurations. This class is used to perform calculations and provides an interface
        to manage computational settings such as calculator setup, elemental references, ground
        state relaxation, and additional calculation parameters.

        :param calculator: The computational calculator object implementing specific calculation
            protocols or methods for performing numerical simulations.
        :type calculator: Calculator

        :param elemental_refs: Specifies the elemental reference data source. It can either be a
            predefined identifier ("MatPES-PBE" or "MatPES-r2SCAN") to load default references or,
            alternatively, it can be a dictionary directly providing custom reference data. The dict should be of the
            format {element_symbol: {"structure": structure_object, "energy_per_atom": energy_per_atom,
            "energy_atomic": energy_atomic}}
        :type elemental_refs: Literal["MatPES-PBE", "MatPES-r2SCAN"] | dict

        :param use_dft_gs_reference: Whether to use the ground state reference from DFT
            calculations for energetics or other property computations.
        :type use_dft_gs_reference: bool

        :param relax_calc_kwargs: Optional dictionary containing additional keyword arguments
            for customizing the configurations and execution of the relaxation calculations.
        :type relax_calc_kwargs: dict | None
        """
        self.calculator = calculator
        if isinstance(elemental_refs, str):
            self.elemental_refs = loadfn(ELEMENTAL_REFS_DIR / f"{elemental_refs}-Element-Refs.json.gz")
        else:
            self.elemental_refs = elemental_refs
        self.use_dft_gs_reference = use_dft_gs_reference
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs

    def calc(self, structure: Structure | dict[str, Any]) -> dict[str, Any]:
        """
        Calculates the formation energy per atom, cohesive energy per atom, and final
        relaxed structure for a given input structure using a relaxation calculation
        and reference elemental data. This function also optionally utilizes DFT
        ground-state references for formation energy calculations. The cohesive energy is always referenced to the
        DFT atomic energies.

        :param structure: The input structure to be relaxed and analyzed.
        :type structure: Structure
        :return: A dictionary containing the formation energy per atom, cohesive
                 energy per atom, and the final relaxed structure.
        :rtype: dict[str, Any]
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]
        relaxer = RelaxCalc(
            self.calculator,
            **(self.relax_calc_kwargs or {}),
        )
        if self.relax_structure:
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        atoms = AseAtomsAdaptor.get_atoms(structure_in)
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        nsites = len(structure_in)

        def get_gs_energy(el: Element | Species) -> float:
            """
            Returns the ground state energy for a given element. If use_dft_gs_reference is True, we use the
            pre-computed DFT energy.

            :param el: Element symbol.
            :return: Energy per atom.
            """
            if self.use_dft_gs_reference:
                return self.elemental_refs[el.symbol]["energy_per_atom"]

            eldata = relaxer.calc(self.elemental_refs[el.symbol]["structure"])
            return eldata["energy"] / eldata["final_structure"].num_sites

        comp = structure_in.composition
        e_form = energy - sum([get_gs_energy(el) * amt for el, amt in comp.items()])

        e_coh = energy - sum([self.elemental_refs[el.symbol]["energy_atomic"] * amt for el, amt in comp.items()])

        return result | {
            "formation_energy_per_atom": e_form / nsites,
            "cohesive_energy_per_atom": e_coh / nsites,
        }
