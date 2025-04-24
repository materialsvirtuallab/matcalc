"""Calculator for stability related properties."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from monty.serialization import loadfn

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Element, Species, Structure

ELEMENTAL_REFS_DIR = Path(__file__).parent / "elemental_refs"


class EnergeticsCalc(PropCalc):
    """
    Handles the computation of energetic properties such as formation energy per atom,
    cohesive energy per atom, and relaxed structures for input compositions. This class
    enables a streamlined setup for performing computational property calculations based
    on different reference data and relaxation configurations.

    :ivar calculator: The computational calculator used for numerical simulations and property
        evaluations.
    :type calculator: Calculator
    :ivar elemental_refs: Reference data dictionary or identifier for elemental properties.
        If a string ("MP-PBE", "MatPES-PBE" or "MatPES-r2SCAN"), loads default references;
        if a dictionary, uses custom provided data.
    :type elemental_refs: Literal["MP-PBE", "MatPES-PBE", "MatPES-r2SCAN"] | dict
    :ivar use_gs_reference: Whether to use DFT ground state data for energy computations
        when referencing elemental properties.
    :type use_gs_reference: bool
    :ivar relax_structure: Specifies whether to relax the input structures before property
        calculations. If True, relaxation is applied.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Optional keyword arguments for fine-tuning relaxation calculation
        settings or parameters.
    :type relax_calc_kwargs: dict | None
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        elemental_refs: Literal["MP-PBE", "MatPES-PBE", "MatPES-r2SCAN"] | dict = "MatPES-PBE",
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        perturb_distance: float | None = None,
        use_gs_reference: bool = False,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Initializes the class with the given calculator and optional configurations for
        elemental references, density functional theory (DFT) ground state reference, and
        options for structural relaxation.

        This constructor allows initializing essential components of the object, tailored
        for specific computational settings. The parameters include configurations for
        elemental references, an optional DFT ground state reference, and structural
        relaxation preferences.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param elemental_refs: Specifies the elemental references to be used. It can either be
            a predefined string identifier ("MP-PBE", "MatPES-PBE", "MatPES-r2SCAN") or a dictionary
            mapping elements to their energy references.
        :type elemental_refs: Literal["MP-PBE", "MatPES-PBE", "MatPES-r2SCAN"] | dict
        :ivar fmax: Force tolerance for convergence (eV/Å).
        :type fmax: float
        :ivar optimizer: Algorithm for performing the optimization.
        :type optimizer: Optimizer | str
        :ivar perturb_distance: Distance (Å) for random perturbation to break symmetry.
        :type perturb_distance: float | None
        :param use_gs_reference: Determines whether to use ground state energy as a reference.
            Defaults to False.
        :type use_gs_reference: bool
        :param relax_structure: Specifies if the structure should be relaxed before
            proceeding with calculations. Defaults to True.
        :type relax_structure: bool
        :param relax_calc_kwargs: Additional keyword arguments for the relaxation
            calculation. Can be a dictionary of settings or None. Defaults to None.
        :type relax_calc_kwargs: dict | None
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.fmax = fmax
        self.optimizer = optimizer
        self.perturb_distance = perturb_distance
        if isinstance(elemental_refs, str):
            self.elemental_refs = loadfn(ELEMENTAL_REFS_DIR / f"{elemental_refs}-Element-Refs.json.gz")
        else:
            self.elemental_refs = elemental_refs
        self.use_gs_reference = use_gs_reference
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
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
        structure_in: Structure | Atoms = result["final_structure"]
        relaxer = RelaxCalc(
            self.calculator,
            fmax=self.fmax,
            optimizer=self.optimizer,
            perturb_distance=self.perturb_distance,
            **(self.relax_calc_kwargs or {}),
        )
        if self.relax_structure:
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]
            energy = result["energy"]
        else:
            energy = run_pes_calc(structure_in, self.calculator).energy
        nsites = len(structure_in)

        def get_gs_energy(el: Element | Species) -> float:
            """
            Returns the ground state energy for a given element. If use_gs_reference is True, we use the
            pre-computed DFT energy.

            :param el: Element symbol.
            :return: Energy per atom.
            """
            if self.use_gs_reference:
                gs_energy = self.elemental_refs[el.symbol]["energy_per_atom"]
                return min(gs_energy) if isinstance(gs_energy, list) else gs_energy

            relaxer.perturb_distance = None
            structure = self.elemental_refs[el.symbol]["structure"]
            structure_list = [structure] if not isinstance(structure, list) else structure
            return min(
                r["energy"] / r["final_structure"].num_sites
                for r in list(relaxer.calc_many(structure_list))
                if r is not None
            )

        comp = to_pmg_structure(structure_in).composition
        e_form = energy - sum([get_gs_energy(el) * amt for el, amt in comp.items()])

        try:
            e_coh = energy - sum([self.elemental_refs[el.symbol]["energy_atomic"] * amt for el, amt in comp.items()])
            result = result | {"cohesive_energy_per_atom": e_coh / nsites}
        except KeyError:
            result = result | {"cohesive_energy_per_atom": None}

        return result | {
            "formation_energy_per_atom": e_form / nsites,
        }
