"""Surface Energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from copy import deepcopy

import numpy as np

from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure, Molecule

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_ase_atoms, to_pmg_structure, to_pmg_molecule

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class AdsorptionCalc(PropCalc):
    """
    A class for performing surface energy calculations by generating and optionally
    relaxing bulk and slab structures. This facilitates materials science and
    computational chemistry workflows, enabling computations of surface properties
    for various crystal orientations and surface terminations.

    Detailed description of the class, its purpose, and usage.

    :ivar calculator: ASE Calculator used for energy and force evaluations. Interface
        to computational backends like DFT or classical force fields.
    :type calculator: Calculator
    :ivar relax_bulk: Indicates whether to relax the bulk structure, including its
        lattice parameters. Default is True.
    :type relax_bulk: bool
    :ivar relax_slab: Indicates whether to relax the slab structure, fixing its cell.
        Default is True.
    :type relax_slab: bool
    :ivar fmax: Force tolerance (in eV/Å) used during relaxation, controlling
        convergence. Default is 0.1.
    :type fmax: float
    :ivar optimizer: Optimizer to be used for structure relaxation. Can be a string
        referring to the optimizer's name (e.g., "BFGS") or an instance of an optimizer
        class. Default is "BFGS".
    :type optimizer: str | Optimizer
    :ivar max_steps: Maximum allowed steps for optimization during relaxation.
        Default is 500.
    :type max_steps: int
    :ivar relax_calc_kwargs: Additional parameters passed to the relaxation calculator
        for bulk and slab structures. Default is None.
    :type relax_calc_kwargs: dict | None
    :ivar final_bulk: Optimized bulk structure after relaxation. Initialized as None
        until relaxation is performed.
    :type final_bulk: Structure | None
    :ivar bulk_energy: Energy of the relaxed bulk structure. Initialized as None and
        updated after relaxation.
    :type bulk_energy: float | None
    :ivar n_bulk_atoms: Number of atoms in the bulk structure. Set after the bulk
        relaxation step.
    :type n_bulk_atoms: int | None
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        relax_adsorbate: bool = True,
        relax_slab: bool = True,
        relax_bulk: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "BFGS",
        max_steps: int = 500,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Constructor for initializing the SurfaceCalc with all parameters needed
        to generate and optionally relax bulk and slab structures.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param relax_bulk: Whether to relax the bulk structure, including its cell.
            Default is True.
        :type relax_bulk: bool, optional
        :param relax_slab: Whether to relax each slab structure (cell fixed).
            Default is True.
        :type relax_slab: bool, optional
        :param fmax: Force tolerance in eV/Å for relaxation. Default is 0.1.
        :type fmax: float, optional
        :param optimizer: The ASE optimizer to use. Can be a string (e.g. "BFGS") or
            an :class:`Optimizer` instance. Default is "BFGS".
        :type optimizer: str | Optimizer, optional
        :param max_steps: Maximum number of optimization steps for relaxation.
            Default is 500.
        :type max_steps: int, optional
        :param relax_calc_kwargs: Additional keyword arguments passed to the
            :class:`RelaxCalc` constructor for both bulk and slabs. Default is None.
        :type relax_calc_kwargs: dict | None, optional
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.relax_adsorbate = relax_adsorbate
        self.relax_bulk = relax_bulk
        self.relax_slab = relax_slab
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

        self.final_bulk: Structure | None = None
        self.bulk_energy: float | None = None
        self.n_bulk_atoms: int | None = None

    def calc_adslabs(
        self,
        adsorbate: Molecule | Atoms,
        # slab parameters
        bulk: Structure | Atoms,
        miller_index: tuple[int, int, int],
        adsorbate_energy: float | None = None,
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 20.0,
        inplane_supercell: tuple[int, int] = (1, 1),
        slab_gen_kwargs: dict | None = None,
        get_slabs_kwargs: dict | None = None,
        # adsorption parameters
        adsorption_sites: dict[str:tuple[float, float]] | str = "all",
        height: float = 0.9,
        mi_vec: tuple[float, float] | None = None,
        fixed_height: float = 5,
        find_adsorption_sites_args: dict | None = None,
        # other
        dry_run: bool = False,
        **kwargs
    ) -> list[dict[str, Any]] | dict:

        adslab_dict = {}
        bulk = to_pmg_structure(bulk)
        bulk = bulk.to_conventional()

        relaxer_bulk = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=self.relax_bulk,
            relax_atoms=self.relax_bulk,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )

        bulk_opt = relaxer_bulk.calc(bulk)

        if adsorbate_energy is not None and not self.relax_adsorbate:
            adsorbate_dict = {
                "adsorbate_energy": adsorbate_energy,
                "adsorbate": adsorbate,
                "final_adsorbate": adsorbate,
            }
        elif adsorbate_energy is not None and self.relax_adsorbate:
            raise ValueError(
                "Cannot provide adsorbate_energy and relax_adsorbate=True"
            )
        else:
            adsorbate_dict = self.calc_adsorbate(adsorbate)

        adsorbate_dict["final_adsorbate"] = to_pmg_molecule(
            adsorbate_dict["final_adsorbate"]
        )

        # Generally want the surface perpendicular to z
        if 'max_normal_search' not in slab_gen_kwargs:
            slab_gen_kwargs['max_normal_search'] = np.max(miller_index)

        slabgen = SlabGenerator(
            initial_structure=bulk_opt["final_structure"],
            miller_index=miller_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            
            **(slab_gen_kwargs or {}),
        )
        slab_dicts = [
            {
                "slab": slab.make_supercell((*inplane_supercell, 1)),
                "miller_index": miller_index,
            }
            for slab in slabgen.get_slabs(**(get_slabs_kwargs or {}))
        ]
        adslabs = []
        for slab_dict in slab_dicts:
            slab = slab_dict["slab"]

            if fixed_height is not None:
                minz = np.min(slab.cart_coords, axis=0)[2]
                for site in slab:
                    if site.coords[2] < minz + fixed_height:
                        site.properties["selective_dynamics"] = np.array(
                            [False] * 3
                        )
                    else:
                        site.properties["selective_dynamics"] = np.array(
                            [True] * 3
                        )

            slab_dict |= self.calc_slab(slab)
            slab_dict |= deepcopy(adsorbate_dict)

            asf = AdsorbateSiteFinder(
                slab_dict["final_slab"],
                height=height,
                mi_vec=mi_vec,
            )

            if isinstance(adsorption_sites, str):
                asf_adsites = asf.find_adsorption_sites(
                    **find_adsorption_sites_args or {}
                )
                if adsorption_sites == "all":
                    asf_adsites.pop("all")
                    adsites = {
                        s: asf_adsites[s] for s in asf_adsites.keys()
                    }
                else:
                    try:
                        adsites = {
                            adsorption_sites: asf_adsites[adsorption_sites]
                        }
                    except KeyError:
                        raise KeyError(
                            f"Provided sites: '{adsorption_sites}' must be one"
                            f" of {asf_adsites.keys()} "
                        )
            else:
                adsites = adsorption_sites

            for adsite in adsites:
                for adsite_idx, adsite_coord in enumerate(adsites[adsite]):
                    adslab = asf.add_adsorbate(
                        molecule=adsorbate_dict["final_adsorbate"],
                        ads_coord=adsite_coord,
                        # repeat=(*inplane_supercell, 1),
                    )
                    adslab_dict = {
                        "adslab": adslab,
                        "adsorption_site": adsite,
                        "adsorption_site_coord": adsite_coord,
                        "adsorption_site_index": adsite_idx,
                    }
                    adslab_dict |= deepcopy(slab_dict)
                    adslabs.append(adslab_dict)

        if dry_run:
            return adslabs
        else:
            return list(self.calc_many(adslabs, **kwargs))  # type:ignore[arg-type]

    def calc_adsorbate(
        self,
        adsorbate: Molecule | Atoms,
    ) -> dict[str, Any]:
        """
        Calculate the energy of the adsorbate, optionally relaxing it.

        :param adsorbate: The adsorbate structure to be calculated.
        :type adsorbate: Molecule | Atoms
        :return: A dictionary containing the adsorbate energy and final structure.
        :rtype: dict[str, Any]
        """

        relaxer = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_atoms=self.relax_adsorbate,
            relax_cell=False,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )

        adsorbate = to_ase_atoms(adsorbate)
        # Add 15 Å of vacuum in all directions for relaxation
        adsorbate.set_cell(
            np.max(adsorbate.positions, axis=0) - \
                np.min(adsorbate.positions, axis=0) + 15
        )
        adsorbate_opt = relaxer.calc(adsorbate)
        adsorbate_energy = adsorbate_opt["energy"]

        return {
            "adsorbate": adsorbate,
            "adsorbate_energy": adsorbate_energy,
            "final_adsorbate": adsorbate_opt["final_structure"],
        }

    def calc_slab(
        self,
        slab: Structure | Atoms,
    ) -> dict[str, Any]:
        """
        Calculate the energy of the slab, optionally relaxing it.

        :param slab: The slab structure to be calculated.
        :type slab: Structure | Atoms
        :return: A dictionary containing the slab energy and final structure.
        :rtype: dict[str, Any]
        """

        relaxer = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_atoms=self.relax_slab,
            relax_cell=False,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        slab_opt = relaxer.calc(slab)

        return {
            "slab": slab,
            "slab_energy": slab_opt["energy"],
            "slab_energy_per_atom": slab_opt["energy"] / len(slab_opt["final_structure"]),
            "final_slab": slab_opt["final_structure"],
        }
    
    def calc(
        self,
        structure: Structure | Atoms | dict[str, Any],
    ) -> dict[str, Any]:
        """
        """
        if not isinstance(structure, dict):
            slab_info = set(structure.keys()).intersection(("slab", "slab_energy_per_atom"))
            adsorbate_info = set(structure.keys()).intersection(("adsorbate", "adsorbate_energy"))
            if not (slab_info and adsorbate_info):
                raise ValueError(
                    "For adsorption calculations, structure must be dict with"
                    " keys ('adslab' and 'slab' and 'adsorbate') and optionally"
                    " ('slab_energy_per_atom' or 'slab_energy') and/or"
                    " 'adsorbate_energy'"
                )

        result_dict = structure.copy()

        if not ("adsorbate_energy" in structure):
            result_dict |= self.calc_adsorbate(structure["adsorbate"])

        if not ("slab_energy_per_atom" in structure):
            result_dict |= self.calc_slab(structure["slab"])

        adslab = structure["adslab"]

        n_adsorbate_atoms = len(structure["adsorbate"])
        n_slab_atoms = len(adslab) - n_adsorbate_atoms
        if len(adslab) != n_slab_atoms + n_adsorbate_atoms:
            raise ValueError(
                "The number of atoms in the adslab does not equal the sum of "
                "the slab and adsorbate atoms."
            )

        relaxer = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=False,
            relax_atoms=self.relax_slab,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        adslab_opt = relaxer.calc(adslab)
        final_adslab = adslab_opt["final_structure"]
        adslab_energy = adslab_opt["energy"]

        ads_energy = (
            adslab_energy
            - n_slab_atoms * result_dict["slab_energy_per_atom"]
            - result_dict["adsorbate_energy"]
        )

        return result_dict | {
            "final_adslab": final_adslab,
            "adslab_energy": adslab_energy,
            "adsorption_energy": ads_energy,
        }
