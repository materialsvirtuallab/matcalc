"""Surface Energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatgen.core.surface import SlabGenerator

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class SurfaceCalc(PropCalc):
    """
        A class to perform surface-energy calculations on a bulk structure.

    Steps:
      1. Relax (or single-point) the bulk structure to obtain its energy.
      2. Generate slabs corresponding to a specified Miller index.
      3. Optionally relax each slab and evaluate its total energy.
      4. Compute the surface energy from the difference between slab and bulk
         (per-atom) energies, divided by the slab surface area. For a symmetric
         slab, it is assumed there are 2 identical surfaces.

    Typical usage:
      1. Instantiate with all relevant parameters (e.g. ``miller_index``,
         ``min_slab_size``, ``relax_bulk``).
      2. Call :meth:`calc_slabs` on a Pymatgen bulk :class:`Structure`.
         This returns a dictionary of slab structures and internally stores
         bulk data.
      3. Pass that slab dictionary to :meth:`calc` to get a final dictionary
         of surface energies for each slab.

    Example:
        >> my_calc = SomeASECalculator(...)
        >> surf_calc = SurfaceCalc(calculator=my_calc)
        >> results = surf_calc.calc_slabs(bulk_structure, miller_index=(1,0,0), ...)
        >> print(results)

    :ivar calculator: The ASE Calculator object used for energy/force evaluations.
    :type calculator: Calculator
    :ivar relax_bulk: Whether to relax the bulk structure (cell+atomic DOFs).
    :type relax_bulk: bool
    :ivar relax_slab: Whether to relax each slab structure (atomic DOFs).
    :type relax_slab: bool
    :ivar fmax: Force tolerance (eV/Å) for relaxation.
    :type fmax: float
    :ivar optimizer: The ASE optimizer to use, e.g. "BFGS" or an :class:`Optimizer`.
    :type optimizer: str | Optimizer
    :ivar max_steps: Maximum number of optimization steps.
    :type max_steps: int
    :ivar relax_calc_kwargs: Additional keyword arguments passed to :class:`RelaxCalc`.
    :type relax_calc_kwargs: dict | None
    """

    def __init__(
        self,
        calculator: Calculator,
        *,
        relax_bulk: bool = True,
        relax_slab: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "BFGS",
        max_steps: int = 500,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Constructor for initializing the SurfaceCalc with all parameters needed
        to generate and optionally relax bulk and slab structures.

        :param calculator: ASE Calculator for energy/force evaluations.
        :type calculator: Calculator
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
        self.calculator = calculator
        self.relax_bulk = relax_bulk
        self.relax_slab = relax_slab
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

        self.final_bulk: Structure | None = None
        self.bulk_energy: float | None = None
        self.n_bulk_atoms: int | None = None

    def calc_slabs(
        self,
        bulk: Structure,
        miller_index: tuple[int, int, int] = (1, 0, 0),
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 20.0,
        symmetrize: bool = True,  # noqa: FBT001,FBT002
        inplane_supercell: tuple[int, int] = (1, 1),
        slab_gen_kwargs: dict | None = None,
        get_slabs_kwargs: dict | None = None,
        **kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Calculates slabs based on a given bulk structure and generates a set of slabs
        using specified parameters. The function leverages slab generation tools and
        defines the in-plane supercell, symmetry, and optimizes the bulk structure
        prior to slab generation. This is useful for surface calculations in materials
        science and computational chemistry.

        :param bulk: The bulk structure from which slabs are generated. Must
            be an instance of pymatgen's Structure class.
        :type bulk: Structure
        :param miller_index: The Miller index used for generating the slabs. Defines the
            crystallographic orientation. Defaults to (1, 0, 0).
        :type miller_index: tuple[int, int, int]
        :param min_slab_size: Minimum thickness of the slab in angstroms. Defines
            the slab's physical size. Defaults to 10.0.
        :type min_slab_size: float
        :param min_vacuum_size: Minimum vacuum layer thickness in angstroms to ensure
            surface isolation. Defaults to 20.0.
        :type min_vacuum_size: float
        :param symmetrize: A boolean indicating whether or not to symmetrize the slab
            structure based on the bulk symmetry. Defaults to True.
        :type symmetrize: bool
        :param inplane_supercell: Tuple defining the scaling factors for creating
            the supercell in the plane of the slab. Defaults to (1, 1).
        :type inplane_supercell: tuple[int, int]
        :param slab_gen_kwargs: Optional dictionary of additional arguments to
            customize the slab generation process.
        :type slab_gen_kwargs: dict | None
        :param get_slabs_kwargs: Optional dictionary of additional arguments passed
            to the `get_slabs` method for further customization.
        :type get_slabs_kwargs: dict | None
        :param kwargs: Additional keyword arguments passed through to calc_many method.
        :type kwargs: dict
        :return: A dictionary containing the generated slab information, including the
            slab structure, bulk energy per atom, optimized bulk structure, and Miller
            index.
        :rtype: dict[str, Any]
        """
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

        slabgen = SlabGenerator(
            initial_structure=bulk,
            miller_index=miller_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            **(slab_gen_kwargs or {}),
        )
        slabs = [
            {
                "slab": slab.make_supercell((*inplane_supercell, 1)),
                "bulk_energy_per_atom": bulk_opt["energy"] / len(bulk),
                "final_bulk": bulk_opt["final_structure"],
                "miller_index": miller_index,
            }
            for slab in slabgen.get_slabs(symmetrize=symmetrize, **(get_slabs_kwargs or {}))
        ]

        return list(self.calc_many(slabs, **kwargs))  # type:ignore[arg-type]

    def calc(
        self,
        structure: Structure | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Performs surface energy calculation for a given structure dictionary. The function handles
        the relaxation of both bulk and slab structures when necessary and computes the surface energy
        using the slab's relaxed energy, number of atoms, bulk energy per atom, and surface area.

        :param structure: Dictionary containing information about the bulk and slab structures.
                          It must have the format:
                          {'slab': slab_structure, 'bulk': bulk_structure}
                          or
                          {'slab': slab_structure, 'bulk_energy_per_atom': energy}.
        :type structure: Structure | dict[str, Any]

        :return: A dictionary containing the updated structure data, including fields like
                 'slab', 'final_slab', 'slab_energy', 'surface_energy', and possibly updated
                 'bulk_energy_per_atom' and 'final_bulk'.
        :rtype: dict[str, Any]
        """
        if not (isinstance(structure, dict) and set(structure.keys()).intersection(("bulk", "bulk_energy_per_atom"))):
            raise ValueError(
                "For surface calculations, structure must be a dict in one of the following formats: "
                "{'slab': slab_struct, 'bulk': bulk_struct} or {'slab': slab_struct, 'bulk_energy': energy}."
            )

        result_dict = structure.copy()

        if "bulk_energy_per_atom" in structure:
            bulk_energy_per_atom = structure["bulk_energy_per_atom"]
        else:
            relaxer = RelaxCalc(
                calculator=self.calculator,
                fmax=self.fmax,
                max_steps=self.max_steps,
                relax_cell=self.relax_bulk,
                relax_atoms=self.relax_bulk,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
            )
            bulk_opt = relaxer.calc(structure["bulk"])
            bulk_energy_per_atom = bulk_opt["energy"] / len(structure["final_bulk"])
            result_dict["bulk_energy_per_atom"] = bulk_energy_per_atom
            result_dict["final_bulk"] = structure["final_bulk"]

        slab = structure["slab"]
        relaxer = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=False,
            relax_atoms=self.relax_slab,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        slab_opt = relaxer.calc(slab)
        final_slab = slab_opt["final_structure"]
        slab_energy = slab_opt["energy"]

        # Compute surface energy
        # Assuming two surfaces: (E_slab - N_slab_atoms * E_bulk_per_atom) / (2 * area)
        area = slab.surface_area
        n_slab_atoms = len(final_slab)
        gamma = (slab_energy - n_slab_atoms * bulk_energy_per_atom) / (2 * area)

        return result_dict | {
            "slab": slab,
            "final_slab": final_slab,
            "slab_energy": slab_energy,
            "surface_energy": gamma,
        }
