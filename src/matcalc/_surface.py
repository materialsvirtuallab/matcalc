"""Surface Energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer


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
        bulk_struct: Structure,
        miller_index: tuple[int, int, int] = (1, 0, 0),
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 20.0,
        symmetrize: bool = True,  # noqa: FBT001,FBT002
        inplane_supercell: tuple[int, int] = (1, 1),
        slab_gen_kwargs: dict | None = None,
        get_slabs_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """
        Relax (or single-point calculate) a bulk structure, then generate slab structures
        using the specified Miller index. Stores the resulting bulk data internally:
        ``final_bulk``, ``bulk_energy``.

        :param bulk_struct: A Pymatgen bulk structure. It is converted to its conventional
            cell before relaxation or single-point calculation.
        :type bulk_struct: Structure
        :param miller_index: Miller index for the slab surfaces (e.g., (1, 0, 0)).
            Default is (1, 0, 0).
        :type miller_index: tuple[int,int,int], optional
        :param min_slab_size: Thickness of the slab in Å. Default is 10.0 Å.
        :type min_slab_size: float, optional
        :param min_vacuum_size: Thickness of the vacuum region in Å. Default is 20.0 Å.
        :type min_vacuum_size: float, optional
        :param symmetrize: Whether to produce symmetrically distinct slabs. Default True.
        :type symmetrize: bool, optional
        :param inplane_supercell: Multiplicative factors for in-plane supercell expansion
            of the slabs. Default is (1, 1).
        :type inplane_supercell: tuple[int,int], optional
        :param slab_gen_kwargs: Additional kwargs passed to :class:`SlabGenerator`.
            Default is None.
        :type slab_gen_kwargs: dict | None, optional
        :param get_slabs_kwargs: Additional kwargs passed to
            :meth:`SlabGenerator.get_slabs`. Default is None.
        :type get_slabs_kwargs: dict | None, optional
        :return: A dictionary of the generated slab structures keyed by names like
            ``"slab_00"``, ``"slab_01"``.
        :rtype: dict[str, Any]
        """
        if not isinstance(bulk_struct, Structure):
            raise TypeError("`bulk_struct` must be a pymatgen Structure")

        bulk = bulk_struct.to_conventional()

        relax_cell = self.relax_bulk
        relax_atoms = self.relax_bulk

        relaxer_bulk = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=relax_cell,
            relax_atoms=relax_atoms,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        bulk_opt = relaxer_bulk.calc(bulk)

        slab_dict = {}
        slab_dict["final_bulk"] = bulk_opt["final_structure"]
        slab_dict["bulk_energy"] = bulk_opt["energy"]

        slabgen = SlabGenerator(
            initial_structure=bulk,
            miller_index=miller_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            **(slab_gen_kwargs or {}),
        )
        slabs = slabgen.get_slabs(symmetrize=symmetrize, **(get_slabs_kwargs or {}))

        for id_, slab in enumerate(slabs):
            key = f"slab_{id_:02d}"
            slab_ = slab.make_supercell((*inplane_supercell, 1))
            slab_dict[key] = slab_
        return self.calc(slab_dict)

    def calc(
        self,
        structure: Structure | dict[str, Any],
    ) -> dict[str, Any]:
        """
        For each slab structure in the given dictionary, compute its energy and
        surface energy based on the stored bulk reference.

        The dictionary keys are enumerated, and the slab structures are passed
        to a :class:`RelaxCalc` if ``relax_slab=True``, else a single-point
        calculation is performed. The output is keyed by integer index.

        :param structure: A dictionary mapping keys (e.g. "slab_00") to slab structures.
        :type structure: dict[str, Any]
        :return: A dictionary keyed by integer indices. Each value is another dict
            containing ``bulk_energy``, ``final_bulk_structure``, ``slab_energy``,
            ``surface_energy``, and ``final_slab_structure``.
        :rtype: dict[int, dict[str, Any]]
        :raises ValueError: If ``structure`` is not a dictionary.
        """
        if not isinstance(structure, dict):
            raise ValueError(  # noqa:TRY004
                "For surface calculations, \
                    structure must be a dict containing the images with keys slab_00, slab_01, etc."
            )

        if "bulk_energy" not in structure or "final_bulk" not in structure:
            raise ValueError("Bulk energy or final bulk structure is not initialized.")

        result_dict = {}

        for key, slab in structure.items():
            if "bulk" in key:
                continue

            relax_atoms = self.relax_slab

            relaxer_slab = RelaxCalc(
                calculator=self.calculator,
                fmax=self.fmax,
                max_steps=self.max_steps,
                relax_cell=False,
                relax_atoms=relax_atoms,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
            )
            slab_opt = relaxer_slab.calc(slab)
            final_slab = slab_opt["final_structure"]
            slab_energy = slab_opt["energy"]

            # Compute surface energy
            # Assuming two surfaces: (E_slab - N_slab_atoms * E_bulk_per_atom) / (2 * area)
            area = slab.surface_area
            n_slab_atoms = len(final_slab)
            bulk_e_per_atom = structure["bulk_energy"] / len(structure["final_bulk"])
            gamma = (slab_energy - n_slab_atoms * bulk_e_per_atom) / (2 * area)

            result_dict[key] = {
                "bulk_energy": structure["bulk_energy"],
                "final_bulk_structure": structure["final_bulk"],
                "initial_slab_structure": slab,
                "final_slab_structure": final_slab,
                "slab_energy": slab_energy,
                "surface_energy": gamma,
            }

        return result_dict
