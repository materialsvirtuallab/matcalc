"""Surface Energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatgen.core.surface import SlabGenerator

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import get_ase_optimizer

if TYPE_CHECKING:
    from collections.abc import Generator

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
      4. Optionally, use :meth:`calc_many` to process multiple inputs in parallel.

    Example:
        >> my_calc = SomeASECalculator(...)
        >> surf_calc = SurfaceCalc(calculator=my_calc, miller_index=(1,0,0), ...)
        >> slabs_dict = surf_calc.calc_slabs(bulk_structure)
        >> results = surf_calc.calc(slabs_dict)
        >> print(results)

    :ivar calculator: The ASE Calculator object used for energy/force evaluations.
    :type calculator: Calculator
    :ivar miller_index: Miller index for the surface, e.g. (1, 0, 0).
    :type miller_index: tuple[int,int,int]
    :ivar min_slab_size: Thickness of the slab in Å.
    :type min_slab_size: float
    :ivar min_vacuum_size: Thickness of the vacuum region in Å.
    :type min_vacuum_size: float
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
        miller_index: tuple[int, int, int] = (1, 0, 0),
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 20.0,
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
        :param miller_index: Miller index for the slab surfaces (e.g., (1, 0, 0)).
            Default is (1, 0, 0).
        :type miller_index: tuple[int,int,int], optional
        :param min_slab_size: Thickness of the slab in Å. Default is 10.0 Å.
        :type min_slab_size: float, optional
        :param min_vacuum_size: Thickness of the vacuum region in Å. Default is 20.0 Å.
        :type min_vacuum_size: float, optional
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
        self.miller_index = miller_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.relax_bulk = relax_bulk
        self.relax_slab = relax_slab
        self.fmax = fmax
        self.optimizer = get_ase_optimizer(optimizer)
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

        self.final_bulk = None
        self.bulk_energy = None
        self.n_bulk_atoms = None

    def calc_slabs(
        self,
        bulk_struct: Structure,
        symmetrize: bool = True,  # noqa: FBT001,FBT002
        inplane_supercell: tuple[int, int] = (1, 1),
        slab_gen_kwargs: dict | None = None,
        get_slabs_kwargs: dict | None = None,
    ) -> dict[str, Structure]:
        """
        Relax (or single-point calculate) a bulk structure, then generate slab structures
        using the specified Miller index. Stores the resulting bulk data internally:
        ``self.final_bulk``, ``self.bulk_energy``, and ``self.n_bulk_atoms``.

        :param bulk_struct: A Pymatgen bulk structure. It is converted to its conventional
            cell before relaxation or single-point calculation.
        :type bulk_struct: Structure
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
        :rtype: dict[str, Structure]
        """
        bulk = bulk_struct.to_conventional()

        relax_cell = self.relax_bulk
        relax_atoms = self.relax_bulk

        relaxer_bulk = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=relax_cell,
            relax_atoms=relax_atoms,
            **(self.relax_calc_kwargs or {}),
        )
        bulk_opt = relaxer_bulk.calc(bulk)

        self.final_bulk = bulk_opt["final_structure"]
        self.bulk_energy = bulk_opt["energy"]
        self.n_bulk_atoms = len(self.final_bulk)

        # Generate slabs
        slabgen = SlabGenerator(
            initial_structure=bulk,
            miller_index=self.miller_index,
            min_slab_size=self.min_slab_size,
            min_vacuum_size=self.min_vacuum_size,
            **(slab_gen_kwargs or {}),
        )
        slabs = slabgen.get_slabs(symmetrize=symmetrize, **(get_slabs_kwargs or {}))

        slab_dict = {}
        for id_, slab in enumerate(slabs):
            key = f"slab_{id_:02d}"
            slab_ = slab.make_supercell((*inplane_supercell, 1))
            slab_dict[key] = slab_
        return slab_dict

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

        result_dict = {}

        for key, slab in enumerate(structure.items()):
            relax_atoms = self.relax_slab

            relaxer_slab = RelaxCalc(
                calculator=self.calculator,
                fmax=self.fmax,
                max_steps=self.max_steps,
                relax_cell=False,
                relax_atoms=relax_atoms,
                **(self.relax_calc_kwargs or {}),
            )
            slab_opt = relaxer_slab.calc(slab)
            final_slab = slab_opt["final_structure"]
            slab_energy = slab_opt["energy"]

            # Compute surface energy
            # Assuming two surfaces: (E_slab - N_slab_atoms * E_bulk_per_atom) / (2 * area)
            area = final_slab.surface_area
            n_slab_atoms = len(final_slab)
            bulk_e_per_atom = self.bulk_energy / self.n_bulk_atoms
            gamma = (slab_energy - n_slab_atoms * bulk_e_per_atom) / (2 * area)

            result_dict[key] = {
                "bulk_energy": self.bulk_energy,
                "final_bulk_structure": self.final_bulk,
                "slab_energy": slab_energy,
                "surface_energy": gamma,
                "final_slab_structure": final_slab,
            }

        return result_dict

    def calc_many(
        self,
        structures: dict[str, Any],
        n_jobs: None | int = None,
        allow_errors: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> Generator[dict | None]:
        """
        Parallelize :meth:`calc` over multiple slab entries.

        The input dictionary is split into single-slab dictionaries, each of which
        is processed by :meth:`calc` independently in parallel (depending on ``n_jobs``).

        :param structures: A dictionary of slabs, e.g. {"slab_00": Slab(...), "slab_01": Slab(...)}.
        :type structures: dict[str, Any]
        :param n_jobs: Number of parallel jobs. If None, runs serially or
            uses a default parallelism set by joblib. Default is None.
        :type n_jobs: int | None, optional
        :param allow_errors: If True, any exception in :meth:`calc` is caught and yields
            None for that slab; otherwise, the exception propagates. Default is False.
        :type allow_errors: bool, optional
        :param kwargs: Additional arguments passed to joblib.Parallel.
        :return: A generator that yields the result of :meth:`calc` for each slab
            dictionary, or None if an error occurred and ``allow_errors=True``.
        :rtype: Generator[dict | None, None, None]
        :raises ValueError: If ``structures`` is not a dict.
        """
        if not isinstance(structures, dict):
            raise ValueError(  # noqa:TRY004
                "For surface calculations, \
                    structure must be a dict containing the images with keys slab_00, slab_01, etc."
            )

        structures = [{key: value} for key, value in structures.items()]

        return super().calc_many(structures, n_jobs, allow_errors, **kwargs)
