"""Relaxation properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.filters import FrechetCellFilter

from ._base import PropCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class RelaxCalc(PropCalc):
    """
    A class to perform structural relaxation calculations using ASE (Atomic Simulation Environment).

    This class facilitates structural relaxation by integrating with ASE tools for
    optimized geometry and/or cell parameters. It enables convergence on forces, stress,
    and total energy, offering customization for relaxation parameters and further
    enabling properties to be extracted from the relaxed structure.

    :ivar calculator: ASE Calculator used for force and energy evaluations.
    :type calculator: Calculator

    :ivar optimizer: Algorithm for performing the optimization.
    :type optimizer: Optimizer | str

    :ivar max_steps: Maximum number of optimization steps allowed.
    :type max_steps: int

    :ivar traj_file: Path to save relaxation trajectory (optional).
    :type traj_file: str | None

    :ivar interval: Interval for saving trajectory frames during relaxation.
    :type interval: int

    :ivar fmax: Force tolerance for convergence (eV/Å).
    :type fmax: float

    :ivar relax_atoms: Whether atomic positions are relaxed.
    :type relax_atoms: bool

    :ivar relax_cell: Whether the cell parameters are relaxed.
    :type relax_cell: bool

    :ivar cell_filter: ASE filter used for modifying the cell during relaxation.
    :type cell_filter: Filter

    :ivar perturb_distance: Distance (Å) for random perturbation to break symmetry.
    :type perturb_distance: float | None
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        optimizer: Optimizer | str = "FIRE",
        max_steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        fmax: float = 0.1,
        relax_atoms: bool = True,
        relax_cell: bool = True,
        cell_filter: Filter = FrechetCellFilter,  # type: ignore[assignment]
        perturb_distance: float | None = None,
    ) -> None:
        """
        Initializes the relaxation procedure for an atomic configuration system.

        This constructor sets up the relaxation pipeline, configuring the required
        calculator, optimizer, relaxation parameters, and logging options. The
        relaxation process aims to find the minimum energy configuration, optionally
        relaxing atoms and/or the simulation cell within the specified constraints.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param optimizer: The optimization algorithm to use for relaxation. It can
            either be an instance of an Optimizer class or a string identifier for
            a recognized ASE optimizer. Defaults to "FIRE".
        :param max_steps: The maximum number of optimization steps to perform
            during the relaxation process. Defaults to 500.
        :param traj_file: Path to a file for periodic trajectory output (if specified).
            This file logs the atomic positions and cell configurations after a given
            interval. Defaults to None.
        :param interval: The interval (in steps) at which the trajectory file is
            updated. Defaults to 1.
        :param fmax: The force convergence threshold. Relaxation continues until the
            maximum force on any atom falls below this value. Defaults to 0.1.
        :param relax_atoms: A flag indicating whether the atomic positions are to
            be relaxed. Defaults to True.
        :param relax_cell: A flag indicating whether the simulation cell is to
            be relaxed. Defaults to True.
        :param cell_filter: The filter to apply when relaxing the simulation cell.
            This determines constraints or allowed degrees of freedom during
            cell relaxation. Defaults to FrechetCellFilter.
        :param perturb_distance: A perturbation distance used for initializing
            the system configuration before relaxation. If None, no perturbation
            is applied. Defaults to None.
        """
        self.calculator = calculator  # type: ignore[assignment]

        self.optimizer = optimizer
        self.fmax = fmax
        self.interval = interval
        self.max_steps = max_steps
        self.traj_file = traj_file
        self.relax_cell = relax_cell
        self.relax_atoms = relax_atoms
        self.cell_filter = cell_filter
        self.perturb_distance = perturb_distance

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """
        Calculate the final relaxed structure, energy, forces, and stress for a given
        structure and update the result dictionary with additional geometric properties.

        This method takes an input structure and performs a relaxation process
        depending on the specified parameters. If the `perturb_distance` attribute
        is provided, the structure is perturbed before relaxation. The relaxation
        process can involve optimization of both atomic positions and the unit cell
        if specified. Results of the calculation including final structure geometry,
        energy, forces, stresses, and lattice parameters are returned in a dictionary.

        :param structure: Input structure for calculation. Can be provided as a
                          `Structure` object or a dictionary convertible to `Structure`.
        :return: Dictionary containing the final relaxed structure, calculated
                 energy, forces, stress, and lattice parameters.
        :rtype: dict
        """
        result = super().calc(structure)

        structure_in: Structure | Atoms = result["final_structure"]

        if self.perturb_distance is not None:
            structure_in = to_pmg_structure(structure_in).perturb(distance=self.perturb_distance, seed=None)

        r = run_pes_calc(
            structure_in,
            self.calculator,
            relax_atoms=self.relax_atoms,
            relax_cell=self.relax_cell,
            optimizer=self.optimizer,
            max_steps=self.max_steps,
            traj_file=self.traj_file,
            interval=self.interval,
            fmax=self.fmax,
            cell_filter=self.cell_filter,
        )

        lattice = r.structure.lattice
        result.update(
            {
                "final_structure": r.structure,
                "energy": r.potential_energy,
                "forces": r.forces,
                "stress": r.stress,
                "a": lattice.a,
                "b": lattice.b,
                "c": lattice.c,
                "alpha": lattice.alpha,
                "beta": lattice.beta,
                "gamma": lattice.gamma,
                "volume": lattice.volume,
            }
        )

        return result
