"""Relaxation properties."""

from __future__ import annotations

import contextlib
import io
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.filters import FrechetCellFilter
from pymatgen.io.ase import AseAtomsAdaptor

from matcalc.utils import get_ase_optimizer

from ._base import PropCalc

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


@dataclass
class TrajectoryObserver:
    """
    Class for observing and recording the properties of an atomic structure during relaxation.

    The `TrajectoryObserver` class is designed to track and store the atomic properties like
    energies, forces, stresses, atom positions, and cell structure of an atomic system
    represented by an `Atoms` object. It provides functionality to save recorded data
    to a file for further analysis or usage.

    :ivar atoms: The atomic structure being observed.
    :type atoms: Atoms
    :ivar energies: List of potential energy values of the atoms during relaxation.
    :type energies: list[float]
    :ivar forces: List of force arrays recorded for the atoms during relaxation.
    :type forces: list[np.ndarray]
    :ivar stresses: List of stress tensors recorded for the atoms during relaxation.
    :type stresses: list[np.ndarray]
    :ivar atom_positions: List of atomic positions recorded during relaxation.
    :type atom_positions: list[np.ndarray]
    :ivar cells: List of unit cell arrays recorded during relaxation.
    :type cells: list[np.ndarray]
    """

    atoms: Atoms
    energies: list[float] = field(default_factory=list)
    forces: list[np.ndarray] = field(default_factory=list)
    stresses: list[np.ndarray] = field(default_factory=list)
    atom_positions: list[np.ndarray] = field(default_factory=list)
    cells: list[np.ndarray] = field(default_factory=list)

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)


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
        calculator: Calculator,
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

        :param calculator: A calculator object used to perform energy and force
            calculations during the relaxation process.
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
        self.calculator = calculator

        self.optimizer = get_ase_optimizer(optimizer)
        self.fmax = fmax
        self.interval = interval
        self.max_steps = max_steps
        self.traj_file = traj_file
        self.relax_cell = relax_cell
        self.relax_atoms = relax_atoms
        self.cell_filter = cell_filter
        self.perturb_distance = perturb_distance

    def calc(self, structure: Structure | dict) -> dict:
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
        structure_in: Structure = result["final_structure"]

        if self.perturb_distance is not None:
            structure_in = structure_in.perturb(distance=self.perturb_distance)
        atoms = AseAtomsAdaptor.get_atoms(structure_in)
        atoms.calc = self.calculator
        if self.relax_atoms:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                obs = TrajectoryObserver(atoms)
                if self.relax_cell:
                    atoms = self.cell_filter(atoms)  # type:ignore[operator]
                optimizer = self.optimizer(atoms)  # type:ignore[operator]
                optimizer.attach(obs, interval=self.interval)
                optimizer.run(fmax=self.fmax, steps=self.max_steps)
                if self.traj_file is not None:
                    obs()
                    obs.save(self.traj_file)
            if self.relax_cell:
                atoms = atoms.atoms  # type:ignore[attr-defined]
            energy = obs.energies[-1]
            forces = obs.forces[-1]
            stress = obs.stresses[-1]
            final_structure = AseAtomsAdaptor.get_structure(atoms)

        else:
            final_structure = structure_in
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()

        lattice = final_structure.lattice
        result.update(
            {
                "final_structure": final_structure,
                "energy": energy,
                "forces": forces,
                "stress": stress,
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
