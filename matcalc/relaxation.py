"""Relaxation properties."""
from __future__ import annotations

import contextlib
import io
import pickle
from typing import TYPE_CHECKING

from ase.constraints import ExpCellFilter
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    import numpy as np
    from ase.optimize.optimize import Optimizer

from .base import PropCalc

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}
if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Init the Trajectory Observer from a Atoms.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

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
    """Relaxes and computes the relaxed parameters of a structure."""

    def __init__(
        self,
        calculator: Calculator,
        optimizer: Optimizer | str = "FIRE",
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        interval=1,
        relax_cell=True,
    ):
        """
        Args:
            calculator: ASE Calculator to use.
            optimizer (str or ase Optimizer): the optimization algorithm.
                Defaults to "FIRE"
            fmax (float): Total force tolerance for relaxation convergence. fmax is a sum of force and stress forces.
            steps (int): Max number of steps for relaxation.
            traj_file (str): The trajectory file for saving
            interval (int): The step interval for saving the trajectories.
            relax_cell (bool): Whether to relax the cell.
        """
        self.calculator = calculator
        self.optimizer: Optimizer = OPTIMIZERS[optimizer] if isinstance(optimizer, str) else optimizer
        self.fmax = fmax
        self.interval = interval
        self.steps = steps
        self.traj_file = traj_file
        self.relax_cell = relax_cell

    def calc(self, structure) -> dict:
        """
        Perform relaxation to obtain properties.

        Args:
            structure: Pymatgen structure.

        Returns: {
            "final_structure": final_structure,
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
            "volume": lattice.volume,
        }
        """
        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(structure)
        atoms.set_calculator(self.calculator)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.optimizer(atoms)
            optimizer.attach(obs, interval=self.interval)
            optimizer.run(fmax=self.fmax, steps=self.steps)
            if self.traj_file is not None:
                obs()
                obs.save(self.traj_file)
        if self.relax_cell:
            atoms = atoms.atoms

        final_structure = ase_adaptor.get_structure(atoms)
        lattice = final_structure.lattice

        return {
            "final_structure": final_structure,
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
            "volume": lattice.volume,
            "energy": obs.energies[-1],
        }
