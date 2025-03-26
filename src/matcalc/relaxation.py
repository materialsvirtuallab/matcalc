"""Relaxation properties."""

from __future__ import annotations

import contextlib
import io
import pickle
from typing import TYPE_CHECKING

from ase.filters import FrechetCellFilter
from pymatgen.io.ase import AseAtomsAdaptor

from matcalc.utils import get_ase_optimizer

from .base import PropCalc

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Init the Trajectory Observer from a Atoms.

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
        """Args:
            calculator: ASE Calculator to use.
            optimizer (str | ase Optimizer): The optimization algorithm. Defaults to "FIRE".
            max_steps (int): Max number of steps for relaxation. Defaults to 500.
            traj_file (str | None): File to save the trajectory to. Defaults to None.
            interval (int): The step interval for saving the trajectories. Defaults to 1.
            fmax (float): Total force tolerance for relaxation convergence.
                fmax is a sum of force and stress forces. Defaults to 0.1 (eV/A).
            relax_atoms (bool): Whether to relax the atoms (or just static calculation).
            relax_cell (bool): Whether to relax the cell (or just atoms).
            cell_filter (Filter): The ASE Filter used to relax the cell. Default is FrechetCellFilter.
            perturb_distance (float | None): Distance in angstrom to randomly perturb each site to break symmetry.
                Defaults to None.

        Raises:
            ValueError: If the optimizer is not a valid ASE optimizer.
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
        """Perform relaxation to obtain properties.

        Args:
            structure: Pymatgen structure.

        Returns: {
            final_structure: final_structure,
            energy: static energy or trajectory observer final energy in eV,
            forces: forces in eV/A,
            stress: stress in eV/A^3,
            volume: lattice.volume in A^3,
            a: lattice.a in A,
            b: lattice.b in A,
            c: lattice.c in A,
            alpha: lattice.alpha in degrees,
            beta: lattice.beta in degrees,
            gamma: lattice.gamma in degrees,
        }
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
