"""
This module implements convenience functions to perform various atomic simulation types, specifically relaxation and
static, using either ASE or LAMMPS. This enables the code to use either for the computation of various properties.
"""

from __future__ import annotations

import contextlib
import io
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from ase.filters import FrechetCellFilter

from .utils import get_ase_optimizer, to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.optimize.optimize import Optimizer
    from pymatgen.core.structure import Structure


class PESResult(NamedTuple):
    """Container for results from PES calculators."""

    structure: Structure
    energy: float
    forces: np.ndarray
    stress: np.ndarray


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


def run_ase(
    structure: Structure | Atoms,
    calculator: Calculator,
    *,
    relax_atoms: bool = False,
    relax_cell: bool = False,
    optimizer: Optimizer | str = "FIRE",
    max_steps: int = 500,
    traj_file: str | None = None,
    interval: int = 1,
    fmax: float = 0.1,
    cell_filter: Filter = FrechetCellFilter,
) -> PESResult:
    """
    Run ASE static calculation using the given structure and calculator.

    Parameters:
    structure (Structure|Atoms): The input structure to calculate potential energy, forces, and stress.
    calculator (Calculator): The calculator object to use for the calculation.

    Returns:
    PESResult: Object containing potential energy, forces, and stress of the input structure.
    """
    atoms = to_ase_atoms(structure)
    atoms.calc = calculator
    if relax_atoms:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if relax_cell:
                atoms = cell_filter(atoms)  # type:ignore[operator]
            opt = get_ase_optimizer(optimizer)(atoms)  # type:ignore[operator]
            opt.attach(obs, interval=interval)
            opt.run(fmax=fmax, steps=max_steps)
            if traj_file is not None:
                obs()
                obs.save(traj_file)
        if relax_cell:
            atoms = atoms.atoms  # type:ignore[attr-defined]
        return PESResult(to_pmg_structure(atoms), obs.energies[-1], obs.forces[-1], obs.stresses[-1])

    return PESResult(to_pmg_structure(structure), atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
