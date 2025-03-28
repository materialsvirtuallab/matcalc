"""NEB calculations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ase.io import Trajectory
from ase.neb import NEB, NEBTools
from pymatgen.core import Structure

from ._base import PropCalc
from .utils import get_ase_optimizer

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer


class NEBCalc(PropCalc):
    """Nudged Elastic Band calculator."""

    def __init__(
        self,
        calculator: Calculator,
        images: list[Structure],
        *,
        optimizer: str | Optimizer = "BFGS",
        traj_folder: str | None = None,
        interval: int = 1,
        climb: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            images(list): A list of pymatgen structures as NEB image structures.
            calculator(str | Calculator): ASE Calculator to use. Defaults to M3GNet-MP-2021.2.8-DIRECT-PES.
            optimizer(str | Optimizer): The optimization algorithm. Defaults to "BEGS".
            traj_folder(str | None): The folder address to store NEB trajectories. Defaults to None.
            interval(int): The step interval for saving the trajectories. Defaults to 1.
            climb(bool): Whether to enable climb image NEB. Defaults to True.
            kwargs: Other arguments passed to ASE NEB object.
        """
        self.calculator = calculator

        self.optimizer = get_ase_optimizer(optimizer)
        self.traj_folder = traj_folder
        self.interval = interval
        self.climb = climb

        self.images: list[Atoms] = []
        for image in images:
            atoms = image.to_ase_atoms() if isinstance(image, Structure) else image
            atoms.calc = self.calculator
            self.images.append(atoms)

        self.neb = NEB(self.images, climb=self.climb, allow_shared_calculator=True, **kwargs)
        self.optimizer = self.optimizer(self.neb)  # type:ignore[operator]

    @classmethod
    def from_end_images(
        cls: type[NEBCalc],
        calculator: Calculator,
        start_struct: Structure,
        end_struct: Structure,
        *,
        n_images: int = 7,
        interpolate_lattices: bool = False,
        autosort_tol: float = 0.5,
        **kwargs: Any,
    ) -> NEBCalc:
        """Initialize a NEBCalc from end images.

        Args:
            start_struct(Structure): The starting image as a pymatgen Structure.
            end_struct(Structure): The ending image as a pymatgen Structure.
            calculator(str | Calculator): ASE Calculator to use. Defaults to M3GNet-MP-2021.2.8-DIRECT-PES.
            n_images(int): The number of intermediate image structures to create.
            interpolate_lattices(bool): Whether to interpolate the lattices when creating NEB
                path with Structure.interpolate() in pymatgen. Defaults to False.
            autosort_tol(float): A distance tolerance in angstrom in which to automatically
                sort end_struct to match to the closest points in start_struct. This
                argument is required for Structure.interpolate() in pymatgen.
                Defaults to 0.5.
            kwargs: Other arguments passed to construct NEBCalc.
        """
        images = start_struct.interpolate(
            end_struct,
            nimages=n_images + 1,
            interpolate_lattices=interpolate_lattices,
            pbc=False,
            autosort_tol=autosort_tol,
        )
        return cls(images=images, calculator=calculator, **kwargs)

    def calc(  # type: ignore[override]
        self, fmax: float = 0.1, max_steps: int = 1000
    ) -> tuple[float, float]:
        """Perform NEB calculation.

        Args:
            fmax (float): Convergence criteria for NEB calculations defined by Max forces.
                Defaults to 0.1 eV/A.
            max_steps (int): Maximum number of steps in NEB calculations. Defaults to 1000.

        Returns:
            float: The energy barrier in eV.
        """
        if self.traj_folder is not None:
            os.makedirs(self.traj_folder, exist_ok=True)
            for idx, img in enumerate(self.images):
                self.optimizer.attach(
                    Trajectory(f"{self.traj_folder}/image-{idx}.traj", "w", img),
                    interval=self.interval,
                )
        self.optimizer.run(fmax=fmax, steps=max_steps)
        neb_tool = NEBTools(self.neb.images)
        return neb_tool.get_barrier()
