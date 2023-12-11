"""NEB calculations."""
from __future__ import annotations

import os
from inspect import isclass
from typing import TYPE_CHECKING

from ase import optimize
from ase.io import Trajectory
from ase.neb import NEB, NEBTools
from ase.optimize.optimize import Optimizer
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matcalc.base import PropCalc
from matcalc.util import get_universal_calculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class NEBCalc(PropCalc):
    """Nudged Elastic Band calculator."""

    def __init__(
            self,
            images: list[Structure],
            calculator: str | Calculator = "M3GNet-MP-2021.2.8-DIRECT-PES",
            optimizer: str | Optimizer = "BFGS",
            traj_folder: str | None = None,
            interval: int = 1,
            climb: bool = True,
            **kwargs,
    ):
        """
        Args:
            images(list): A list of pymatgen structures as NEB image structures.
            calculator(str|Calculator): ASE Calculator to use. Default to M3GNet-MP-2021.2.8-DIRECT-PES.
            optimizer(str|Optimizer): The optimization algorithm. Defaults to "BEGS".
            traj_folder(str|None): The folder address to store NEB trajectories. Default to None.
            interval(int): The step interval for saving the trajectories. Defaults to 1.
            climb(bool): Whether to enable climb image NEB. Default to True.
            kwargs: Other arguments passed to ASE NEB object.
        """
        self.images = images
        self.calculator = calculator

        # check str is valid optimizer key
        def is_ase_optimizer(key):
            return isclass(obj := getattr(optimize, key)) and issubclass(obj, Optimizer)

        valid_keys = [key for key in dir(optimize) if is_ase_optimizer(key)]
        if isinstance(optimizer, str) and optimizer not in valid_keys:
            raise ValueError(f"Unknown {optimizer=}, must be one of {valid_keys}")

        self.optimizer: Optimizer = (
            getattr(optimize, optimizer) if isinstance(optimizer, str) else optimizer
        )
        self.traj_folder = traj_folder
        self.interval = interval
        self.climb = climb

        self.images = []
        for image in images:
            if isinstance(image, Structure):
                atoms = AseAtomsAdaptor().get_atoms(image)
            atoms.calc = get_universal_calculator(self.calculator)
            self.images.append(atoms)

        self.neb = NEB(
            self.images,
            climb=self.climb,
            allow_shared_calculator=True,
            **kwargs,
        )
        self.optimizer = self.optimizer(self.neb)

    @classmethod
    def from_end_images(
            cls,
            start_struct: Structure,
            end_struct: Structure,
            calculator: str | Calculator = "M3GNet-MP-2021.2.8-DIRECT-PES",
            nimages: int = 7,
            interpolate_lattices: bool = False,
            autosort_tol: float = 0.5,
            **kwargs,
    ):
        """
        Initialize a NEBCalc from end images.

        Args:
            start_struct(Structure): The starting image as a pymatgen Structure.
            end_struct(Structure): The ending image as a pymatgen Structure.
            calculator(str|Calculator): ASE Calculator to use. Default to M3GNet-MP-2021.2.8-DIRECT-PES.
            nimages(int): The number of intermediate image structures to create.
            interpolate_lattices(bool): Whether to interpolate the lattices when creating NEB
                path with Structure.interpolate() in pymatgen. Default to False.
            autosort_tol(float): A distance tolerance in angstrom in which to automatically
                sort end_struct to match to the closest points in start_struct. This
                argument is required for Structure.interpolate() in pymatgen.
                Default to 0.5.
            kwargs: Other arguments passed to construct NEBCalc.
        """
        images = start_struct.interpolate(
            end_struct,
            nimages=nimages + 1,
            interpolate_lattices=interpolate_lattices,
            pbc=False,
            autosort_tol=autosort_tol,
        )
        return cls(images=images, calculator=calculator, **kwargs)

    def calc(
            self,
            fmax: float = 0.1,
            max_steps: int = 1000,
    ):
        """
        Perform NEB calculation.

        Args:
            fmax (float): Convergence criteria for NEB calculations defined by Max forces.
                Default to 0.1 eV/Angstrom.
            max_steps (int): Maximum number of steps in NEB calculations. Default to 1000.

        Returns:
            NEB barrier.
        """
        if self.traj_folder is not None:
            os.makedirs(self.traj_folder, exist_ok=True)
            for i in range(len(self.images)):
                self.optimizer.attach(
                    Trajectory(
                        f"{self.traj_folder}/image_{i}.traj", "w", self.images[i]
                    ),
                    interval=self.interval,
                )
        self.optimizer.run(fmax=fmax, steps=max_steps)
        neb_tool = NEBTools(self.neb.images)
        return neb_tool.get_barrier()
