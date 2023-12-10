from __future__ import annotations

import contextlib
import io
import numpy as np
import os

from ase import Atoms
from ase import optimize
from ase.calculators.calculator import Calculator
from ase.io import Trajectory
from ase.neb import NEB, NEBTools
from ase.optimize.optimize import Optimizer
from ase.calculators.emt import EMT

from inspect import isclass
from matcalc.base import PropCalc
from matcalc.util import get_universal_calculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class NEBcal(PropCalc):
    """
    Nudged Elastic Band calculator.
    """

    def __init__(
        self,
        images: list,
        calculator: str = "M3GNet-MP-2021.2.8-DIRECT-PES",
        optimizer: Optimizer | str = "BFGS",
        traj_folder: str | None = None,
        interval: int = 1,
        climb=True,
        **kwargs,
    ):
        """
        Args:
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
        for atoms in images:
            if isinstance(atoms, Structure):
                atoms = AseAtomsAdaptor().get_atoms(atoms)
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
        calculator: Calculator,
        nimages: int | Iterable = 7,
        interpolate_lattices: bool = False,
        autosort_tol: float = 0.5,
        **kwargs,
    ):
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
        fmax=0.1,
        max_steps: int = 1000,
    ):
        """
        Thin wrapper of ase MD run
        Args:
            fmax (float): Convergence criteria for NEB calculations.
            max_steps (int): Maximum number of steps in NEB calculations. Default to 1000.
        Returns:
            NEB barrier.
        """
        if self.traj_folder is not None:
            os.makedirs(self.traj_folder, exist_ok=True)
            for i in range(0, len(self.images)):
                self.optimizer.attach(
                    Trajectory(
                        f"{self.traj_folder}/image_{i}.traj", "w", self.images[i]
                    )
                )
        self.optimizer.run(fmax=fmax, steps=max_steps)
        neb_tool = NEBTools(self.optimizer.neb.images)
        return neb_tool.get_barrier()
