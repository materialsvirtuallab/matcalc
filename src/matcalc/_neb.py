"""NEB calculations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ase.io import Trajectory
from ase.mep import NEBTools
from ase.neb import NEB
from ase.utils.forcecurve import fit_images

from ._base import PropCalc
from .backend._ase import get_ase_optimizer
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class NEBCalc(PropCalc):
    """NEB calculator."""

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        optimizer: str | Optimizer = "BFGS",
        traj_folder: str | None = None,
        interval: int = 1,
        climb: bool = True,
        fmax: float = 0.1,
        max_steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize the instance of the class.

        Parameters:
            calculator (Calculator | str): An ASE calculator object used to perform energy and force
                calculations. If string is provided, the corresponding universal calculator is loaded.
            optimizer (str | Optimizer, optional): The optimization algorithm to use. Defaults to "BFGS".
            traj_folder (str | None, optional): The folder to save trajectory information. Defaults to None.
            interval (int, optional): The interval for recording trajectory information. Defaults to 1.
            climb (bool, optional): Whether to perform climbing image nudged elastic band (CI-NEB). Defaults to True.
            fmax (float, optional): The maximum force tolerance for convergence. Defaults to 0.1.
            max_steps (int, optional): The maximum number of optimization steps. Defaults to 1000.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None
        """
        self.calculator = calculator  # type: ignore[assignment]

        self.traj_folder = traj_folder
        self.interval = interval
        self.climb = climb
        self.optimizer = get_ase_optimizer(optimizer)
        self.fmax = fmax
        self.max_steps = max_steps
        self.kwargs = kwargs

    def calc_images(
        self,
        start_struct: Structure,
        end_struct: Structure,
        *,
        n_images: int = 7,
        interpolate_lattices: bool = False,
        autosort_tol: float = 0.5,
    ) -> dict[str, Any]:
        """Calculate NEB images between given start and end structures.

        Parameters:
            start_struct (Structure): Initial structure.
            end_struct (Structure): Final structure.
            n_images (int): Number of images to calculate (default is 7).
            interpolate_lattices (bool): Whether to interpolate lattices between start and end structures (default is
                False).
            autosort_tol (float): Tolerance for autosorting the images (default is 0.5).

        Returns:
            NEBCalc: NEB calculation object containing the interpolated images.
        """
        images = start_struct.interpolate(
            end_struct,
            nimages=n_images + 1,
            interpolate_lattices=interpolate_lattices,
            pbc=False,
            autosort_tol=autosort_tol,
        )
        return self.calc({f"image{i:02d}": s for i, s in enumerate(images)})

    def calc(
        self,
        structure: Structure | Atoms | dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate the energy barrier using the nudged elastic band method.

        Parameters:
            - structure: A dictionary containing the images with keys 'image0', 'image1', etc. Must be of type dict.

        Returns:
                dict:
                    - "barrier" (float): The energy barrier of the reaction pathway.
                    - "force" (float): The force exerted on the atoms during the NEB calculation.
                    - "mep" (dict): a dictionary containing the images and their respective energies.
        """
        if not isinstance(structure, dict):
            raise ValueError(  # noqa:TRY004
                "For NEB calculations, structure must be a dict containing the images with keys image00, image01, etc."
            )
        images: list[Atoms] = []
        for _, image in sorted(structure.items(), key=lambda x: x[0]):
            atoms = to_ase_atoms(image)
            atoms.calc = self.calculator
            images.append(atoms)

        self.neb = NEB(images, climb=self.climb, allow_shared_calculator=True, **self.kwargs)
        optimizer = self.optimizer(self.neb)  # type:ignore[operator]
        if self.traj_folder is not None:
            os.makedirs(self.traj_folder, exist_ok=True)
            for idx, img in enumerate(images):
                optimizer.attach(
                    Trajectory(f"{self.traj_folder}/image-{idx}.traj", "w", img),
                    interval=self.interval,
                )

        optimizer.run(fmax=self.fmax, steps=self.max_steps)
        neb_tool = NEBTools(self.neb.images)
        data = neb_tool.get_barrier()  # add structures
        result = {"barrier": data[0], "force": data[1]}

        energies = fit_images(self.neb.images).energies
        mep = {
            f"image{i:02d}": {"structure": to_pmg_structure(image), "energy": energy}
            for i, (image, energy) in enumerate(zip(self.neb.images, energies, strict=False))
        }
        result["mep"] = mep
        return result
