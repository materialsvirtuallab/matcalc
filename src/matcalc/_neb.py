"""NEB calculations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ase.io import Trajectory
from ase.neb import NEB, NEBTools
from ase.utils.forcecurve import fit_images
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ._base import PropCalc
from .utils import get_ase_optimizer

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer


class NEBCalc(PropCalc):
    """NEB calculator."""

    def __init__(
        self,
        calculator: Calculator,
        *,
        optimizer: str | Optimizer = "BFGS",
        traj_folder: str | None = None,
        interval: int = 1,
        climb: bool = True,
        fmax: float = 0.1,
        max_steps: int = 1000,
        get_mep: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the instance of the class.

        Parameters:
            calculator: Calculator - The calculator object to use.
            optimizer: str | Optimizer - The optimizer algorithm to use, default is "BFGS".
            traj_folder: str | None - The folder path to save the trajectory files, default is None.
            interval: int - The interval for saving the trajectory, default is 1.
            climb: bool - Specifies whether to perform climbing image nudged elastic band (CI-NEB) calculations,
                default is True.
            fmax: float - The maximum force allowed on atoms, default is 0.1.
            max_steps: int - The maximum number of optimization steps, default is 1000.
            **kwargs: Any - Additional keyword arguments.

        Returns:
            None
        """
        self.calculator = calculator

        self.traj_folder = traj_folder
        self.interval = interval
        self.climb = climb
        self.optimizer = get_ase_optimizer(optimizer)
        self.fmax = fmax
        self.max_steps = max_steps
        self.kwargs = kwargs
        self.get_mep = get_mep

    def calc_images(
        self,
        start_struct: Structure,
        end_struct: Structure,
        *,
        n_images: int = 7,
        interpolate_lattices: bool = False,
        autosort_tol: float = 0.5,
    ) -> dict[str, Any]:
        """
        Calculate NEB images between given start and end structures.

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
        structure: Structure | dict[str, Any],
        
    ) -> dict[str, Any]:
        """
        Calculate the energy barrier using the nudged elastic band method.

        Parameters:
            - structure: A dictionary containing the images with keys 'image0', 'image1', etc. Must be of type dict.

        Returns:
                dict:
                    - "barrier" (float): The energy barrier of the reaction pathway.
                    - "force" (float): The force exerted on the atoms during the NEB calculation.
                    - "mep" (optional, dict): If `self.get_mep` is True, a dictionary containing the images and their respective energies,
                    with keys like 'image00', 'image01', etc.
                    - "energies" (optional, list): The energy values for each image along the NEB path, if `self.get_mep` is True.
        """
        if not isinstance(structure, dict):
            raise ValueError(  # noqa:TRY004
                "For NEB calculations, structure must be a dict containing the images with keys image00, image01, etc."
            )
        images: list[Atoms] = []
        for _, image in sorted(structure.items(), key=lambda x: x[0]):
            atoms = image.to_ase_atoms() if isinstance(image, Structure) else image
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
        data = neb_tool.get_barrier() #add structures
        result = {"barrier": data[0], "force": data[1]}
        
        if self.get_mep:
            energies = fit_images(self.neb.images).energies
            mep = {f"image{i:02d}": {"structure": AseAtomsAdaptor.get_structure(image), "energy": energy} for i, (image, energy) in enumerate(zip(self.neb.images, energies))}
            result["mep"] = mep
        return result 
