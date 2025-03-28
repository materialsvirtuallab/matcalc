"""Calculator for phonon-phonon interaction and related properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


@dataclass
class Phonon3Calc(PropCalc):
    """Calculator for phonon-phonon interaction and related properties.

    Args:
        calculator (Calculator): ASE Calculator to use.
        fmax (float): Maximum force tolerance (in eV/Angstrom). More stringent than for simple relaxation.
            Defaults to 0.1.
        optimizer (str): Optimizer used for structure relaxation in RelaxCalc.
        fc2_supercell (ArrayLike): Supercell matrix for 2nd order force constants. Defaults to a 2x2x2 supercell.
        fc3_supercell (ArrayLike): Supercell matrix for 3rd order force constants. Defaults to a 2x2x2 supercell.
        mesh_numbers (ArrayLike): Sampling mesh numbers along reciprocal axes. Defaults to (20, 20, 20).
        t_step (float): Temperature step (in Kelvin).
        t_max (float): Maximum temperature (in Kelvin).
        t_min (float): Minimum temperature (in Kelvin).
        relax_structure (bool): Whether to relax the structure before phonon calculations.
            Set to False if the provided structure is already relaxed with the same calculator.
        relax_calc_kwargs (dict): Additional arguments for RelaxCalc when relax_structure is True.
        disp_kwargs (dict): Additional arguments for generate_displacements.
        thermal_conductivity_kwargs (dict): Additional arguments for run_thermal_conductivity.
        write_phonon3 (bool | str | Path): Whether to save the Phono3py object.
            Set to True to save with default filename, or pass a string/Path for a custom filename.
            Defaults to False.
        write_kappa (bool): Whether to save thermal conductivity related properties. Defaults to False.
    """

    calculator: Calculator
    fc2_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    fc3_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    mesh_numbers: ArrayLike = (20, 20, 20)
    disp_kwargs: dict | None = None
    thermal_conductivity_kwargs: dict | None = None
    relax_structure: bool = True
    relax_calc_kwargs: dict | None = None
    fmax: float = 0.1
    optimizer: str = "FIRE"
    t_min: float = 0
    t_max: float = 1000
    t_step: float = 10
    write_phonon3: bool | str | Path = False
    write_kappa: bool = False

    def __post_init__(self) -> None:
        """Set default paths for saving output files."""
        # Map True to canonical default path, False to "" and leave Path/string unchanged.
        for key, val, default_path in (("write_phonon3", self.write_phonon3, "phonon3.yaml"),):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:
        """Calculate phonon-phonon interactions and related properties.

        Args:
            structure: A pymatgen Structure object or dictionary that can be converted to one.

        Returns:
            dict: {
                "phonon3": Phono3py object with force constants produced,
                "temperatures": list of temperatures (in Kelvin),
                "thermal_conductivity": list of average lattice thermal conductivity under relaxation time
                    approximation at corresponding temperatures (in Watts/meter/Kelvin),
            }
        """
        try:
            from phono3py import Phono3py
        except ImportError as e:
            raise ImportError(
                "Need to install phono3py before using Phonon3Calc. "
                "I personally recommend using 'conda install -c conda-forge phono3py' for installation. "
                "See https://phonopy.github.io/phono3py/install.html for full instructions."
            ) from e

        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator,
                fmax=self.fmax,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        cell = get_phonopy_structure(structure_in)
        phonon3 = Phono3py(
            unitcell=cell,
            supercell_matrix=self.fc3_supercell,
            phonon_supercell_matrix=self.fc2_supercell,
            primitive_matrix="auto",
        )  # type: ignore[arg-type]

        if self.mesh_numbers is not None:
            phonon3.mesh_numbers = self.mesh_numbers

        phonon3.generate_displacements(**self.disp_kwargs)

        num_atoms2 = len(phonon3.phonon_supercells_with_displacements[0])
        phonon_forces = []
        for supercell in phonon3.phonon_supercells_with_displacements:
            if supercell is not None:
                struct_supercell = get_pmg_structure(supercell)
                atoms_supercell = AseAtomsAdaptor.get_atoms(struct_supercell)
                atoms_supercell.calc = self.calculator
                f = atoms_supercell.get_forces()
            else:
                f = np.zeros((num_atoms2, 3))
            phonon_forces.append(f)
        fc2_set = np.array(phonon_forces)
        phonon3.phonon_forces = fc2_set

        num_atoms3 = len(phonon3.supercells_with_displacements[0])
        forces = []
        for supercell in phonon3.supercells_with_displacements:
            if supercell is not None:
                struct_supercell = get_pmg_structure(supercell)
                atoms_supercell = AseAtomsAdaptor.get_atoms(struct_supercell)
                atoms_supercell.calc = self.calculator
                f = atoms_supercell.get_forces()
            else:
                f = np.zeros((num_atoms3, 3))
            forces.append(f)
        fc3_set = np.array(forces)
        phonon3.forces = fc3_set

        phonon3.produce_fc2(symmetrize_fc2=True)
        phonon3.produce_fc3(symmetrize_fc3r=True)
        phonon3.init_phph_interaction()

        temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)
        phonon3.run_thermal_conductivity(
            temperatures=temperatures,
            **self.thermal_conductivity_kwargs,
            write_kappa=self.write_kappa,
        )

        kappa = np.asarray(phonon3.thermal_conductivity.kappa_TOT_RTA)
        kappa_ave = np.nan if kappa.size == 0 or np.any(np.isnan(kappa)) else kappa[..., :3].mean(axis=-1)

        if self.write_phonon3:
            phonon3.save(filename=self.write_phonon3)

        return {
            "phonon3": phonon3,
            "temperatures": temperatures,
            "thermal_conductivity": np.squeeze(kappa_ave),
        }


def _calc_forces(calculator: Calculator, supercell: PhonopyAtoms) -> ArrayLike:
    """Helper to compute forces on a structure.

    Args:
        calculator: ASE Calculator.
        supercell: Supercell from phonopy.

    Returns:
        ArrayLike: Forces on the atoms.
    """
    struct = get_pmg_structure(supercell)
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator
    return atoms.get_forces()
