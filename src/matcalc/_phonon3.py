"""Calculator for phonon-phonon interaction and related properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from phono3py import Phono3py
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


@dataclass
class Phonon3Calc(PropCalc):
    """
    Handles the calculation of phonon-phonon interactions and thermal conductivity
    using the Phono3py package. Provides functionality for generating displacements,
    calculating force constants, and computing lattice thermal conductivity.

    Primarily designed for automating the usage of Phono3py in conjunction with
    a chosen calculator for force evaluations. Supports options for structure
    relaxation before calculation and customization of various computational parameters.

    :ivar calculator: ASE Calculator used for force evaluations during the calculation.
    :type calculator: Calculator
    :ivar fc2_supercell: Supercell matrix for the second-order force constants calculation.
    :type fc2_supercell: ArrayLike
    :ivar fc3_supercell: Supercell matrix for the third-order force constants calculation.
    :type fc3_supercell: ArrayLike
    :ivar mesh_numbers: Grid mesh numbers for thermal conductivity calculation.
    :type mesh_numbers: ArrayLike
    :ivar disp_kwargs: Custom keyword arguments for generating displacements.
    :type disp_kwargs: dict | None
    :ivar thermal_conductivity_kwargs: Additional keyword arguments for thermal conductivity calculations.
    :type thermal_conductivity_kwargs: dict | None
    :ivar relax_structure: Flag indicating whether the input structure should be relaxed before calculation.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Additional keyword arguments for the structure relaxation calculator.
    :type relax_calc_kwargs: dict | None
    :ivar fmax: Maximum force tolerance for the structure relaxation.
    :type fmax: float
    :ivar optimizer: Optimizer name to use for structure relaxation.
    :type optimizer: str
    :ivar t_min: Minimum temperature (in Kelvin) for thermal conductivity calculation.
    :type t_min: float
    :ivar t_max: Maximum temperature (in Kelvin) for thermal conductivity calculation.
    :type t_max: float
    :ivar t_step: Temperature step size (in Kelvin) for thermal conductivity calculation.
    :type t_step: float
    :ivar write_phonon3: Output path for saving Phono3py results, or a boolean to toggle saving.
    :type write_phonon3: bool | str | Path
    :ivar write_kappa: Flag indicating whether to write kappa (thermal conductivity values) to output files.
    :type write_kappa: bool
    """

    calculator: Calculator
    fc2_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    fc3_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    mesh_numbers: ArrayLike = (20, 20, 20)
    disp_kwargs: dict[str, Any] = field(default_factory=dict)
    thermal_conductivity_kwargs: dict = field(default_factory=dict)
    relax_structure: bool = True
    relax_calc_kwargs: dict = field(default_factory=dict)
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
        """
        Performs thermal conductivity calculations using the Phono3py library.

        This method processes a given atomic structure and calculates its thermal
        conductivity through third-order force constants (FC3) computations. The
        process involves optional relaxation of the input structure, generation of
        displacements, and force calculations corresponding to the supercell
        structures. The results include computed thermal conductivity over specified
        temperatures, along with intermediate Phono3py configurations.

        :param structure: The atomic structure to compute thermal conductivity for. This can
            be provided as either a `Structure` object or a dictionary describing
            the structure as per specifications of the input format.
        :return: A dictionary containing the following keys:
            - "phonon3": The configured and processed Phono3py object containing data
              regarding the phonon interactions and force constants.
            - "temperatures": A numpy array of temperatures over which thermal
              conductivity has been computed.
            - "thermal_conductivity": The averaged thermal conductivity values computed
              at the specified temperatures. Returns NaN if the values cannot be
              computed.
        """
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

        len(phonon3.phonon_supercells_with_displacements[0])
        phonon_forces = []
        for supercell in phonon3.phonon_supercells_with_displacements:
            struct_supercell = get_pmg_structure(supercell)
            atoms_supercell = AseAtomsAdaptor.get_atoms(struct_supercell)
            atoms_supercell.calc = self.calculator
            f = atoms_supercell.get_forces()

            phonon_forces.append(f)
        fc2_set = np.array(phonon_forces)
        phonon3.phonon_forces = fc2_set

        len(phonon3.supercells_with_displacements[0])
        forces = []
        for supercell in phonon3.supercells_with_displacements:
            if supercell is not None:
                struct_supercell = get_pmg_structure(supercell)
                atoms_supercell = AseAtomsAdaptor.get_atoms(struct_supercell)
                atoms_supercell.calc = self.calculator
                f = atoms_supercell.get_forces()
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
