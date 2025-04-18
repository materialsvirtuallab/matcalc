"""Calculator for phonon-phonon interaction and related properties using FOURPHONON and FCs from Pheasy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import subprocess


import numpy as np
from pymatgen.io.phonopy import get_phonopy_structure
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
import phonopy
try:
    import f90nml
except ImportError:
    f90nml = None


from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class FourphononCalc(PropCalc):
    """
    Class for calculating thermal conductivity using third-order and fourth-order FCs from Pheasy.

    This class integrates with the Phono3py library to compute thermal conductivity
    based on third-order force constants (FC3). It includes capabilities for optional
    structure relaxation, displacement generation, and force calculations on
    supercells. Results include the thermal conductivity as a function of temperature
    and other intermediate configurations used in the calculation.

    :ivar calculator: Calculator used to compute forces for the atomic structure.
    :type calculator: Calculator | str
    :ivar fc2_supercell: Transformation matrix defining the supercell for second-order
        force constants.
    :type fc2_supercell: ArrayLike
    :ivar fc3_supercell: Transformation matrix defining the supercell for third-order
        force constants.
    :type fc3_supercell: ArrayLike
    :ivar mesh_numbers: Mesh grid dimensions for phonon calculations.
    :type mesh_numbers: ArrayLike
    :ivar disp_kwargs: Keyword arguments for displacement generation.
    :type disp_kwargs: dict[str, Any] | None
    :ivar thermal_conductivity_kwargs: Keyword arguments for thermal conductivity
        calculations.
    :type thermal_conductivity_kwargs: dict | None
    :ivar relax_structure: Flag indicating if the structure needs to be relaxed before
        calculations.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Additional arguments for the relaxation calculator.
    :type relax_calc_kwargs: dict | None
    :ivar fmax: Maximum force criterion for structure relaxation.
    :type fmax: float
    :ivar optimizer: Optimizer name for structure relaxation.
    :type optimizer: str
    :ivar t_min: Minimum temperature for thermal conductivity calculations.
    :type t_min: float
    :ivar t_max: Maximum temperature for thermal conductivity calculations.
    :type t_max: float
    :ivar t_step: Step size for temperature in thermal conductivity calculations.
    :type t_step: float
    :ivar write_phonon3: Path or flag for saving the Phono3py output object.
    :type write_phonon3: bool | str | Path
    :ivar write_kappa: Flag indicating if the thermal conductivity results
        should be written to file.
    :type write_kappa: bool
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        min_length: float = 12,
        force_diagonal: bool = True,
        supercell_matrix: ArrayLike | None = None,
        mesh_numbers: ArrayLike = (20, 20, 20),
        disp_kwargs: dict[str, Any] | None = None,
        thermal_conductivity_kwargs: dict | None = None,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        write_phonon3: bool | str | Path = False,
        write_kappa: bool = False,
    ) -> None:
        """
        Initializes the class for thermal conductivity calculation and structure relaxation
        utilizing third-order force constants (fc3). The class provides configurable
        parameters for the relaxation process, supercell settings, thermal conductivity
        calculation, and file output management.

        :param calculator: The calculator instance or string indicating the method to be
                           used for energy and force calculations.
        :param fc2_supercell: The supercell matrix for generating second-order force constants.
        :param fc3_supercell: The supercell matrix for generating third-order force constants.
        :param mesh_numbers: The grid size for reciprocal space mesh used in phonon calculations.
        :param disp_kwargs: Dictionary containing optional parameters for displacement generation
                            in force constant calculation.
        :param thermal_conductivity_kwargs: Dictionary containing optional parameters for thermal
                                            conductivity calculation.
        :param relax_structure: Boolean flag to enable or disable structure relaxation before
                                force constant calculations.
        :param relax_calc_kwargs: Dictionary containing additional configuration options for the
                                  relaxation calculation.
        :param fmax: The maximum force allowed on atoms during the structure relaxation
                     procedure. Determines the relaxation termination condition.
        :param optimizer: The optimizer algorithm to use for structure relaxation. Defaults to "FIRE."
        :param t_min: The minimum temperature (in Kelvin) for thermal conductivity calculations.
        :param t_max: The maximum temperature (in Kelvin) for thermal conductivity calculations.
        :param t_step: The temperature step size for thermal conductivity calculations.
        :param write_phonon3: Specifies the filename or a boolean flag for writing the
                              third-order force constants to a file. If True, writes to
                              "phonon3.yaml" by default.
        :param write_kappa: Boolean flag to enable or disable saving thermal conductivity
                            results to a file.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.min_length = min_length
        self.force_diagonal = force_diagonal
        self.supercell_matrix = supercell_matrix

        self.mesh_numbers = mesh_numbers
        self.disp_kwargs = disp_kwargs if disp_kwargs is not None else {}
        self.thermal_conductivity_kwargs = (
            thermal_conductivity_kwargs if thermal_conductivity_kwargs is not None else {}
        )
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs if relax_calc_kwargs is not None else {}
        self.fmax = fmax
        self.optimizer = optimizer
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        self.write_phonon3 = write_phonon3
        self.write_kappa = write_kappa

        # Set default paths for saving output files.
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

        if self.supercell_matrix is None:
            transformation = CubicSupercellTransformation(
                min_length=self.min_length, force_diagonal=self.force_diagonal
            )
            supercell = transformation.apply_transformation(structure_in)
            self.supercell_matrix = np.array(transformation.transformation_matrix.transpose().tolist())
            # transfer it to array

            # self.supercell_matrix = supercell.lattice.matrix
        else:
            transformation = None

        phonon = phonopy.Phonopy(cell, self.supercell_matrix)  # type: ignore[arg-type]
        supercell = phonon.get_supercell()

        # begin to write a control file for shengbte software

        allocations = {
            'nelements': 4,
            'natoms': 20,
            'ngrid': [5, 10, 5],
            'norientations': 0
            }

        # Namelist: &crystal
        crystal = {
            'lfactor': 0.1,
            'lattvec': [
                [8.7276802063, 0.0000000000, 0.0000000000],
                [0.0000000000, 4.0527501106, 0.0000000000],
                [0.0000000000, 0.0000000000, 12.9853725433]
                ],
                'elements': ["Bi", "Se", "Cu", "Cl"],
                'types': [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4],
                'positions': [
                    [0.268440098, 0.250000000, 0.956108034],
                    [0.731559873, 0.750000000, 0.043891955],
                    [0.231559902, 0.750000000, 0.456108034],
                    [0.768440127, 0.250000000, 0.543891966],
                    [0.239286929, 0.750000000, 0.808299184],
                    [0.760713100, 0.250000000, 0.191700861],
                    [0.260713071, 0.250000000, 0.308299154],
                    [0.739286900, 0.750000000, 0.691700876],
                    [0.462546855, 0.750000000, 0.705428481],
                    [0.537453115, 0.250000000, 0.294571549],
                    [0.037453145, 0.250000000, 0.205428436],
                    [0.962546885, 0.750000000, 0.794571579],
                    [0.435688823, 0.250000000, 0.597031415],
                    [0.564311147, 0.750000000, 0.402968556],
                    [0.064311177, 0.750000000, 0.097031437],
                    [0.935688853, 0.250000000, 0.902968585],
                    [0.561768413, 0.250000000, 0.898596883],
                    [0.438231587, 0.750000000, 0.101403147],
                    [0.938231587, 0.750000000, 0.398596853],
                    [0.061768413, 0.250000000, 0.601403117],
                    ],
                    'scell': [2, 4, 1]
                    }

        # Namelist: &parameters
        parameters = {
            'T': 300,
            'scalebroad': 0.1
            }

        # Namelist: &flags

        flags = {
            'four_phonon': True,
            'nonanalytic': True,
            'convergence': True,
            'nanowires': False
            }

        # Combine all namelists
        namelists = {
            'allocations': allocations,
            'crystal': crystal,
            'parameters': parameters,
            'flags': flags
            }

        # Write to CONTROL file
        f90nml.write(namelists, 'CONTROL', force=True)

        print("CONTROL file successfully written.")




        return {
            "phonon3": None,
            "temperatures": None,
            "thermal_conductivity": np.squeeze(np.full((1, 1), np.nan)),
        }
