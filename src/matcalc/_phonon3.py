"""Calculator for phonon-phonon interaction and related properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from phono3py import Phono3py
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class Phonon3Calc(PropCalc):
    """
    Class for calculating thermal conductivity using third-order force constants.

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
        fc2_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fc3_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
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
        self.fc2_supercell = fc2_supercell
        self.fc3_supercell = fc3_supercell
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

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
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

        cell = get_phonopy_structure(to_pmg_structure(structure_in))
        phonon3 = Phono3py(
            unitcell=cell,
            supercell_matrix=self.fc3_supercell,
            phonon_supercell_matrix=self.fc2_supercell,
            primitive_matrix="auto",
        )  # type: ignore[arg-type]

        if self.mesh_numbers is not None:
            phonon3.mesh_numbers = self.mesh_numbers

        phonon3.generate_displacements(**self.disp_kwargs)

        phonon_forces = []
        for supercell in phonon3.phonon_supercells_with_displacements:
            struct_supercell = get_pmg_structure(supercell)
            r = run_pes_calc(struct_supercell, self.calculator)
            f = r.forces
            phonon_forces.append(f)
        fc2_set = np.array(phonon_forces)
        phonon3.phonon_forces = fc2_set

        forces = []
        for supercell in phonon3.supercells_with_displacements:
            if supercell is not None:
                struct_supercell = get_pmg_structure(supercell)
                r = run_pes_calc(struct_supercell, self.calculator)
                f = r.forces
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
