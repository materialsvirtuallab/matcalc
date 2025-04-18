"""Calculator for phonon properties using pheasy."""

from __future__ import annotations

import logging
import pickle
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.file_IO import write_FORCE_CONSTANTS as write_force_constants
from phonopy.interface.vasp import write_vasp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

# import pymatgen libraries to determine supercell
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class PheasyCalc(PropCalc):
    """
    A class for phonon and thermal property calculations using phonopy.

    The `PhononCalc` class extends the `PropCalc` class to provide
    functionalities for calculating phonon properties and thermal properties
    of a given structure using the phonopy library. It includes options for
    structural relaxation before phonon property determination, as well as
    methods to export calculated properties to various output files for
    further analysis or visualization.

    :ivar calculator: A calculator object or a string specifying the
        computational backend to be used.
    :type calculator: Calculator | str
    :ivar atom_disp: Magnitude of atomic displacements for phonon
        calculations.
    :type atom_disp: float
    :ivar supercell_matrix: Array defining the transformation matrix to
        construct supercells for phonon calculations.
    :type supercell_matrix: ArrayLike
    :ivar t_step: Temperature step for thermal property calculations in
        Kelvin.
    :type t_step: float
    :ivar t_max: Maximum temperature for thermal property calculations in
        Kelvin.
    :type t_max: float
    :ivar t_min: Minimum temperature for thermal property calculations in
        Kelvin.
    :type t_min: float
    :ivar fmax: Maximum force convergence criterion for structural relaxation.
    :type fmax: float
    :ivar optimizer: String specifying the optimizer type to be used for
        structural relaxation.
    :type optimizer: str
    :ivar relax_structure: Boolean flag to determine whether to relax the
        structure before phonon calculation.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Optional dictionary containing additional
        arguments for the structural relaxation calculation.
    :type relax_calc_kwargs: dict | None
    :ivar write_force_constants: Path, boolean, or string specifying whether
        to write the calculated force constants to an output file, and the
        path or name of the file if applicable.
    :type write_force_constants: bool | str | Path
    :ivar write_band_structure: Path, boolean, or string specifying whether
        to write the calculated phonon band structure to an output file,
        and the path or name of the file if applicable.
    :type write_band_structure: bool | str | Path
    :ivar write_total_dos: Path, boolean, or string specifying whether to
        write the calculated total density of states (DOS) to an output
        file, and the path or name of the file if applicable.
    :type write_total_dos: bool | str | Path
    :ivar write_phonon: Path, boolean, or string specifying whether to write
        the calculated phonon properties (e.g., phonon.yaml) to an output
        file, and the path or name of the file if applicable.
    :type write_phonon: bool | str | Path
    :ivar fitting_method: Method for fitting force constants. Options are
        "FDM", "LASSO", or "MD".
    :type fitting_method: str
    :ivar num_harmonic_snapshots: Number of snapshots for harmonic fitting.
        If None, it is set to double the number of displacements.
    :type num_harmonic_snapshots: Optional[int]
    :ivar num_anharmonic_snapshots: Number of snapshots for anharmonic
        fitting. If None, it is set to ten times the number of displacements.
    :type num_anharmonic_snapshots: Optional[int]
    :ivar calc_anharmonic: Boolean flag to determine whether to perform
        anharmonic calculations.
    :type calc_anharmonic: bool
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        atom_disp: float = 0.015,
        # supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        supercell_matrix: ArrayLike | None = None,
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        write_force_constants: bool | str | Path = False,
        write_band_structure: bool | str | Path = False,
        write_total_dos: bool | str | Path = False,
        write_phonon: bool | str | Path = True,
        fitting_method: str = "LASSO",
        num_harmonic_snapshots: int | None = None,
        num_anharmonic_snapshots: int | None = None,
        calc_anharmonic: bool = False,
    ) -> None:
        """
        Initializes the class with configuration for the phonon calculations. The initialization parameters control
        the behavior of structural relaxation, thermal properties, force calculations, and output file generation.
        The class allows for customization of key parameters to facilitate the study of material behaviors.

        :param calculator: The calculator object or string name specifying the calculation backend to use.
        :param atom_disp: Atom displacement to be used for finite difference calculation of force constants.
        :param supercell_matrix: Transformation matrix to define the supercell for the calculation.
        :param t_step: Temperature step for thermal property calculations.
        :param t_max: Maximum temperature for thermal property calculations.
        :param t_min: Minimum temperature for thermal property calculations.
        :param fmax: Maximum force during structure relaxation, used as a convergence criterion.
        :param optimizer: Name of the optimization algorithm for structural relaxation.
        :param relax_structure: Flag to indicate whether structure relaxation should be performed before calculations.
        :param relax_calc_kwargs: Additional keyword arguments for relaxation phase calculations.
        :param write_force_constants: File path or boolean flag to write force constants.
            Defaults to "force_constants".
        :param write_band_structure: File path or boolean flag to write band structure data.
            Defaults to "band_structure.yaml".
        :param write_total_dos: File path or boolean flag to write total density of states (DOS) data.
            Defaults to "total_dos.dat".
        :param write_phonon: File path or boolean flag to write phonon data. Defaults to "phonon.yaml".
        :param fitting_method: Method for fitting force constants. Options are "FDM", "LASSO", or "MD".
        :param num_harmonic_snapshots: Number of snapshots for harmonic fitting. If None, it is set to double the
            number of displacements.
        :param num_anharmonic_snapshots: Number of snapshots for anharmonic fitting. If None, it is set to ten times
            the number of displacements.
        :param calc_anharmonic: Flag to indicate whether anharmonic calculations should be performed.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.atom_disp = atom_disp
        self.supercell_matrix = supercell_matrix
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.fmax = fmax
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.write_force_constants = write_force_constants
        self.write_band_structure = write_band_structure
        self.write_total_dos = write_total_dos
        self.write_phonon = write_phonon

        # some new parameters for pheasy
        self.fitting_method = fitting_method
        self.num_harmonic_snapshots = num_harmonic_snapshots
        self.num_anharmonic_snapshots = num_anharmonic_snapshots
        self.calc_anharmonic = calc_anharmonic

        # Set default paths for output files.
        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            phonon: Phonopy object with force constants produced
            thermal_properties:
                {
                    temperatures: list of temperatures in Kelvin,
                    free_energy: list of Helmholtz free energies at corresponding temperatures in kJ/mol,
                    entropy: list of entropies at corresponding temperatures in J/K/mol,
                    heat_capacity: list of heat capacities at constant volume at corresponding temperatures in J/K/mol,
                    The units are originally documented in phonopy.
                    See phonopy.Phonopy.run_thermal_properties()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/api_phonopy.py#L2591)
                    -> phonopy.phonon.thermal_properties.ThermalProperties.run()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L498)
                    -> phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_free_energy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L217)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_entropy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L233)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_heat_capacity()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L225)
                }
        }
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator, fmax=self.fmax, optimizer=self.optimizer, **(self.relax_calc_kwargs or {})
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]
        cell = get_phonopy_structure(structure_in)

        # If the supercell matrix is not provided, we need to determine the
        # supercell matrix from the structure. We use the
        # CubicSupercellTransformation to determine the supercell matrix.
        # The supercell matrix is a 3x3 matrix that defines the transformation
        # from the primitive cell to the supercell. The supercell matrix is
        # used to generate the supercell for the phonon calculations.
        if self.supercell_matrix is None:
            transformation = CubicSupercellTransformation(min_length=12.0, force_diagonal=True)
            supercell = transformation.apply_transformation(structure_in)
            self.supercell_matrix = np.array(transformation.transformation_matrix.transpose().tolist())
            # transfer it to array

            # self.supercell_matrix = supercell.lattice.matrix
        else:
            transformation = None

        phonon = phonopy.Phonopy(cell, self.supercell_matrix)  # type: ignore[arg-type]

        if self.fitting_method == "FDM":
            phonon.generate_displacements(distance=self.atom_disp)

        elif self.fitting_method == "LASSO":
            if self.num_harmonic_snapshots is None:
                phonon.generate_displacements(distance=self.atom_disp)

                self.num_harmonic_snapshots = len(phonon.displacements) * 2

                phonon.generate_displacements(
                    distance=self.atom_disp, number_of_snapshots=self.num_harmonic_snapshots, random_seed=42
                )

        elif self.fitting_method == "MD":
            # pass
            # phonon.generate_displacements(distance=self.atom_disp, number_of_snapshots=self.num_snapshots)
            print("MD fitting method is not implemented yet.")

        else:
            raise ValueError(f"Unknown fitting method: {self.fitting_method}")

        disp_supercells = phonon.supercells_with_displacements

        disp_array = []

        phonon.forces = [  # type: ignore[assignment]
            _calc_forces(self.calculator, supercell)
            for supercell in disp_supercells  # type:ignore[union-attr]
            if supercell is not None
        ]

        force_equilibrium = _calc_forces(self.calculator, phonon.supercell)  # type: ignore[union-attr]
        phonon.forces = np.array(phonon.forces) - force_equilibrium  # type: ignore[assignment]

        for i, supercell in enumerate(disp_supercells):
            disp = supercell.get_positions() - phonon.supercell.get_positions()
            disp_array.append(np.array(disp))

        print(disp_array)
        print("Forces calculated for the supercells.")
        print("Producing force constants...")
        disp_array = np.array(disp_array)

        with open("disp_matrix.pkl", "wb") as file:
            pickle.dump(disp_array, file)
        with open("force_matrix.pkl", "wb") as file:
            pickle.dump(phonon.forces, file)

        supercell = phonon.get_supercell()
        write_vasp("POSCAR", cell)
        write_vasp("SPOSCAR", supercell)

        num_har = disp_array.shape[0]
        supercell_matrix = self.supercell_matrix
        symprec = 1e-3

        logger.info("start running pheasy for second order force constants in cluster")

        pheasy_cmd_1 = (
            f'pheasy --dim "{int(supercell_matrix[0][0])}" "{int(supercell_matrix[1][1])}" '
            f'"{int(supercell_matrix[2][2])}" -s -w 2 --symprec "{float(symprec)}" --nbody 2'
        )

        # Create the null space to further reduce the free parameters for
        # specific force constants and make them physically correct.
        pheasy_cmd_2 = (
            f'pheasy --dim "{int(supercell_matrix[0][0])}" "{int(supercell_matrix[1][1])}" '
            f'"{int(supercell_matrix[2][2])}" -c --symprec "{float(symprec)}" -w 2'
        )

        # Generate the Compressive Sensing matrix,i.e., displacement matrix
        # for the input of machine leaning method.i.e., LASSO,
        pheasy_cmd_3 = (
            f'pheasy --dim "{int(supercell_matrix[0][0])}" "{int(supercell_matrix[1][1])}" '
            f'"{int(supercell_matrix[2][2])}" -w 2 -d --symprec "{float(symprec)}" '
            f'--ndata "{int(num_har)}" --disp_file'
        )

        pheasy_cmd_4 = (
            f'pheasy --dim "{int(supercell_matrix[0][0])}" "{int(supercell_matrix[1][1])}" '
            f'"{int(supercell_matrix[2][2])}" -f --full_ifc -w 2 --symprec "{float(symprec)}" '
            f'--rasr BHH --ndata "{int(num_har)}"'
        )

        logger.info("Start running pheasy in cluster")

        subprocess.call(pheasy_cmd_1, shell=True)
        subprocess.call(pheasy_cmd_2, shell=True)
        subprocess.call(pheasy_cmd_3, shell=True)
        subprocess.call(pheasy_cmd_4, shell=True)

        force_constants = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS")
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        print("Force constants produced.")
        print("Running phonon calculations...")
        phonon.run_mesh()
        phonon.run_thermal_properties(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        if self.write_force_constants:
            write_force_constants(phonon.force_constants, filename=self.write_force_constants)
        if self.write_band_structure:
            phonon.auto_band_structure(write_yaml=True, filename=self.write_band_structure)
        if self.write_total_dos:
            phonon.auto_total_dos(write_dat=True, filename=self.write_total_dos)
        if self.write_phonon:
            phonon.save(filename=self.write_phonon)

        logger.info("Phonon calculations finished.")
        logger.info("Calculating anharmonic force constants...")

        # If the anharmonic calculation is requested, we need to
        # generate the displacements and forces for the supercells.
        # The displacements are generated with the same distance as
        # the harmonic calculation, but the number of snapshots is
        # set to 10 times the number of displacements.
        # The forces are calculated with the same calculator as the
        # harmonic calculation, but the forces are shifted by the
        # equilibrium forces of the supercell.
        # The displacements are saved in a file called disp_matrix.pkl
        # and the forces are saved in a file called force_matrix.pkl.
        # The files are used by the pheasy command to calculate the
        # anharmonic force constants.

        if self.calc_anharmonic:
            logger.info("Calculating anharmonic properties...")
            subprocess.call("rm -f disp_matrix.pkl force_matrix.pkl ", shell=True)
            self.atom_disp = 0.1
            if self.num_anharmonic_snapshots is None:
                phonon.generate_displacements(distance=self.atom_disp)
                self.num_anharmonic_snapshots = len(phonon.displacements) * 10
                phonon.generate_displacements(
                    distance=self.atom_disp, number_of_snapshots=self.num_anharmonic_snapshots, random_seed=42
                )
            else:
                phonon.generate_displacements(
                    distance=self.atom_disp, number_of_snapshots=self.num_anharmonic_snapshots, random_seed=42
                )
            disp_supercells = phonon.supercells_with_displacements
            disp_array = []
            phonon.forces = [  # type: ignore[assignment]
                _calc_forces(self.calculator, supercell)
                for supercell in disp_supercells  # type:ignore[union-attr]
                if supercell is not None
            ]
            force_equilibrium = _calc_forces(self.calculator, phonon.supercell)
            phonon.forces = np.array(phonon.forces) - force_equilibrium
            for i, supercell in enumerate(disp_supercells):
                disp = supercell.get_positions() - phonon.supercell.get_positions()
                disp_array.append(np.array(disp))
            disp_array = np.array(disp_array)
            with open("disp_matrix.pkl", "wb") as file:
                pickle.dump(disp_array, file)
            with open("force_matrix.pkl", "wb") as file:
                pickle.dump(phonon.forces, file)
            num_anh = disp_array.shape[0]
            supercell_matrix = self.supercell_matrix
            symprec = 1e-3
            pheasy_cmd_5 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -s -w 4 --symprec "
                f"{float(symprec)} "
                f"--nbody 2 3 3 --c3 6.3 "
                f"--c4 5.3"
            )
            logger.info("pheasy_cmd_5 = %s", pheasy_cmd_5)

            pheasy_cmd_6 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -c --symprec "
                f"{float(symprec)} -w 4"
            )
            logger.info("pheasy_cmd_6 = %s", pheasy_cmd_6)
            pheasy_cmd_7 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -w 4 -d --symprec "
                f"{float(symprec)} "
                f"--ndata {int(num_anh)} --disp_file"
            )
            logger.info("pheasy_cmd_7 = %s", pheasy_cmd_7)
            pheasy_cmd_8 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f -w 4 --fix_fc2 "
                f"--symprec {float(symprec)} "
                f"--ndata {int(num_anh)} "
                f"-l LASSO --std"
            )
            logger.info("pheasy_cmd_8 = %s", pheasy_cmd_8)
            logger.info("Start running pheasy in cluster")

            subprocess.call(pheasy_cmd_5, shell=True)
            subprocess.call(pheasy_cmd_6, shell=True)
            subprocess.call(pheasy_cmd_7, shell=True)
            subprocess.call(pheasy_cmd_8, shell=True)

        return result | {"phonon": phonon, "thermal_properties": phonon.get_thermal_properties_dict()}


def _calc_forces(calculator: Calculator, supercell: PhonopyAtoms) -> ArrayLike:
    """Helper to compute forces on a structure.

    Args:
        calculator: ASE Calculator
        supercell: Supercell from phonopy.

    Return:
        forces
    """
    struct = get_pmg_structure(supercell)
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator
    return atoms.get_forces()
