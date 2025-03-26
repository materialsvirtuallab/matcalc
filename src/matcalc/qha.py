"""Calculator for phonon properties under quasi-harmonic approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from phonopy import PhonopyQHA

from .base import PropCalc
from .phonon import PhononCalc
from .relaxation import RelaxCalc

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


@dataclass
class QHACalc(PropCalc):
    """Calculator for phonon properties under quasi-harmonic approximation.

    Args:
        calculator (Calculator): ASE Calculator to use.
        t_step (float): Temperature step. Defaults to 10 (in Kelvin).
        t_max (float): Max temperature (in Kelvin). Defaults to 1000 (in Kelvin).
        t_min (float): Min temperature (in Kelvin). Defaults to 0 (in Kelvin).
        fmax (float): Max forces. This criterion is more stringent than for simple relaxation.
            Defaults to 0.1 (in eV/Angstrom).
        optimizer (str): Optimizer used for RelaxCalc. Default to "FIRE".
        eos (str): Equation of state used to fit F vs V, including "vinet", "murnaghan" or
            "birch_murnaghan". Default to "vinet".
        relax_structure (bool): Whether to first relax the structure. Set to False if structures
            provided are pre-relaxed with the same calculator.
        relax_calc_kwargs (dict): Arguments to be passed to the RelaxCalc, if relax_structure is True.
        phonon_calc_kwargs (dict): Arguments to be passed to the PhononCalc.
        scale_factors (Sequence[float]): Factors to scale the lattice constants of the structure.
        write_helmholtz_volume (bool | str | Path): Whether to save Helmholtz free energy vs volume in file.
            Pass string or Path for custom filename. Defaults to False.
        write_volume_temperature (bool | str | Path): Whether to save equilibrium volume vs temperature in file.
            Pass string or Path for custom filename. Defaults to False.
        write_thermal_expansion (bool | str | Path): Whether to save thermal expansion vs temperature in file.
            Pass string or Path for custom filename. Defaults to False.
        write_gibbs_temperature (bool | str | Path): Whether to save Gibbs free energy vs temperature in file.
            Pass string or Path for custom filename. Defaults to False.
        write_bulk_modulus_temperature (bool | str | Path): Whether to save bulk modulus vs temperature in file.
            Pass string or Path for custom filename. Defaults to False.
        write_heat_capacity_p_numerical (bool | str | Path): Whether to save heat capacity at constant pressure
            by numerical difference vs temperature in file. Pass string or Path for custom filename.
            Defaults to False.
        write_heat_capacity_p_polyfit (bool | str | Path): Whether to save heat capacity at constant pressure
            by fitting vs temperature in file. Pass string or Path for custom filename. Defaults to False.
        write_gruneisen_temperature (bool | str | Path): Whether to save Grueneisen parameter vs temperature in
            file. Pass string or Path for custom filename. Defaults to False.
    """

    calculator: Calculator
    t_step: float = 10
    t_max: float = 1000
    t_min: float = 0
    fmax: float = 0.1
    optimizer: str = "FIRE"
    eos: str = "vinet"
    relax_structure: bool = True
    relax_calc_kwargs: dict | None = None
    phonon_calc_kwargs: dict | None = None
    scale_factors: Sequence[float] = (
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
        1.00,
        1.01,
        1.02,
        1.03,
        1.04,
        1.05,
    )
    write_helmholtz_volume: bool | str | Path = False
    write_volume_temperature: bool | str | Path = False
    write_thermal_expansion: bool | str | Path = False
    write_gibbs_temperature: bool | str | Path = False
    write_bulk_modulus_temperature: bool | str | Path = False
    write_heat_capacity_p_numerical: bool | str | Path = False
    write_heat_capacity_p_polyfit: bool | str | Path = False
    write_gruneisen_temperature: bool | str | Path = False

    def __post_init__(self) -> None:
        """Set default paths for where to save output files."""
        # map True to canonical default path, False to "" and Path to str
        for key, val, default_path in (
            (
                "write_helmholtz_volume",
                self.write_helmholtz_volume,
                "helmholtz_volume.dat",
            ),
            (
                "write_volume_temperature",
                self.write_volume_temperature,
                "volume_temperature.dat",
            ),
            (
                "write_thermal_expansion",
                self.write_thermal_expansion,
                "thermal_expansion.dat",
            ),
            (
                "write_gibbs_temperature",
                self.write_gibbs_temperature,
                "gibbs_temperature.dat",
            ),
            (
                "write_bulk_modulus_temperature",
                self.write_bulk_modulus_temperature,
                "bulk_modulus_temperature.dat",
            ),
            (
                "write_heat_capacity_p_numerical",
                self.write_heat_capacity_p_numerical,
                "Cp_temperature.dat",
            ),
            (
                "write_heat_capacity_p_polyfit",
                self.write_heat_capacity_p_polyfit,
                "Cp_temperature_polyfit.dat",
            ),
            (
                "write_gruneisen_temperature",
                self.write_gruneisen_temperature,
                "gruneisen_temperature.dat",
            ),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy under quasi-harmonic approximation.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            "qha": Phonopy.qha object,
            "scale_factors": List of scale factors of lattice constants,
            "volumes": List of unit cell volumes at corresponding scale factors (in Angstrom^3),
            "electronic_energies": List of electronic energies at corresponding volumes (in eV),
            "temperatures": List of temperatures in ascending order (in Kelvin),
            "thermal_expansion_coefficients": List of volumetric thermal expansion coefficients at corresponding
                temperatures (in Kelvin^-1),
            "gibbs_free_energies": List of Gibbs free energies at corresponding temperatures (in eV),
            "bulk_modulus_P": List of bulk modulus at constant pressure at corresponding temperatures (in GPa),
            "heat_capacity_P": List of heat capacities at constant pressure at corresponding temperatures (in J/K/mol),
            "gruneisen_parameters": List of Gruneisen parameters at corresponding temperatures,
        }
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

        temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)
        volumes, electronic_energies, free_energies, entropies, heat_capacities = self._collect_properties(structure_in)

        qha = self._create_qha(volumes, electronic_energies, temperatures, free_energies, entropies, heat_capacities)  # type: ignore[arg-type]

        self._write_output_files(qha)

        return result | self._generate_output_dict(qha, volumes, electronic_energies, temperatures)  # type: ignore[arg-type]

    def _collect_properties(self, structure: Structure) -> tuple[list, list, list, list, list]:
        """Helper to collect properties like volumes, electronic energies, and thermal properties.

        Args:
            structure: Pymatgen structure for which the properties need to be calculated.

        Returns:
            Tuple containing lists of volumes, electronic energies, free energies, entropies,
                and heat capacities for different scale factors.
        """
        volumes = []
        electronic_energies = []
        free_energies = []
        entropies = []
        heat_capacities = []
        for scale_factor in self.scale_factors:
            struct = self._scale_structure(structure, scale_factor)
            volumes.append(struct.volume)
            electronic_energies.append(self._calculate_energy(struct))
            thermal_properties = self._calculate_thermal_properties(struct)
            free_energies.append(thermal_properties["free_energy"])
            entropies.append(thermal_properties["entropy"])
            heat_capacities.append(thermal_properties["heat_capacity"])
        return volumes, electronic_energies, free_energies, entropies, heat_capacities

    def _scale_structure(self, structure: Structure, scale_factor: float) -> Structure:
        """Helper to scale the lattice of a structure.

        Args:
            structure: Pymatgen structure to be scaled.
            scale_factor: Factor by which the lattice constants are scaled.

        Returns:
            Pymatgen structure with scaled lattice constants.
        """
        struct = structure.copy()
        struct.apply_strain(scale_factor - 1)
        return struct

    def _calculate_energy(self, structure: Structure) -> float:
        """Helper to calculate the electronic energy of a structure.

        Args:
            structure: Pymatgen structure for which the energy is calculated.

        Returns:
            Electronic energy of the structure.
        """
        static_calc = RelaxCalc(self.calculator, relax_atoms=False, relax_cell=False)
        return static_calc.calc(structure)["energy"]

    def _calculate_thermal_properties(self, structure: Structure) -> dict:
        """Helper to calculate the thermal properties of a structure.

        Args:
            structure: Pymatgen structure for which the thermal properties are calculated.

        Returns:
            Dictionary of thermal properties containing free energies, entropies and heat capacities.
        """
        phonon_calc = PhononCalc(
            self.calculator,
            t_step=self.t_step,
            t_max=self.t_max,
            t_min=self.t_min,
            relax_structure=False,
            write_phonon=False,
            **(self.phonon_calc_kwargs or {}),
        )
        return phonon_calc.calc(structure)["thermal_properties"]

    def _create_qha(
        self,
        volumes: list,
        electronic_energies: list,
        temperatures: list,
        free_energies: list,
        entropies: list,
        heat_capacities: list,
    ) -> PhonopyQHA:
        """Helper to create a PhonopyQHA object for quasi-harmonic approximation.

        Args:
            volumes: List of volumes corresponding to different scale factors.
            electronic_energies: List of electronic energies corresponding to different volumes.
            temperatures: List of temperatures in ascending order (in Kelvin).
            free_energies: List of free energies corresponding to different volumes and temperatures.
            entropies: List of entropies corresponding to different volumes and temperatures.
            heat_capacities: List of heat capacities corresponding to different volumes and temperatures.

        Returns:
            Phonopy.qha object.
        """
        return PhonopyQHA(
            volumes=volumes,
            electronic_energies=electronic_energies,
            temperatures=temperatures,
            free_energy=np.transpose(free_energies),
            entropy=np.transpose(entropies),
            cv=np.transpose(heat_capacities),
            eos=self.eos,
            t_max=self.t_max,
        )

    def _write_output_files(self, qha: PhonopyQHA) -> None:
        """Helper to write various output files based on the QHA calculation.

        Args:
            qha: Phonopy.qha object
        """
        if self.write_helmholtz_volume:
            qha.write_helmholtz_volume(filename=self.write_helmholtz_volume)
        if self.write_volume_temperature:
            qha.write_volume_temperature(filename=self.write_volume_temperature)
        if self.write_thermal_expansion:
            qha.write_thermal_expansion(filename=self.write_thermal_expansion)
        if self.write_gibbs_temperature:
            qha.write_gibbs_temperature(filename=self.write_gibbs_temperature)
        if self.write_bulk_modulus_temperature:
            qha.write_bulk_modulus_temperature(filename=self.write_bulk_modulus_temperature)
        if self.write_heat_capacity_p_numerical:
            qha.write_heat_capacity_P_numerical(filename=self.write_heat_capacity_p_numerical)
        if self.write_heat_capacity_p_polyfit:
            qha.write_heat_capacity_P_polyfit(filename=self.write_heat_capacity_p_polyfit)
        if self.write_gruneisen_temperature:
            qha.write_gruneisen_temperature(filename=self.write_gruneisen_temperature)

    def _generate_output_dict(
        self, qha: PhonopyQHA, volumes: list, electronic_energies: list, temperatures: list
    ) -> dict:
        """Helper to generate the output dictionary after QHA calculation.

        Args:
            qha: Phonopy.qha object.
            volumes: List of volumes corresponding to different scale factors.
            electronic_energies: List of electronic energies corresponding to different volumes.
            temperatures: List of temperatures in ascending order (in Kelvin).

        Returns:
            Dictionary containing the results of QHA calculation.
        """
        return {
            "qha": qha,
            "scale_factors": self.scale_factors,
            "volumes": volumes,
            "electronic_energies": electronic_energies,
            "temperatures": temperatures,
            "thermal_expansion_coefficients": qha.thermal_expansion,
            "gibbs_free_energies": qha.gibbs_temperature,
            "bulk_modulus_P": qha.bulk_modulus_temperature,
            "heat_capacity_P": qha.heat_capacity_P_polyfit,
            "gruneisen_parameters": qha.gruneisen_temperature,
        }
