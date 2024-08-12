"""Calculator for phonon properties under quasi-harmonic approxiamtion."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import copy
from phonopy import PhonopyQHA

from .base import PropCalc
from .relaxation import RelaxCalc
from .phonon import PhononCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

@dataclass
class QhaPhononCalc(PropCalc):
    """Calculator for phonon properties under quasi-harmonic approxiamtion.

    Args:
        calculator (Calculator): ASE Calculator to use.
        t_step (float): Temperature step. Defaults to 10 (in Kelvin)
        t_max (float): Max temperature (in Kelvin). Defaults to 1000 (in Kelvin)
        t_min (float): Min temperature (in Kelvin). Defaults to 0 (in Kelvin)
        fmax (float): Max forces. This criterion is more stringent than for simple relaxation.
            Defaults to 0.1 (in eV/Angstrom)
        optimizer (str): Optimizer used for RelaxCalc. Default to "FIRE"
        eos (str): Equation of state used to fit F vs V, including "vinet", "murnaghan" or 
            "birch_murnaghan". Default to "vinet".
        relax_structure (bool): Whether to first relax the structure. Set to False if structures
            provided are pre-relaxed with the same calculator.
        relax_calc_kwargs (dict): Arguments to be passed to the RelaxCalc, if relax_structure is True.
        phonon_calc_kwargs (dict): Arguments to be passed to the PhononCalc.
        scale_factors (list[float]): Factors to scale the lattice constants of the structure.
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
        write_heat_capacity_P_numerical (bool | str | Path): Whether to save heat capacity at constant pressure
            by numerical difference vs temperature in file. Pass string or Path for custom filename. 
            Defaults to False.
        write_heat_capacity_P_polyfit (bool | str | Path): Whether to save heat capacity at constant pressure
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
    scale_factors: list[float] = field(default_factory=lambda: np.arange(0.95, 1.05, 0.01).tolist())
    write_helmholtz_volume: bool | str | Path = False
    write_volume_temperature: bool | str | Path = False
    write_thermal_expansion: bool | str | Path = False
    write_gibbs_temperature: bool | str | Path = False
    write_bulk_modulus_temperature: bool | str | Path = False
    write_heat_capacity_P_numerical: bool | str | Path = False
    write_heat_capacity_P_polyfit: bool | str | Path = False
    write_gruneisen_temperature: bool | str | Path = False
    
    def __post_init__(self) -> None:
        """Set default paths for where to save output files."""
        # map True to canonical default path, False to "" and Path to str
        for key, val, default_path in (
            ("write_helmholtz_volume", self.write_helmholtz_volume, "helmholtz_volume.dat"),
            ("write_volume_temperature", self.write_volume_temperature, "volume_temperature.dat"),
            ("write_thermal_expansion", self.write_thermal_expansion, "thermal_expansion.dat"),
            ("write_write_gibbs_temperature", self.write_gibbs_temperature, "gibbs_temperature.dat"),
            ("write_bulk_modulus_temperature", self.write_bulk_modulus_temperature, "bulk_modulus_temperature.dat"),
            ("write_heat_capacity_P_numerical", self.write_heat_capacity_P_numerical, "Cp_temperature.dat"),
            ("write_heat_capacity_P_polyfit", self.write_heat_capacity_P_polyfit, "Cp_temperature_polyfit.dat"),
            ("write_gruneisen_temperature", self.write_gruneisen_temperature, "gruneisen_temperature.dat"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy under quasi-harmonic approxiamtion.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            "scale_factors": list of scale factors of lattice constants,
            "volumes": list of unit cell volumes at corresponding scale factors (in Angstrom^3),
            "electronic_energies": list of electronic energies at corresponding volumes (in eV),
            "temperatures": list of temperatures in ascending order (in Kelvin),
            "thermal_expansion_coefficients": list of volumetric thermal expansion coefficients at corresponding 
                temperatures (in Kelvin^-1),
            "gibbs_free_energies": list of Gibbs free energies at corresponding temperatures (in eV),
            "bulk_modulus_P": list of bulk modulus at constant presure at corresponding temperatures (in GPa),
            "heat_capacity_P": list of heat capacities at constant pressure at corresponding temperatures (in J/K/mol),
            "gruneisen_parameters": list of Gruneisen parameters at corresponding temperatures,
        }
        """
        volumes = []
        electronic_energies = []
        temperatures = np.arange(self.t_min, self.t_max+self.t_step, self.t_step)

        free_energies = []
        entropies = []
        heat_capacities = []

        if self.relax_structure:
                relaxer = RelaxCalc(
                    self.calculator, fmax=self.fmax, optimizer=self.optimizer, **(self.relax_calc_kwargs or {})
                )
                structure = relaxer.calc(structure)["final_structure"]

        for scale_factor in self.scale_factors:
            struct = copy.deepcopy(structure)
            struct.scale_lattice(struct.volume*scale_factor**3)
            
            static_calc = RelaxCalc(
                    self.calculator, relax_atoms=False, relax_cell=False)            
            volumes.append(struct.volume)
            electronic_energies.append(static_calc.calc(struct)["energy"])

            phonon_calc = PhononCalc(
                self.calculator, t_step=self.t_step, t_max=self.t_max, t_min=self.t_min, relax_structure=False, write_phonon=False, **(self.phonon_calc_kwargs or {})
            )
            thermal_properties = phonon_calc.calc(struct)["thermal_properties"]
            free_energies.append(thermal_properties["free_energy"])
            entropies.append(thermal_properties["entropy"])
            heat_capacities.append(thermal_properties["heat_capacity"])
            
        qha = PhonopyQHA(
            volumes=volumes,
            electronic_energies=electronic_energies,
            temperatures=temperatures,
            free_energy=np.transpose(free_energies),
            entropy=np.transpose(entropies),
            cv=np.transpose(heat_capacities),
            eos=self.eos,
            t_max=self.t_max
        )

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
        if self.write_heat_capacity_P_numerical:
            qha.write_heat_capacity_P_numerical(filename=self.write_heat_capacity_P_numerical)
        if self.write_heat_capacity_P_polyfit:
            qha.write_heat_capacity_P_polyfit(filename=self.write_heat_capacity_P_polyfit)
        if self.write_gruneisen_temperature:
            qha.write_gruneisen_temperature(filename=self.write_gruneisen_temperature)
        
        return {"scale_factors": self.scale_factors,
                "volumes": volumes,
                "electronic_energies": electronic_energies,
                "temperatures": temperatures,
                "thermal_expansion_coefficients": qha.thermal_expansion,
                "gibbs_free_energies": qha.gibbs_temperature,
                "bulk_modulus_P": qha.bulk_modulus_temperature,
                "heat_capacity_P": qha.heat_capacity_P_polyfit,
                "gruneisen_parameters": qha.gruneisen_temperature,
        }
