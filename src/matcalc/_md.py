"""Calculator for MD properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from ase import units
from ase.md import Langevin
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend._ase import TrajectoryObserver
from .utils import to_ase_atoms

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


class MDCalc(PropCalc):
    """
    Performs molecular dynamics (MD) simulations on the input structure to calculate the final energy and
    obtain the final structure. Optionally, the structure is relaxed prior to the MD simulation.
    The simulation setup is flexible, supporting various ensembles (e.g., "nvt", "nve", "npt", etc.) and
    user-defined parameters such as temperature, time step, and number of steps.
    """

    def __init__(
        self,
        calculator: Calculator,
        *,
        ensemble: Literal[
            "nve",
            "nvt",
            "nvt_nose_hoover",
            "nvt_berendsen",
            "nvt_langevin",
            "nvt_andersen",
            "nvt_bussi",
            "npt",
            "npt_nose_hoover",
            "npt_berendsen",
            "npt_inhomogeneous",
        ] = "nvt",
        temperature: int = 300,
        timestep: float = 1.0,
        steps: int = 100,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        friction: float = 1.0e-3,
        andersen_prob: float = 1.0e-2,
        ttime: float = 25.0,
        pfactor: float = 75.0**2.0,
        external_stress: float | np.ndarray | None = None,
        compressibility_au: float | None = None,
        trajfile: Any = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        mask: tuple | np.ndarray | None = None,
        relax_structure: bool = True,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        frames: int = 10,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Initializes an MDCalc instance with the specified simulation parameters and relaxation settings.

        Parameters:
            calculator (Calculator): The calculator used for energy, force, and stress evaluations.
                Default to the provided calculator.
            ensemble (str): Ensemble for MD simulation. Options include "nve", "nvt_langevin",
                "nvt_andersen", "nvt_bussi", "npt", "npt_berendsen", "npt_nose_hoover". Default to "nvt".
            temperature (int): Simulation temperature in Kelvin. Default to 300.
            timestep (float): Time step in femtoseconds. Default to 1.0.
            steps (int): Number of MD simulation steps. Default to 100.
            pressure (float): External pressure for NPT simulations (in eV/Å³). Default to 1.01325 * units.bar.
            taut (float | None): Time constant for temperature coupling. If None, defaults to 100 * timestep * fs.
            taup (float | None): Time constant for pressure coupling. If None, defaults to 1000 * timestep * fs.
            friction (float): Friction coefficient for Langevin dynamics. Default to 1.0e-3.
            andersen_prob (float): Collision probability for Andersen thermostat. Default to 1.0e-2.
            ttime (float): Characteristic time scale for the thermostat in ASE units (fs). Default to 25.0.
            pfactor (float): Barostat differential equation constant. Default to 75.0**2.0.
            external_stress (float | np.ndarray | None): External stress applied to the system.
                If not provided, defaults to 0.0.
            compressibility_au (float | None): Material compressibility in Å³/eV. Default to None.
            trajfile (Any): Trajectory object or file for storing simulation data. Default to None.
            logfile (str | None): Filename for simulation logs. Default to None.
            loginterval (int): Interval (in steps) for logging simulation data. Default to 1.
            append_trajectory (bool): Whether to append to an existing trajectory file. Default to False.
            mask (tuple | np.ndarray | None): Constraint mask for NPT simulation cell deformations.
                If not provided, defaults to np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]).
            relax_structure (bool): Whether to relax the input structure before MD simulation. Default to True.
            fmax (float): Maximum force tolerance for structure relaxation (in eV/Å). Default to 0.1.
            optimizer (str): Optimizer used for structure relaxation. Default to "FIRE".
            frames (int): Number of MD frames for analysis. Default to 10.
            relax_calc_kwargs (dict | None): Additional keyword arguments for the relaxation calculation.
                Default to None.
        """
        self.calculator = calculator
        self.ensemble = ensemble
        self.temperature = temperature
        self.timestep = timestep
        self.steps = steps
        self.pressure = pressure
        self.taut = taut
        self.taup = taup
        self.friction = friction
        self.andersen_prob = andersen_prob
        self.ttime = ttime
        self.pfactor = pfactor
        self.external_stress = external_stress
        self.compressibility_au = compressibility_au
        self.trajfile = trajfile
        self.logfile = logfile
        self.loginterval = loginterval
        self.append_trajectory = append_trajectory
        self.mask = mask
        self.relax_structure = relax_structure
        self.fmax = fmax
        self.optimizer = optimizer
        self.frames = frames
        self.relax_calc_kwargs = relax_calc_kwargs

    def _initialize_md(self, atoms: Atoms) -> Any:
        """
        Initializes the MD simulation object based on the provided ASE atoms object and simulation parameters.

        Parameters:
            atoms (Atoms): ASE atoms object representing the structure.

        Returns:
            An MD simulation object (e.g., an instance of NVTBerendsen, VelocityVerlet, etc.)
        """
        atoms.calc = self.calculator

        timestep_fs = self.timestep * units.fs
        taut = self.taut if self.taut is not None else 100 * self.timestep * units.fs
        taup = self.taup if self.taup is not None else 1000 * self.timestep * units.fs
        mask = self.mask if self.mask is not None else np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        external_stress = self.external_stress if self.external_stress is not None else 0.0
        md: Any

        if self.ensemble.lower() == "nve":
            md = VelocityVerlet(
                atoms,
                timestep_fs,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        elif self.ensemble.lower() == "nvt" or self.ensemble.lower() == "nvt_nose_hoover":
            self._upper_triangular_cell(atoms)
            md = NPT(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                externalstress=external_stress,  # type: ignore[arg-type]
                ttime=self.ttime * units.fs,
                pfactor=None,  # Disable pressure coupling to convert to NVT ensemble
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
                mask=mask,
            )
        elif self.ensemble.lower() == "nvt_berendsen":
            md = NVTBerendsen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                taut=taut,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        elif self.ensemble.lower() == "nvt_langevin":
            md = Langevin(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                friction=self.friction,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        elif self.ensemble.lower() == "nvt_andersen":
            md = Andersen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                andersen_prob=self.andersen_prob,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        elif self.ensemble.lower() == "nvt_bussi":
            md = Bussi(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                taut=taut,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        elif self.ensemble.lower() == "npt" or self.ensemble.lower() == "npt_nose_hoover":
            self._upper_triangular_cell(atoms)
            md = NPT(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                externalstress=external_stress,  # type: ignore[arg-type]
                ttime=self.ttime * units.fs,
                pfactor=self.pfactor * units.fs,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
                mask=mask,
            )
        elif self.ensemble.lower() == "npt_berendsen":
            md = NPTBerendsen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                pressure_au=self.pressure,
                taut=taut,
                taup=taup,
                compressibility_au=self.compressibility_au,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        elif self.ensemble.lower() == "npt_inhomogeneous":
            md = Inhomogeneous_NPTBerendsen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                pressure_au=self.pressure,
                taut=taut,
                taup=taup,
                compressibility_au=self.compressibility_au,
                trajectory=self.trajfile,
                logfile=self.logfile,
                loginterval=self.loginterval,
                append_trajectory=self.append_trajectory,
            )
        else:
            raise ValueError(
                "The specified ensemble is not supported, choose from 'nve', 'nvt',"
                " 'nvt_nose_hoover', 'nvt_berendsen', 'nvt_langevin', 'nvt_andersen',"
                " 'nvt_bussi', 'npt', 'npt_nose_hoover', 'npt_berendsen', 'npt_inhomogeneous'."
            )
        return md

    def _upper_triangular_cell(self, atoms: Atoms) -> None:
        """
        Transforms the cell of the given atoms object to an upper triangular form if it is not already,
        as required by the Nose-Hoover NPT implementation.

        Parameters:
            atoms (Atoms): ASE atoms object.
        """
        if not atoms.cell[1, 0] == atoms.cell[2, 0] == atoms.cell[2, 1] == 0.0:
            a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _ = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = np.sqrt(1 - cos_p**2)
            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]
            atoms.set_cell(new_basis, scale_atoms=True)

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Performs the MD simulation calculation for the input structure. Prior to generating initial velocities,
        this method calls the superclass's calc method for preprocessing and optionally relaxes the structure.
        It then initializes the atomic velocities using the Maxwell-Boltzmann distribution, runs the MD simulation,
        and returns the final energy along with the final atomic configuration.

        Parameters:
            structure (Structure | Atoms | dict[str, Any]): Input structure as a Structure instance or a dictionary.

        Returns:
            dict: A dictionary containing the final energy and the final atomic configuration.
                  It includes the keys "final_structure" and "energy".
        """
        # Preprocess the input structure using the superclass's calc method.
        # This initial processing returns a dictionary containing a "final_structure".
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        # If structure relaxation is enabled, relax the structure (atoms only) before the MD simulation.
        if self.relax_structure:
            # Create a RelaxCalc instance with the specified calculator, convergence criteria (fmax),
            # optimizer, and any additional keyword arguments for the relaxation calculation.
            relaxer = RelaxCalc(
                self.calculator,
                fmax=self.fmax,
                optimizer=self.optimizer,
                relax_atoms=True,
                relax_cell=False,
                **(self.relax_calc_kwargs or {}),
            )
            # Run the relaxation calculation and update the result dictionary.
            result |= relaxer.calc(structure_in)
            # Update the input structure with the relaxed final structure.
            structure_in = result["final_structure"]

        # Convert the structure to an ASE atoms object,
        # which is required for subsequent molecular dynamics (MD) simulation.
        atoms = to_ase_atoms(structure_in)

        # Initialize the atomic velocities based on the Maxwell-Boltzmann distribution
        # at the specified temperature, ensuring proper kinetic energy.
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        # Initialize the molecular dynamics (MD) simulation and set up the simulation parameters.
        md = self._initialize_md(atoms)

        # Attach a callback to the MD simulation to record the atoms' state at intervals defined by self.loginterval.
        traj = TrajectoryObserver(atoms)
        md.attach(traj)

        # Run the MD simulation for the specified number of steps.
        md.run(self.steps)

        traj = traj.get_slice(slice(-self.frames, len(traj), 1))

        # Calculate the average potential energy over the selected frames.
        energy_pot = sum(traj.potential_energies) / self.frames
        # Calculate the average kinetic energy over the selected frames.
        energy_kin = sum(traj.kinetic_energies) / self.frames
        # Calculate the average total energy (potential + kinetic) over the selected frames.
        energy_tot = sum(traj.total_energies) / self.frames

        # Update the result dictionary with the simulation trajectory and computed energy.
        result |= {
            "trajectory": traj,
            "potential_energy": energy_pot,
            "kinetic_energy": energy_kin,
            "total_energy": energy_tot,
        }

        # Return the complete result dictionary.
        return result
