"""Calculator for MD properties using LAMMPS."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsRun
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from typing import Any, Literal

    from ase import Atoms

LAMMPS_TEMPLATES_DIR = Path(__file__).parent / "lammps_templates"

# Provide simple aliases for some common models. The key in MODEL_ALIASES must be lower case.
MODEL_ALIASES = {
    "tensornet": "TensorNet-MatPES-PBE-v2025.1-PES",
    "m3gnet": "M3GNet-MatPES-PBE-v2025.1-PES",
    "chgnet": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "pbe": "TensorNet-MatPES-PBE-v2025.1-PES",
    "r2scan": "TensorNet-MatPES-r2SCAN-v2025.1-PES",
}


class LAMMPSMDCalc(PropCalc):
    """
    Class to manage molecular dynamics simulations using LAMMPS.

    :ivar calculator: Name of the potential/calculator model.
    :type calculator: str
    :ivar temperature: Simulation temperature in Kelvin.
    :type temperature: int
    :ivar ensemble: Statistical ensemble for simulation.
    :type ensemble: Literal["nve", "nvt", "npt", "nvt_nose_hoover", "npt_nose_hoover"]
    :ivar timestep: Simulation timestep in picoseconds.
    :type timestep: float
    :ivar steps: Number of MD steps.
    :type steps: int
    :ivar pressure: Simulation pressure in bars.
    :type pressure: float
    :ivar taut: Temperature coupling constant (ps).
    :type taut: float | None
    :ivar taup: Pressure coupling constant (ps).
    :type taup: float | None
    :ivar infile: File for Input.
    :type infile: str
    :ivar trajfile: File for trajectory output.
    :type trajfile: str
    :ivar logfile: Logfile path.
    :type logfile: str
    :ivar loginterval: Logging interval in steps.
    :type loginterval: int
    :ivar relax_structure: Whether to perform initial structural relaxation.
    :type relax_structure: bool
    :ivar fmax: Maximum force for relaxation convergence.
    :type fmax: float
    :ivar optimizer: Optimizer for structural relaxation.
    :type optimizer: str
    :ivar frames: Number of frames saved from the trajectory.
    :type frames: int
    :ivar settings: Settings for the template script.
    :type settings: dict | None
    :ivar relax_calc_kwargs: Additional kwargs for relaxation calculation.
    :type relax_calc_kwargs: dict | None
    """

    def __init__(
        self,
        calculator: str,
        *,
        temperature: int = 300,
        ensemble: Literal["nve", "nvt", "npt", "nvt_nose_hoover", "npt_nose_hoover"] = "nvt",
        timestep: float = 0.001,
        steps: int = 100,
        pressure: float = 1,
        taut: float | None = None,
        taup: float | None = None,
        infile: str = "in.md",
        trajfile: str = "md.lammpstrj",
        logfile: str = "log.lammps",
        loginterval: int = 1,
        relax_structure: bool = True,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        frames: int = 10,
        settings: dict | None = None,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """Initialize LAMMPSMDCalc with simulation parameters."""
        self.model_name = calculator
        self.calculator = calculator  # type: ignore[assignment]
        self.ensemble = ensemble
        self.timestep = timestep
        self.temperature = temperature
        self.pressure = pressure
        self.steps = steps
        self.taut = taut if taut is not None else 100 * timestep
        self.taup = taup if taup is not None else 1000 * timestep
        self.infile = infile
        self.trajfile = trajfile
        self.logfile = logfile
        self.loginterval = loginterval
        self.relax_structure = relax_structure
        self.fmax = fmax
        self.optimizer = optimizer
        self.frames = frames
        self.settings = settings
        self.relax_calc_kwargs = relax_calc_kwargs

        if any(self.model_name.lower().startswith(m) for m in ("m3gnet", "chgnet", "tensornet", "pbe", "r2scan")):
            self.model_name = MODEL_ALIASES.get(self.model_name.lower(), self.model_name)
            self.gnnp_type = "matgl"
        else:
            self.gnnp_type = self.model_name.lower()

        self.fix_command = self._generate_fix_command()

    def _generate_fix_command(self) -> str:
        """Generate LAMMPS fix command based on the selected ensemble."""
        if self.ensemble in ("nvt", "nvt_nose_hoover"):
            return f"fix             1 all nvt temp {self.temperature} {self.temperature} {self.taut}"
        if self.ensemble in ("npt", "npt_nose_hoover"):
            return (
                f"fix             1 all npt temp {self.temperature} {self.temperature} {self.taut} "
                f"iso {self.pressure} {self.pressure} {self.taup}"
            )
        if self.ensemble == "nve":
            return "fix             1 all nve"
        raise ValueError(
            "The specified ensemble is not supported, choose from 'nve', 'nvt',"
            " 'nvt_nose_hoover', 'npt', 'npt_nose_hoover'."
        )

    def write_inputs(
        self,
        structure: Structure | Atoms | dict[str, Any],
        script_template: str | Path = Path(LAMMPS_TEMPLATES_DIR / "md.template"),
    ) -> LammpsRun:
        """Write LAMMPS input files based on a given structure.

        Parameters:
            structure (Structure | Atoms | dict[str, Any]): Input pymatgen Structure or equivalent dictionary.
            script_template (str | Path): Template of the input script.

        Returns:
            LammpsRun: Instance representing written LAMMPS inputs.
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

        lmpdata = LammpsData.from_structure(structure_in, atom_style="atomic")

        default_settings = {
            "logfile": self.logfile,
            "trajfile": self.trajfile,
            "gnnp_type": self.gnnp_type,
            "model_name": self.model_name,
            "symbol_set": " ".join(structure_in.symbol_set),
            "temperature": self.temperature,
            "timestep": self.timestep,
            "steps": self.steps,
            "loginterval": self.loginterval,
            "taut": self.taut,
            "taup": self.taup,
            "pressure": self.pressure,
            "fix_command": self.fix_command,
        }
        full_settings = {**default_settings, **(self.settings or {})}

        if not isinstance(script_template, str):
            with open(script_template) as f:
                script_template = f.read()

        lammps_run = LammpsRun(
            script_template=script_template,
            settings=full_settings,
            data=lmpdata,
            script_filename=self.infile,
        )
        lammps_run.write_inputs(output_dir=str(Path.cwd()))
        return lammps_run

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """Run the MD calculation using LAMMPS."""
        # Preprocess the input structure using the superclass's calc method.
        # This initial processing returns a dictionary containing a "final_structure".
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if not (os.path.exists(self.infile) and os.path.exists("md.data")):
            self.write_inputs(structure_in)

        lammps_command = f"lmp < {self.infile}"
        subprocess.run(lammps_command, shell=True, check=False)  # noqa: S602

        # Create a list to record the state of atoms at each simulation step.
        traj = []
        for snapshot in parse_lammps_dumps(self.trajfile):
            lattice = snapshot.box.to_lattice()
            coords = snapshot.data[["x", "y", "z"]].to_numpy()
            species = snapshot.data["element"].tolist()
            struct = Structure(lattice, species, coords)

            traj.append(AseAtomsAdaptor.get_atoms(struct))
        # Select the last 'self.frames' frames from the trajectory for further analysis.
        traj = traj[-self.frames :]

        thermo = parse_lammps_log(self.logfile)[0]
        # Calculate the average potential energy over the selected frames.
        energy_pot = thermo["PotEng"][-self.frames :].mean()
        # Calculate the average kinetic energy over the selected frames.
        energy_kin = thermo["KinEng"][-self.frames :].mean()
        # Calculate the average total energy (potential + kinetic) over the selected frames.
        energy_tot = thermo["TotEng"][-self.frames :].mean()

        # Update the result dictionary with the simulation trajectory and computed energy.
        result |= {
            "trajectory": traj,
            "potential_energy": energy_pot,
            "kinetic_energy": energy_kin,
            "total_energy": energy_tot,
        }

        # Return the complete result dictionary.
        return result
