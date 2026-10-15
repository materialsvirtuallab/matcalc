"""Calculator for ionic diffusivity and conductivity via MD and DiffusionAnalyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase import units

from ._base import PropCalc
from ._md import MDCalc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

_ASE_ENSEMBLES = Literal[
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
    "npt_mtk",
    "npt_isotropic_mtk",
]


class DiffusionCalc(PropCalc):
    """Calculate ionic diffusivity and conductivity from an MD trajectory.

    Runs MD with :class:`MDCalc` and analyses the equilibrated trajectory with
    pymatgen's DiffusionAnalyzer. An input dict containing ``trajectory`` skips
    MD and uses those frames directly. Requires the ``diffusion`` extra.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        specie: str,
        temperature: float = 300,
        ensemble: _ASE_ENSEMBLES = "nvt_nose_hoover",
        timestep: float = 1.0,
        steps: int = 100_000,
        loginterval: int = 10,
        equilibration_frames: int = 500,
        taut: float = 100.0,
        seed: int | None = 42,
        relax_structure: bool = False,
        set_com_stationary: bool = True,
        set_zero_rotation: bool = True,
        relax_calc_kwargs: dict | None = None,
        analyzer_kwargs: dict[str, Any] | None = None,
        **md_kwargs: Any,
    ) -> None:
        """Initialize the MD protocol and diffusion analysis.

        Args:
            calculator: ASE calculator instance or universal model name.
            specie: Symbol of the mobile ion species, e.g. ``"Li"``.
            temperature: Simulation temperature in Kelvin.
            ensemble: Ensemble supported by MDCalc.
            timestep: MD time step in femtoseconds.
            steps: Total number of MD steps; defaults to 100 ps at a 1 fs step.
            loginterval: Steps between saved frames, also used as the analyzer's
                ``step_skip``.
            equilibration_frames: Leading saved frames discarded by the internal MD run.
            taut: Thermostat time constant in femtoseconds.
            seed: Seed for initial velocities on each MD run. None uses a random seed.
            relax_structure: Whether to relax the structure before MD.
            set_com_stationary: Remove initial center-of-mass momentum.
            set_zero_rotation: Remove initial angular momentum.
            relax_calc_kwargs: Extra arguments for RelaxCalc when relaxation is enabled.
            analyzer_kwargs: Extra arguments for DiffusionAnalyzer, e.g. ``smoothed``.
            **md_kwargs: Additional arguments forwarded to MDCalc.
        """
        self.calculator = calculator
        self.specie = specie
        self.temperature = temperature
        self.ensemble = ensemble
        self.timestep = timestep
        self.steps = steps
        self.loginterval = loginterval
        self.equilibration_frames = equilibration_frames
        self.taut = taut
        self.seed = seed
        self.relax_structure = relax_structure
        self.set_com_stationary = set_com_stationary
        self.set_zero_rotation = set_zero_rotation
        self.relax_calc_kwargs = relax_calc_kwargs
        self.analyzer_kwargs = analyzer_kwargs or {}
        self.md_kwargs = md_kwargs

    def _build_md_calc(self) -> MDCalc:
        """Build the MD calculator with a fresh initial-velocity generator."""
        return MDCalc(
            self.calculator,
            ensemble=self.ensemble,
            temperature=self.temperature,
            timestep=self.timestep,
            steps=self.steps,
            taut=self.taut * units.fs,
            rng=np.random.RandomState(self.seed),
            loginterval=self.loginterval,
            equilibration_frames=self.equilibration_frames,
            relax_structure=self.relax_structure,
            set_com_stationary=self.set_com_stationary,
            set_zero_rotation=self.set_zero_rotation,
            relax_calc_kwargs=self.relax_calc_kwargs,
            **self.md_kwargs,
        )

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """Compute diffusivity and conductivity.

        Args:
            structure: Structure, ASE Atoms, or a result dict. If the dict contains
                ``trajectory`` (ASE Atoms or pymatgen Structures), all supplied frames
                are analysed without MD or further equilibration trimming. Set
                temperature, timestep and loginterval to match that trajectory.

        Returns:
            Dict with ``diffusivity`` (cm²/s), ``conductivity`` (mS/cm),
            ``final_structure``, ``trajectory`` and ``_units``, preserving upstream
            fields and the energy outputs from any internal MD run.
        """
        from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

        result = super().calc(structure)
        if "trajectory" not in result:
            result.update(self._build_md_calc().calc(result))
        trajectory = [to_pmg_structure(frame) for frame in result["trajectory"]]
        analyzer = DiffusionAnalyzer.from_structures(
            trajectory,
            specie=self.specie,
            temperature=self.temperature,
            time_step=self.timestep,  # type: ignore[arg-type]
            step_skip=self.loginterval,
            **self.analyzer_kwargs,
        )
        result["diffusivity"] = float(analyzer.diffusivity)
        result["conductivity"] = float(analyzer.conductivity)
        result["_units"] = self._merge_units(
            result.get("_units"),
            {"diffusivity": "cm^2/s", "conductivity": "mS/cm"},
        )
        return result
