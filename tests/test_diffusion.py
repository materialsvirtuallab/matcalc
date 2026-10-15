"""Tests for DiffusionCalc."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase import units
from ase.calculators.calculator import Calculator
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.constants import Boltzmann, elementary_charge

from matcalc import DiffusionCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize("as_atoms", [False, True])
@pytest.mark.parametrize(
    ("temperature", "timestep", "loginterval"),
    [(600, 1, 10), (1200, 0.5, 10), (600, 1, 20)],
)
def test_existing_trajectory(
    Li2O: Structure,
    monkeypatch: pytest.MonkeyPatch,
    as_atoms: bool,
    temperature: float,
    timestep: float,
    loginterval: int,
) -> None:
    trajectory = []
    mobile_indices = [i for i, site in enumerate(Li2O) if site.specie.symbol == "Li"]
    for i in range(16):
        frame = Li2O.copy()
        frame.translate_sites(mobile_indices, [np.sqrt(0.01 * i), 0, 0], frac_coords=False)
        trajectory.append(AseAtomsAdaptor.get_atoms(frame) if as_atoms else frame)

    def unexpected_md(_self: DiffusionCalc) -> None:
        pytest.fail("An existing trajectory must not start MD.")

    monkeypatch.setattr(DiffusionCalc, "_build_md_calc", unexpected_md)
    calc = DiffusionCalc(
        Calculator(),
        specie="Li",
        temperature=temperature,
        timestep=timestep,
        loginterval=loginterval,
        analyzer_kwargs={"smoothed": False},
    )
    upstream = {
        "final_structure": Li2O,
        "trajectory": trajectory,
        "potential_energy": -10.0,
        "_units": {"potential_energy": "eV"},
    }
    result = calc.calc(upstream)

    # MSD rises by 0.01 Å² per saved frame; D = slope / 6, with Å²/fs -> cm²/s.
    diffusivity = 0.01 / (timestep * loginterval * 60)
    number_density = len(mobile_indices) / (Li2O.volume * 1e-30)
    conductivity = number_density * elementary_charge**2 * (diffusivity * 1e-4) / (Boltzmann * temperature)
    assert result["diffusivity"] == pytest.approx(diffusivity)
    assert result["conductivity"] == pytest.approx(conductivity * 10)
    assert result["trajectory"] is trajectory
    assert result["final_structure"] is Li2O
    assert result["potential_energy"] == -10.0
    assert result["_units"] == {"potential_energy": "eV", "diffusivity": "cm^2/s", "conductivity": "mS/cm"}
    assert upstream["_units"] == {"potential_energy": "eV"}
    assert "diffusivity" not in upstream


def test_diffusion_defaults() -> None:
    md_calc = DiffusionCalc(Calculator(), specie="Li")._build_md_calc()  # noqa: SLF001
    assert md_calc.ensemble == "nvt_nose_hoover"
    assert md_calc.steps == 100_000
    assert md_calc.timestep == 1.0
    assert md_calc.loginterval == 10
    assert md_calc.equilibration_frames == 500
    assert md_calc.taut == pytest.approx(100 * units.fs)
    assert not md_calc.relax_structure
    assert md_calc.set_com_stationary
    assert md_calc.set_zero_rotation


def test_diffusion_md(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    calc = DiffusionCalc(
        matpes_calculator,
        specie="Li",
        temperature=600.5,
        steps=200,
        equilibration_frames=5,
        analyzer_kwargs={"smoothed": False},
    )
    original_coords = Li2O.cart_coords.copy()
    upstream = {"final_structure": Li2O, "label": "short MD", "_units": {"volume": "Å^3"}}
    result = calc.calc(upstream)

    # Reference setup follows run_worker.py's MD protocol, shortened to 200 steps.
    atoms = AseAtomsAdaptor.get_atoms(Li2O)
    atoms.calc = matpes_calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.5, rng=np.random.RandomState(42))
    Stationary(atoms)
    ZeroRotation(atoms)
    md = NoseHooverChainNVT(atoms, timestep=units.fs, temperature_K=600.5, tdamp=100 * units.fs)
    reference = []
    md.attach(lambda: reference.append(atoms.copy()), interval=10)
    md.run(200)

    assert len(result["trajectory"]) == 16
    for actual, expected in zip(result["trajectory"], reference[5:], strict=True):
        np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-10)
        np.testing.assert_allclose(actual.get_momenta(), expected.get_momenta(), atol=1e-10)
    np.testing.assert_allclose(result["final_structure"].cart_coords, reference[-1].positions)
    np.testing.assert_array_equal(Li2O.cart_coords, original_coords)
    assert result["label"] == "short MD"
    assert result["_units"]["volume"] == "Å^3"
    assert result["_units"]["total_energy"] == "eV"
    assert np.isfinite(result["total_energy"])
    assert result["diffusivity"] >= 0
    assert result["conductivity"] >= 0
    assert "trajectory" not in upstream
