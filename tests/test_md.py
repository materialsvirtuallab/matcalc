"""Tests for MDCalc class"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from matcalc import MDCalc

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.fixture(scope="module", autouse=True)
def set_seed() -> None:
    np.random.seed(42)  # noqa: NPY002


@pytest.mark.parametrize(
    ("ensemble", "expected_energy"),
    [
        ("nve", -10.819535868347996),
        ("nvt", -10.84265926910263),
        ("nvt_berendsen", -10.82033001817293),
        ("nvt_langevin", -10.774126994388347),
        ("nvt_andersen", -10.827990433233442),
        ("nvt_bussi", -10.772954463144568),
        ("npt_inhomogeneous", -10.8222801574377),
        ("npt_berendsen", -10.801131048759578),
        ("npt_nose_hoover", -10.776961885598102),
        ("npt_isotropic_mtk", -10.802633196032712),
        ("npt_mtk", -10.819341706561369),
    ],
)
def test_md_calc(
    Si: Structure,
    matpes_calculator: PESCalculator,
    ensemble: str,
    expected_energy: float,
) -> None:
    """Tests for MDCalc class"""
    # Note: fmax is set relatively high for testing purposes only.

    log_file = f"{ensemble}.log"
    traj_file = f"{ensemble}.traj"

    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble=ensemble,
        temperature=300,
        steps=10,
        frames=5,
        compressibility_au=1,
        logfile=log_file,
        trajfile=traj_file,
    )
    initial_vol = Si.lattice.volume
    results = md_calc.calc(Si)

    assert isinstance(results, dict)

    assert "trajectory" in results
    assert "potential_energy" in results
    assert "kinetic_energy" in results
    assert "total_energy" in results

    assert results["total_energy"] == pytest.approx(expected_energy, rel=1e-2)

    energies = np.array(results["trajectory"].total_energies)

    if ensemble != "nve":
        assert not np.allclose(energies - energies[0], 0, atol=1e-9), f"Energies are too close for {ensemble}"

    if ensemble.startswith("nvt"):
        # There should be no volume change for NVT simulations.
        assert np.linalg.det(results["trajectory"].cells[-1]) == pytest.approx(initial_vol, rel=1e-2)

    assert len(results["trajectory"]) == 5

    # Verify that the log file and trajectory file have been created
    assert os.path.isfile(str(log_file))
    assert os.path.isfile(str(traj_file))


def test_md_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for MDCalc class"""
    # Note: fmax is set relatively high for testing purposes only.
    md_calc = MDCalc(
        calculator=matpes_calculator,
        temperature=300,
        taut=0.1,
        taup=0.1,
        steps=10,
        frames=5,
        compressibility_au=1,
    )
    results = md_calc.calc(Si_atoms)

    assert isinstance(results, dict)

def test_md_relax_cell(
    Si: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for MDCalc class with cell relaxation"""
    # Note: fmax is set relatively high for testing purposes only.

    # default behavior (relax_cell = False)
    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble="npt_mtk",
        temperature=300,
        steps=0,
        compressibility_au=1,
        logfile="test.log",
        trajfile="test.traj,
    )
    initial_vol = Si.lattice.volume
    results = md_calc.calc(Si)
    volume_after_relax = np.linalg.det(results["trajectory"].cells[0])
    assert volume_after_relax == pytest.approx(initial_vol)

    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble="npt_mtk",
        temperature=300,
        steps=0,
        compressibility_au=1,
        logfile="test.log",
        trajfile="test.traj,
        relax_calc_kwargs={"relax_cell": True}
    )
    initial_vol = Si.lattice.volume
    results = md_calc.calc(Si)
    volume_after_relax = np.linalg.det(results["trajectory"].cells[0])
    assert volume_after_relax != pytest.approx(initial_vol, rel=0.1)


def test_invalid_ensemble(Si: Structure, matpes_calculator: PESCalculator) -> None:
    with pytest.raises(
        ValueError,
        match="The specified ensemble is not supported, choose from 'nve', 'nvt',"
        " 'nvt_nose_hoover', 'nvt_berendsen', 'nvt_langevin', 'nvt_andersen',"
        " 'nvt_bussi', 'npt', 'npt_nose_hoover', 'npt_berendsen', 'npt_inhomogeneous'.",
    ):
        MDCalc(
            calculator=matpes_calculator,
            ensemble="you_know_who",
            temperature=300,
            taut=0.1,
            taup=0.1,
            compressibility_au=10,
            steps=10,
            logfile=None,
            trajfile=None,
        ).calc(Si)
