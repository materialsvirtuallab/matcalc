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


@pytest.mark.parametrize(
    ("ensemble", "expected_energy"),
    [
        ("nve", -10.74606),
        ("nvt", -10.81289),
        ("nvt_berendsen", -10.73112),
        ("nvt_langevin", -10.78238),
        ("nvt_andersen", -10.82750),
        ("nvt_bussi", -10.77756),
        ("npt_inhomogeneous", -10.74737),
        ("npt_berendsen", -10.74714),
        ("npt_nose_hoover", -10.86120),
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
        taut=0.1,
        taup=0.1,
        steps=10,
        frames=5,
        compressibility_au=1,
        logfile=log_file,
        trajfile=traj_file,
    )
    results = md_calc.calc(Si)

    assert isinstance(results, dict)

    assert "trajectory" in results
    assert "potential_energy" in results
    assert "kinetic_energy" in results
    assert "total_energy" in results

    assert results["total_energy"] == pytest.approx(expected_energy, rel=1e-1)

    energies = np.array(results["trajectory"].total_energies)

    if ensemble != "nve":
        assert not np.allclose(energies - energies[0], 0, atol=1e-9), f"Energies are too close for {ensemble}"

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
