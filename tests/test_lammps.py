"""Tests for LAMMPSMDCalc class"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.calculators.calculator import Calculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData

from matcalc import DiffusionCalc, LAMMPSMDCalc
from matcalc.utils import PESCalculator

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

LAMMPS_TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "matcalc" / "lammps_templates"


@pytest.mark.parametrize(
    ("ensemble", "expected_energy"),
    [
        ("nve", -10.74606),
        ("nvt", -10.81289),
        ("npt_nose_hoover", -9.75230),
    ],
)
def test_lammps_calc(
    ensemble: str,
    expected_energy: float,
    Si: Structure,
) -> None:
    if not which("lmp"):
        pytest.skip("lammps not available")
    """Tests for LAMMPSMDCalc class"""
    in_file = f"{ensemble}.in"
    log_file = f"{ensemble}.log"
    traj_file = f"{ensemble}.lammpstrj"

    md_calc = LAMMPSMDCalc(
        calculator="TensorNet",
        ensemble=ensemble,
        temperature=300,
        taut=0.1,
        taup=0.1,
        steps=10,
        frames=5,
        infile=in_file,
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
    assert len(results["trajectory"]) == 5

    # Verify that the log file and trajectory file have been created
    assert os.path.isfile(log_file)
    assert os.path.isfile(traj_file)


def test_lammps_atoms(Si_atoms: Atoms) -> None:
    """Tests for MDCalc class using ASE Atoms input"""
    if not which("lmp"):
        pytest.skip("lammps not available")
    with open(Path(LAMMPS_TEMPLATES_DIR / "md.template")) as f:
        script_template = f.read()

    md_calc = LAMMPSMDCalc(
        calculator="TensorNet",
        temperature=300,
        taut=0.1,
        taup=0.1,
        steps=10,
        frames=5,
    )
    md_calc.write_inputs(Si_atoms, script_template=script_template)
    results = md_calc.calc(Si_atoms)
    assert isinstance(results, dict)


def test_invalid_ensemble(Si: Structure) -> None:
    """Ensure unsupported ensemble raises ValueError"""
    with pytest.raises(
        ValueError,
        match=(
            "The specified ensemble is not supported, choose from 'nve', 'nvt',"
            r" 'nvt_nose_hoover', 'npt', 'npt_nose_hoover'."
        ),
    ):
        LAMMPSMDCalc(
            calculator="TensorNet",
            ensemble="you_know_who",
            temperature=300,
            taut=0.1,
            taup=0.1,
            steps=10,
        ).calc(Si)


@pytest.mark.parametrize("as_atoms", [False, True])
@pytest.mark.parametrize("oxidation_states", [False, True])
def test_lammps_element_mapping(
    Li2O: Structure,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_atoms: bool,
    oxidation_states: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PESCalculator, "load_universal", lambda _name: Calculator())
    structure = Li2O.get_sorted_structure(key=lambda site: site.specie.symbol != "O")
    structure.remove_oxidation_states()
    if oxidation_states:
        structure.add_oxidation_state_by_element({"Li": 1, "O": -2})
    calc = LAMMPSMDCalc("pbe", relax_structure=False)
    calc.write_inputs(AseAtomsAdaptor.get_atoms(structure) if as_atoms else structure)

    data = LammpsData.from_file("md.data", atom_style="atomic")
    script = Path(calc.infile).read_text()
    pair_coeff = next(line.split() for line in script.splitlines() if line.startswith("pair_coeff"))
    dump_modify = next(line.split() for line in script.splitlines() if line.startswith("dump_modify"))
    elements = pair_coeff[-2:]
    assert elements == ["Li", "O"]
    assert dump_modify[-2:] == elements
    assert [elements[int(atom_type) - 1] for atom_type in data.atoms["type"]] == ["O", "Li", "Li"]


def _write_lammps_outputs(directory: Path) -> None:
    frames = []
    for step, oxygen_x, lithium_x1, lithium_x2 in (
        (0, 0.0, 1.0, 2.0),
        (10, 0.1, 1.2, 2.3),
        (20, 0.2, 1.4, 2.6),
    ):
        frames.append(
            f"ITEM: TIMESTEP\n{step}\n"
            "ITEM: NUMBER OF ATOMS\n3\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id element x y z\n"
            f"1 O {oxygen_x} 0 0\n"
            f"2 Li {lithium_x1} 1 1\n"
            f"3 Li {lithium_x2} 2 2\n"
        )
    (directory / "md.lammpstrj").write_text("".join(frames))
    (directory / "log.lammps").write_text(
        "LAMMPS (29 Aug 2024)\n"
        "Memory usage per processor = 1.0\n"
        "Step PotEng KinEng TotEng\n"
        "0 1 10 11\n10 2 20 22\n20 4 40 44\n25 100 1000 1100\n"
        "Loop time of 0.001 on 1 procs for 25 steps with 3 atoms\n"
    )


def test_lammps_sampled_results(Li2O: Structure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PESCalculator, "load_universal", lambda _name: Calculator())
    calc = LAMMPSMDCalc("pbe", relax_structure=False, steps=25, loginterval=10, frames=2)
    calc.write_inputs(Li2O)
    _write_lammps_outputs(tmp_path)
    monkeypatch.setattr(
        "matcalc._lammps.subprocess.run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    result = calc.calc(Li2O)

    assert len(result["trajectory"]) == 2
    assert result["potential_energy"] == pytest.approx(3.0)
    assert result["kinetic_energy"] == pytest.approx(30.0)
    assert result["total_energy"] == pytest.approx(33.0)
    np.testing.assert_allclose(result["trajectory"][0].positions[:, 0], [0.1, 1.2, 2.3])
    np.testing.assert_allclose(result["final_structure"].cart_coords, [[0.2, 0, 0], [1.4, 1, 1], [2.6, 2, 2]])
    assert result["final_structure"] != Li2O


def test_lammps_failed_run(Li2O: Structure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PESCalculator, "load_universal", lambda _name: Calculator())
    calc = LAMMPSMDCalc("pbe", relax_structure=False, frames=2)
    calc.write_inputs(Li2O)
    _write_lammps_outputs(tmp_path)
    executable = tmp_path / "lmp"
    executable.write_text("#!/bin/sh\nexit 7\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(subprocess.CalledProcessError) as error:
        calc.calc(Li2O)
    assert error.value.returncode == 7


@pytest.mark.parametrize(
    ("frames", "equilibration_frames", "count", "energy"), [(None, 0, 3, 7 / 3), (None, 1, 2, 3), (1, 1, 1, 4)]
)
def test_lammps_trajectory_window(
    Li2O: Structure,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: int | None,
    equilibration_frames: int,
    count: int,
    energy: float,
) -> None:
    monkeypatch.setattr(PESCalculator, "load_universal", lambda _name: Calculator())

    def run(command: list[str], *, cwd: Path, **_kwargs: object) -> subprocess.CompletedProcess:
        assert cwd == tmp_path / "run"
        assert (cwd / "in.md").is_file()
        _write_lammps_outputs(cwd)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("matcalc._lammps.subprocess.run", run)
    calc = LAMMPSMDCalc("pbe", relax_structure=False, frames=frames, equilibration_frames=equilibration_frames)
    result = calc.calc(Li2O, directory=tmp_path / "run")
    assert len(result["trajectory"]) == count
    assert result["potential_energy"] == pytest.approx(energy)
    assert result["kinetic_energy"] == pytest.approx(energy * 10)
    assert result["total_energy"] == pytest.approx(energy * 11)


def test_lammps_empty_trajectory_window(Li2O: Structure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PESCalculator, "load_universal", lambda _name: Calculator())
    calc = LAMMPSMDCalc("pbe", relax_structure=False, equilibration_frames=3)
    calc.write_inputs(Li2O, directory=tmp_path)
    _write_lammps_outputs(tmp_path)
    monkeypatch.setattr(
        "matcalc._lammps.subprocess.run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    with pytest.raises(ValueError, match="No frames left"):
        calc.calc(Li2O, directory=tmp_path)


def test_diffusion_lammps_run_directories(Li2O: Structure, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PESCalculator, "load_universal", lambda _name: Calculator())
    directories = []
    volumes = []

    def run(command: list[str], *, cwd: Path, **_kwargs: object) -> subprocess.CompletedProcess:
        directories.append(cwd)
        data = LammpsData.from_file(str(cwd / "md.data"), atom_style="atomic")
        volumes.append(data.box.to_lattice().volume)
        _write_lammps_outputs(cwd)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("matcalc._lammps.subprocess.run", run)
    calc = DiffusionCalc(
        "pbe", backend="lammps", specie="Li", equilibration_frames=0, analyzer_kwargs={"smoothed": False}
    )
    for scale in (1, 2):
        structure = Li2O.copy()
        structure.scale_lattice(Li2O.volume * scale)
        upstream = {"final_structure": structure, "label": "transport", "_units": {"volume": "Å^3"}}
        result = calc.calc(upstream)
        assert len(result["trajectory"]) == 3
        assert np.isfinite(result["diffusivity"])
        assert result["label"] == "transport"
        assert result["_units"]["volume"] == "Å^3"

    assert directories[0] != directories[1]
    assert not any(directory.exists() for directory in directories)
    np.testing.assert_allclose(volumes, [Li2O.volume, Li2O.volume * 2], rtol=1e-6)
