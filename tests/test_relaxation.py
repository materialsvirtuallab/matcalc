from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.filters import ExpCellFilter, FrechetCellFilter
from matcalc.relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase.filters import Filter
    from matgl.ext.ase import PESCalculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    ("cell_filter", "expected_a", "expected_energy"),
    [(ExpCellFilter, 3.288585, -14.176867), (FrechetCellFilter, 3.291072, -14.176713)],
)
def test_relax_calc_relax_cell(
    Li2O: Structure,
    pes_calculator: PESCalculator,
    tmp_path: Path,
    cell_filter: Filter,
    expected_a: float,
    expected_energy: float,
) -> None:
    relax_calc = RelaxCalc(
        pes_calculator,
        traj_file=f"{tmp_path}/li2o_relax.txt",
        optimizer="FIRE",
        cell_filter=cell_filter,
        relax_atoms=True,
        relax_cell=True,
    )
    result = relax_calc.calc(Li2O)
    for key in (
        "final_structure",
        "energy",
        "forces",
        "stress",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ):
        assert key in result, f"{key=} not in result"
    final_struct: Structure = result["final_structure"]
    energy: float = result["energy"]
    missing_keys = {*final_struct.lattice.params_dict} - {*result}
    assert len(missing_keys) == 0, f"{missing_keys=}"
    a, b, c, alpha, beta, gamma = final_struct.lattice.parameters

    assert energy == pytest.approx(expected_energy, rel=1e-3)
    assert a == pytest.approx(expected_a, rel=1e-3)
    assert b == pytest.approx(expected_a, rel=1e-3)
    assert c == pytest.approx(expected_a, rel=1e-3)
    assert alpha == pytest.approx(60, abs=0.5)
    assert beta == pytest.approx(60, abs=0.5)
    assert gamma == pytest.approx(60, abs=0.5)
    assert final_struct.volume == pytest.approx(a * b * c / 2**0.5, abs=0.1)


@pytest.mark.parametrize(("expected_a", "expected_energy"), [(3.291072, -14.176713)])
def test_relax_calc_relax_atoms(
    Li2O: Structure,
    pes_calculator: PESCalculator,
    tmp_path: Path,
    expected_a: float,
    expected_energy: float,
) -> None:
    relax_calc = RelaxCalc(
        pes_calculator,
        traj_file=f"{tmp_path}/li2o_relax.txt",
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=False,
    )
    result = relax_calc.calc(Li2O)

    for key in (
        "final_structure",
        "energy",
        "forces",
        "stress",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ):
        assert key in result, f"{key=} not in result"

    final_struct: Structure = result["final_structure"]
    energy: float = result["energy"]
    missing_keys = {*final_struct.lattice.params_dict} - {*result}
    assert len(missing_keys) == 0, f"{missing_keys=}"
    a, b, c, alpha, beta, gamma = final_struct.lattice.parameters

    assert energy == pytest.approx(expected_energy, rel=1e-3)
    assert a == pytest.approx(expected_a, rel=1e-3)
    assert b == pytest.approx(expected_a, rel=1e-3)
    assert c == pytest.approx(expected_a, rel=1e-3)
    assert alpha == pytest.approx(60, abs=0.5)
    assert beta == pytest.approx(60, abs=0.5)
    assert gamma == pytest.approx(60, abs=0.5)
    assert final_struct.volume == pytest.approx(a * b * c / 2**0.5, abs=0.1)


@pytest.mark.parametrize(
    ("expected_energy", "expected_forces", "expected_stresses"),
    [
        (
            -14.176713,
            np.array(
                [
                    [6.577218e-06, 1.851469e-06, -7.080846e-06],
                    [-4.507415e-03, -3.310852e-03, -7.090813e-03],
                    [4.500971e-03, 3.309000e-03, 7.097944e-03],
                ],
                dtype=np.float32,
            ),
            np.array(
                [0.003883, 0.004126, 0.003089, -0.000617, -0.000839, -0.000391],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_static_calc(
    Li2O: Structure,
    pes_calculator: PESCalculator,
    expected_energy: float,
    expected_forces: ArrayLike,
    expected_stresses: ArrayLike,
) -> None:
    relax_calc = RelaxCalc(pes_calculator, relax_atoms=False, relax_cell=False)
    result = relax_calc.calc(Li2O)
    for key in (
        "final_structure",
        "energy",
        "forces",
        "stress",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ):
        assert key in result, f"{key=} not in result"
    energy: float = result["energy"]
    forces: ArrayLike = result["forces"]
    stresses: ArrayLike = result["stress"]

    assert energy == pytest.approx(expected_energy, rel=1e-3)
    assert np.allclose(forces, expected_forces, rtol=1e-3)
    assert np.allclose(stresses, expected_stresses, rtol=1e-3)


@pytest.mark.parametrize(
    ("cell_filter", "expected_a"),
    [(ExpCellFilter, 3.288585), (FrechetCellFilter, 3.291072)],
)
def test_relax_calc_many(
    Li2O: Structure,
    pes_calculator: PESCalculator,
    cell_filter: Filter,
    expected_a: float,
) -> None:
    relax_calc = RelaxCalc(pes_calculator, optimizer="FIRE", cell_filter=cell_filter)
    results = list(relax_calc.calc_many([Li2O] * 2))
    assert len(results) == 2
    assert results[-1]["a"] == pytest.approx(expected_a, rel=1e-3)


def test_relax_calc_invalid_optimizer(pes_calculator: PESCalculator) -> None:
    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        RelaxCalc(pes_calculator, optimizer="invalid")
