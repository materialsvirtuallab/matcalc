from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.filters import ExpCellFilter, FrechetCellFilter

from matcalc import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase.filters import Filter
    from matgl.ext.ase import PESCalculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


def test_bad_input(Li2O: Structure, m3gnet_calculator: PESCalculator) -> None:
    relax_calc = RelaxCalc(
        m3gnet_calculator,
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
    )
    with pytest.raises(ValueError, match="Structure must be either a pymatgen Structure"):
        relax_calc.calc({"bad": Li2O})

    data = list(relax_calc.calc_many([Li2O, None], allow_errors=True))
    assert data[0] is not None
    assert data[1] is None


@pytest.mark.parametrize(
    ("perturb_distance", "expected_a", "expected_energy"),
    [(0, 3.291072, -14.176680), (0.2, 3.291072, -14.176716)],
)
def test_relax_calc_relax_cell(
    Li2O: Structure,
    m3gnet_calculator: PESCalculator,
    tmp_path: Path,
    perturb_distance: float | None,
    expected_a: float,
    expected_energy: float,
) -> None:
    relax_calc = RelaxCalc(
        m3gnet_calculator,
        traj_file=f"{tmp_path}/li2o_relax.txt",
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
        perturb_distance=perturb_distance,
    )
    temp_structure = Li2O.copy()
    result = relax_calc.calc(temp_structure)
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

    assert energy == pytest.approx(expected_energy, rel=1e-1)
    assert a == pytest.approx(expected_a, rel=1e-1)
    assert b == pytest.approx(expected_a, rel=1e-1)
    assert c == pytest.approx(expected_a, rel=1e-1)
    assert alpha == pytest.approx(60, abs=5)
    assert beta == pytest.approx(60, abs=5)
    assert gamma == pytest.approx(60, abs=5)
    assert final_struct.volume == pytest.approx(a * b * c / 2**0.5, abs=2)


@pytest.mark.parametrize(("expected_a", "expected_energy"), [(3.291072, -14.176713)])
def test_relax_calc_relax_atoms(
    Li2O: Structure,
    m3gnet_calculator: PESCalculator,
    tmp_path: Path,
    expected_a: float,
    expected_energy: float,
) -> None:
    relax_calc = RelaxCalc(
        m3gnet_calculator,
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

    assert energy == pytest.approx(expected_energy, rel=1e-1)
    assert a == pytest.approx(expected_a, rel=1e-3)
    assert b == pytest.approx(expected_a, rel=1e-3)
    assert c == pytest.approx(expected_a, rel=1e-3)
    assert alpha == pytest.approx(60, abs=5)
    assert beta == pytest.approx(60, abs=5)
    assert gamma == pytest.approx(60, abs=5)
    assert final_struct.volume == pytest.approx(a * b * c / 2**0.5, abs=2)


@pytest.mark.parametrize(
    ("expected_energy", "expected_forces", "expected_stresses"),
    [
        (
            -14.176713,
            np.array(
                [
                    [6.5772183e-06, 1.8514693e-06, -7.0808455e-06],
                    [-4.5074150e-03, -3.3108518e-03, -7.0908130e-03],
                    [4.5009712e-03, 3.3089996e-03, 7.0979437e-03],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.00242333, -0.00024411, -0.00052359],
                    [-0.00024411, 0.00257503, -0.00038525],
                    [-0.00052359, -0.00038525, 0.00192819],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_static_calc(
    Li2O: Structure,
    m3gnet_calculator: PESCalculator,
    expected_energy: float,
    expected_forces: ArrayLike,
    expected_stresses: ArrayLike,
) -> None:
    relax_calc = RelaxCalc(m3gnet_calculator, relax_atoms=False, relax_cell=False)
    result = relax_calc.calc({"structure": Li2O})
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
        "structure",
    ):
        assert key in result, f"{key=} not in result"
    energy: float = result["energy"]
    forces: ArrayLike = result["forces"]
    stresses: ArrayLike = result["stress"]

    assert energy == pytest.approx(expected_energy, rel=1e-1)
    assert np.allclose(forces, expected_forces, rtol=1e-1)
    assert np.allclose(stresses, expected_stresses, rtol=1e-1)


@pytest.mark.parametrize(
    ("cell_filter", "expected_a"),
    [(ExpCellFilter, 3.288585), (FrechetCellFilter, 3.291072)],
)
def test_relax_calc_many(
    Li2O: Structure,
    m3gnet_calculator: PESCalculator,
    cell_filter: Filter,
    expected_a: float,
) -> None:
    relax_calc = RelaxCalc(m3gnet_calculator, optimizer="FIRE", cell_filter=cell_filter)
    results = list(relax_calc.calc_many([Li2O] * 2))
    assert len(results) == 2
    assert results[-1]["a"] == pytest.approx(expected_a, rel=1e-1)


def test_relax_calc_invalid_optimizer(m3gnet_calculator: PESCalculator, Li2O: Structure) -> None:
    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        RelaxCalc(m3gnet_calculator, optimizer="invalid").calc(Li2O)
