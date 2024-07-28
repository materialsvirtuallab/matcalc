from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ase.filters import ExpCellFilter, FrechetCellFilter

from matcalc.relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase.filters import Filter
    from matgl.ext.ase import M3GNetCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize(("cell_filter", "expected_a"), [(ExpCellFilter, 3.291071), (FrechetCellFilter, 3.288585)])
def test_relax_calc_single(
    Li2O: Structure, M3GNetCalc: M3GNetCalculator, tmp_path: Path, cell_filter: Filter, expected_a: float
) -> None:
    relax_calc = RelaxCalc(
        M3GNetCalc, traj_file=f"{tmp_path}/li2o_relax.txt", optimizer="FIRE", cell_filter=cell_filter
    )
    result = relax_calc.calc(Li2O)
    final_struct: Structure = result["final_structure"]
    missing_keys = {*final_struct.lattice.params_dict} - {*result}
    assert len(missing_keys) == 0, f"{missing_keys=}"
    a, b, c, alpha, beta, gamma = final_struct.lattice.parameters

    assert a == pytest.approx(expected_a, rel=0.002)
    assert b == pytest.approx(expected_a, rel=0.002)
    assert c == pytest.approx(expected_a, rel=0.002)
    assert alpha == pytest.approx(60, abs=1)
    assert beta == pytest.approx(60, abs=1)
    assert gamma == pytest.approx(60, abs=1)
    assert final_struct.volume == pytest.approx(a * b * c / 2**0.5, abs=0.1)


@pytest.mark.parametrize(("cell_filter", "expected_a"), [(ExpCellFilter, 3.291071), (FrechetCellFilter, 3.288585)])
def test_relax_calc_many(Li2O: Structure, M3GNetCalc: M3GNetCalculator, cell_filter: Filter, expected_a: float) -> None:
    relax_calc = RelaxCalc(M3GNetCalc, optimizer="FIRE", cell_filter=cell_filter)
    results = list(relax_calc.calc_many([Li2O] * 2))
    assert len(results) == 2
    assert results[-1]["a"] == pytest.approx(expected_a, rel=0.002)


def test_relax_calc_invalid_optimizer(M3GNetCalc: M3GNetCalculator) -> None:
    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        RelaxCalc(M3GNetCalc, optimizer="invalid")
