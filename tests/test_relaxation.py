from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ase.filters import ExpCellFilter

from matcalc.relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from matgl.ext.ase import M3GNetCalculator
    from pymatgen.core import Structure


def test_relax_calc(Li2O: Structure, M3GNetCalc: M3GNetCalculator, tmp_path: Path) -> None:
    relax_calc = RelaxCalc(
        M3GNetCalc, traj_file=f"{tmp_path}/li2o_relax.txt", optimizer="FIRE", cell_filter=ExpCellFilter
    )
    result = relax_calc.calc(Li2O)
    assert result["a"] == pytest.approx(3.291071792359756, rel=0.002)
    assert result["b"] == pytest.approx(3.291071899625086, rel=0.002)
    assert result["c"] == pytest.approx(3.291072056855788, rel=0.002)
    assert result["alpha"] == pytest.approx(60, abs=1)
    assert result["beta"] == pytest.approx(60, abs=1)
    assert result["gamma"] == pytest.approx(60, abs=1)
    assert result["volume"] == pytest.approx(result["a"] * result["b"] * result["c"] / 2**0.5, abs=0.1)

    results = list(relax_calc.calc_many([Li2O] * 2))
    assert len(results) == 2
    assert results[-1]["a"] == pytest.approx(3.291071792359756, rel=0.002)

    relax_calc_frechet = RelaxCalc(M3GNetCalc, optimizer="FIRE")
    result_frechet = relax_calc_frechet.calc(Li2O)
    assert result_frechet["a"] == pytest.approx(3.288585037486192, rel=0.002)
    assert result_frechet["b"] == pytest.approx(3.288586588568963, rel=0.002)
    assert result_frechet["c"] == pytest.approx(3.2885877985629786, rel=0.002)
    assert result_frechet["alpha"] == pytest.approx(60, abs=1)
    assert result_frechet["beta"] == pytest.approx(60, abs=1)
    assert result_frechet["gamma"] == pytest.approx(60, abs=1)
    assert result_frechet["volume"] == pytest.approx(
        result_frechet["a"] * result_frechet["b"] * result_frechet["c"] / 2**0.5, abs=0.1
    )

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        RelaxCalc(M3GNetCalc, optimizer="invalid")
