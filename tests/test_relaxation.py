from __future__ import annotations

import pytest

from matcalc.relaxation import RelaxCalc


def test_relax_calc(Li2O, M3GNetCalc, tmp_path):
    pcalc = RelaxCalc(M3GNetCalc, traj_file=f"{tmp_path}/li2o_relax.txt", optimizer="FIRE")
    results = pcalc.calc(Li2O)
    assert results["a"] == pytest.approx(3.291071792359756, rel=0.002)
    assert results["b"] == pytest.approx(3.291071899625086, rel=0.002)
    assert results["c"] == pytest.approx(3.291072056855788, rel=0.002)
    assert results["alpha"] == pytest.approx(60, abs=1)
    assert results["beta"] == pytest.approx(60, abs=1)
    assert results["gamma"] == pytest.approx(60, abs=1)
    assert results["volume"] == pytest.approx(results["a"] * results["b"] * results["c"]/2**0.5, abs=0.1)

    results = list(pcalc.calc_many([Li2O] * 2))
    assert len(results) == 2
    assert results[-1]["a"] == pytest.approx(3.291071792359756, rel=0.002)

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        RelaxCalc(M3GNetCalc, optimizer="invalid")
