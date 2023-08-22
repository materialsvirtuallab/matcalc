from __future__ import annotations

import pytest

from matcalc.relaxation import RelaxCalc


def test_relax_calc(LiFePO4, M3GNetCalc, tmp_path):
    pcalc = RelaxCalc(M3GNetCalc, traj_file=f"{tmp_path}/lfp_relax.txt", optimizer="FIRE")
    results = pcalc.calc(LiFePO4)
    assert results["a"] == pytest.approx(4.75571137)
    assert results["b"] == pytest.approx(6.13161423)
    assert results["c"] == pytest.approx(10.4385933)
    assert results["alpha"] == pytest.approx(90, abs=1)
    assert results["beta"] == pytest.approx(90, abs=1)
    assert results["gamma"] == pytest.approx(90, abs=1)
    assert results["volume"] == pytest.approx(results["a"] * results["b"] * results["c"], abs=0.1)

    results = list(pcalc.calc_many([LiFePO4] * 2))
    assert len(results) == 2
    assert results[-1]["a"] == pytest.approx(4.7557113)

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        RelaxCalc(M3GNetCalc, optimizer="invalid")
