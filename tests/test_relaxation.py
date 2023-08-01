from __future__ import annotations

import os

import matgl
import pytest


from matcalc.relaxation import RelaxCalc


def test_RelaxCalc(LiFePO4, M3GNetUPCalc):
    calculator = M3GNetUPCalc

    pcalc = RelaxCalc(calculator, traj_file="lfp_relax.txt")
    results = pcalc.calc(LiFePO4)
    assert results["a"] == pytest.approx(4.755711375217371)
    assert results["b"] == pytest.approx(6.131614236614623)
    assert results["c"] == pytest.approx(10.43859339794175)
    assert results["alpha"] == pytest.approx(90, abs=1)
    assert results["beta"] == pytest.approx(90, abs=1)
    assert results["gamma"] == pytest.approx(90, abs=1)
    assert results["volume"] == pytest.approx(results["a"] * results["b"] * results["c"], abs=0.1)

    results = list(pcalc.calc_many([LiFePO4] * 2))
    assert len(results) == 2
    assert results[-1]["a"] == pytest.approx(4.755711375217371)
    os.remove("lfp_relax.txt")
