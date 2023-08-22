"""Tests for PhononCalc class"""
from __future__ import annotations

import pytest

from matcalc.eos import EOSCalc


def test_eos_calc(Li2O, LiFePO4, M3GNetCalc):
    """Tests for EOSCalc class"""
    calculator = M3GNetCalc
    # Note that the fmax is probably too high. This is for testing purposes only.
    pcalc = EOSCalc(calculator, fmax=0.1)
    results = pcalc.calc(Li2O)

    assert {*results} == {"eos", "bulk_modulus_bm"}
    assert results["bulk_modulus_bm"] == pytest.approx(69.868, rel=1e-2)
    assert {*results["eos"]} == {"volumes", "energies"}
    assert results["eos"]["volumes"] == pytest.approx(
        [18.70, 19.97, 21.30, 22.69, 24.14, 25.65, 27.22, 28.85, 30.55, 32.31, 34.14],
        rel=1e-3,
    )
    assert results["eos"]["energies"] == pytest.approx(
        [-13.51, -13.78, -13.98, -14.11, -14.17, -14.19, -14.17, -14.12, -14.04, -13.94, -13.81],
        rel=1e-3,
    )

    results = list(pcalc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus_bm"] == pytest.approx(60.083, rel=1e-2)
