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
    assert results["bulk_modulus_bm"] == pytest.approx(73.094, rel=1e-2)
    assert {*results["eos"]} == {"volumes", "energies"}
    print(results["eos"]["volumes"], results["eos"]["energies"])
    assert results["eos"]["volumes"] == pytest.approx(
        [18.92, 20.21, 21.56, 22.96, 24.43, 25.95, 27.54, 29.19, 30.91, 32.69, 34.54],
        rel=1e-3,
    )
    assert results["eos"]["energies"] == pytest.approx(
        [-13.81, -14.08, -14.27, -14.39, -14.46, -14.48, -14.45, -14.40, -14.31, -14.19, -14.05],
        rel=1e-3,
    )

    results = list(pcalc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus_bm"] == pytest.approx(149.327, rel=1e-2)
