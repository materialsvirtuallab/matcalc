"""Tests for PhononCalc class"""
from __future__ import annotations

import pytest

from matcalc.eos import EOSCalc


def test_eos_calc(Li2O, LiFePO4, M3GNetCalc):
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    pcalc = EOSCalc(M3GNetCalc, fmax=0.1)
    results = pcalc.calc(Li2O)

    assert {*results} == {"eos", "r2_score_bm", "bulk_modulus_bm"}
    assert results["bulk_modulus_bm"] == pytest.approx(65.57980045603279, rel=1e-2)
    assert {*results["eos"]} == {"volumes", "energies"}
    assert results["eos"]["volumes"] == pytest.approx(
        [18.38, 19.63, 20.94, 22.3, 23.73, 25.21, 26.75, 28.36, 30.02, 31.76, 33.55],
        rel=1e-3,
    )
    assert results["eos"]["energies"] == pytest.approx(
        [-13.52, -13.77, -13.94, -14.08, -14.15, -14.18, -14.16, -14.11, -14.03, -13.94, -13.83],
        rel=1e-3,
    )
    pcalc = EOSCalc(M3GNetCalc, relax_structure=False)
    results = list(pcalc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus_bm"] == pytest.approx(54.5953851822073, rel=1e-2)
