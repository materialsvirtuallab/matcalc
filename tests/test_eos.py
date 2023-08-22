"""Tests for PhononCalc class"""
from __future__ import annotations

import pytest

from matcalc.eos import EOSCalc


def test_PhononCalc(Li2O, LiFePO4, M3GNetCalc):
    """Tests for PhononCalc class"""
    calculator = M3GNetCalc
    # Note that the fmax is probably too high. This is for testing purposes only.
    pcalc = EOSCalc(calculator, fmax=0.1)
    results = pcalc.calc(Li2O)

    assert results["bulk_modulus_bm"] == pytest.approx(69.86879801931632, rel=0.1)

    results = list(pcalc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus_bm"] == pytest.approx(60.083102790525366, rel=0.1)
