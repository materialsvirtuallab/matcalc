"""Tests for PhononCalc class"""
from __future__ import annotations

import pytest

from matcalc.eos import EOSCalc


def test_PhononCalc(Li2O, LiFePO4, M3GNetUPCalc):
    """Tests for PhononCalc class"""
    calculator = M3GNetUPCalc
    # Note that the fmax is probably too high. This is for testing purposes only.
    pcalc = EOSCalc(calculator, fmax=0.01)
    results = pcalc.calc(Li2O)

    assert results["bulk_modulus"] == pytest.approx(69.86879801931632, 1e-2)

    results = list(pcalc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus"] == pytest.approx(53.791519828150925, 1e-2)
