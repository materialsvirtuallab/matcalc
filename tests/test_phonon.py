"""Tests for PhononCalc class"""
from __future__ import annotations

import pytest

from matcalc.phonon import PhononCalc


def test_PhononCalc(Li2O, M3GNetUPCalc):
    """Tests for PhononCalc class"""
    calculator = M3GNetUPCalc
    pcalc = PhononCalc(calculator, supercell_matrix=((3, 0, 0), (0, 3, 0), (0, 0, 3)), fmax=0.001)
    results = pcalc.calc(Li2O)
    assert results["thermal_properties"]["C_v"][100] == pytest.approx(46.686590135216676, abs=0.01)
    assert results["thermal_properties"]["entropy"][100] == pytest.approx(29.347199723052057, abs=0.01)
    assert results["thermal_properties"]["free_energy"][100] == pytest.approx(16.273094775993236, abs=0.01)
    results = list(pcalc.calc_many([Li2O] * 2))
    assert len(results) == 2
    assert results[-1]["thermal_properties"]["C_v"][100] == pytest.approx(46.68654747038025, abs=0.01)
