"""Tests for PhononCalc class"""
from __future__ import annotations

import pytest

from matcalc.phonon import PhononCalc


def test_phonon_calc(Li2O, LiFePO4, M3GNetCalc):
    """Tests for PhononCalc class"""
    calculator = M3GNetCalc
    # Note that the fmax is probably too high. This is for testing purposes only.
    pcalc = PhononCalc(calculator, supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)), fmax=0.1, t_step=50, t_max=1000)
    results = pcalc.calc(Li2O)

    # Test values at 100 K
    ind = results["temperatures"].tolist().index(300)
    assert results["heat_capacity"][ind] == pytest.approx(59.91894069664282, rel=1e-2)
    assert results["entropy"][ind] == pytest.approx(51.9081928335805, rel=1e-2)
    assert results["free_energy"][ind] == pytest.approx(11.892105644441045, rel=1e-2)

    results = list(pcalc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[-1]["heat_capacity"][ind] == pytest.approx(550.6419940551511, rel=1e-2)
