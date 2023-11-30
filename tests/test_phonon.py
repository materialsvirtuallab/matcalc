"""Tests for PhononCalc class"""
from __future__ import annotations

import pytest

from matcalc.phonon import PhononCalc


def test_phonon_calc(Li2O, M3GNetCalc):
    """Tests for PhononCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    pcalc = PhononCalc(M3GNetCalc, supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)), fmax=0.1, t_step=50, t_max=1000)
    results = pcalc.calc(Li2O)

    # Test values at 100 K
    ind = results["thermal_properties"]["temperatures"].tolist().index(300)
    assert results["thermal_properties"]["heat_capacity"][ind] == pytest.approx(58.42898370395005, rel=1e-2)
    assert results["thermal_properties"]["entropy"][ind] == pytest.approx(49.3774618162247, rel=1e-2)
    assert results["thermal_properties"]["free_energy"][ind] == pytest.approx(13.245478097108784, rel=1e-2)

    results = list(pcalc.calc_many([Li2O, Li2O]))
    assert len(results) == 2
