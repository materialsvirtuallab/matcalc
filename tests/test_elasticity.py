"""Tests for ElasticCalc class"""
from __future__ import annotations

import pytest

from matcalc.elasticity import ElasticityCalc


def test_elastic_calc(LiFePO4, M3GNetCalc):
    """Tests for ElasticCalc class"""
    calculator = M3GNetCalc
    ecalc = ElasticityCalc(calculator, norm_strains=0.02, shear_strains=0.04, fmax=0.1)

    # Test LiFePO4 with relaxation
    results = ecalc.calc(LiFePO4)
    assert results["elastic_tensor"].shape == (3, 3, 3, 3)
    assert results["elastic_tensor"][0][1][1][0] == pytest.approx(0.5507453854534105)
    assert results["bulk_modulus_vrh"] == pytest.approx(1.2984489572716424, rel=1e-2)
    assert results["shear_modulus_vrh"] == pytest.approx(0.5637053266436473, rel=1e-2)
    assert results["youngs_modulus"] == pytest.approx(1477328151.6142423, rel=1e-2)
