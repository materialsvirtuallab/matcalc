"""Tests for ElasticCalc class"""
from __future__ import annotations

import pytest

from matcalc.elasticity import ElasticityCalc
import numpy as np


def test_elastic_calc(Li2O, M3GNetCalc):
    """Tests for ElasticCalc class"""
    calculator = M3GNetCalc
    ecalc = ElasticityCalc(
        calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=True,
    )

    # Test Li2O with equilibrium structure
    results = ecalc.calc(Li2O)
    assert results["elastic_tensor"].shape == (3, 3, 3, 3)
    assert results["elastic_tensor"][0][1][1][0] == pytest.approx(0.48107847906067014, rel=1e-3)
    assert results["bulk_modulus_vrh"] == pytest.approx(0.6463790972016057, rel=1e-3)
    assert results["shear_modulus_vrh"] == pytest.approx(0.4089633387159433, rel=1e-3)
    assert results["youngs_modulus"] == pytest.approx(1013205376.4173204, rel=1e-3)
    assert results["residuals_sum"] == pytest.approx(3.511205625736905e-08, rel=1e-2)
    assert results["structure"].lattice.a == pytest.approx(3.3089642445687586, rel=1e-4)

    # Test Li2O without the equilibrium structure
    ecalc = ElasticityCalc(
        calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=False,
    )

    results = ecalc.calc(Li2O)
    assert results["residuals_sum"] == pytest.approx(2.673485844932801e-08, rel=1e-2)
