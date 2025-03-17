"""Tests for ElasticCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.filters import ExpCellFilter
from matcalc.elasticity import ElasticityCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize("relax_deformed_structures", [False, True])
def test_elastic_calc(
    Li2O: Structure,
    pes_calculator: PESCalculator,
    relax_deformed_structures: bool,  # noqa: FBT001
) -> None:
    """Tests for ElasticCalc class"""
    elast_calc = ElasticityCalc(
        pes_calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=True,
        relax_deformed_structures=relax_deformed_structures,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )
    # Test Li2O with equilibrium structure
    results = elast_calc.calc(Li2O)
    assert results["elastic_tensor"].shape == (3, 3, 3, 3)
    assert results["structure"].lattice.a == pytest.approx(3.2885851104196875, rel=1e-4)

    assert results["elastic_tensor"][0][1][1][0] == pytest.approx(0.5014895636122672, rel=1e-3)
    assert results["bulk_modulus_vrh"] == pytest.approx(0.6737897607182401, rel=1e-3)
    assert results["shear_modulus_vrh"] == pytest.approx(0.4179219576918434, rel=1e-3)
    assert results["youngs_modulus"] == pytest.approx(1038959096.5809333, rel=1e-3)
    assert results["residuals_sum"] == pytest.approx(3.8487476828544434e-08, rel=1e-2)

    # Test Li2O without the equilibrium structure
    elast_calc = ElasticityCalc(
        pes_calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=False,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    results = elast_calc.calc(Li2O)
    assert results["residuals_sum"] == pytest.approx(2.9257237571340992e-08, rel=1e-2)

    # Test Li2O with float
    elast_calc = ElasticityCalc(
        pes_calculator,
        fmax=0.1,
        norm_strains=0.004,
        shear_strains=0.004,
        use_equilibrium=True,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    results = elast_calc.calc(Li2O)
    assert results["residuals_sum"] == 0.0
    assert results["bulk_modulus_vrh"] == pytest.approx(0.6631894154825593, rel=1e-3)


def test_elastic_calc_invalid_states(pes_calculator: PESCalculator) -> None:
    with pytest.raises(ValueError, match="shear_strains is empty"):
        ElasticityCalc(pes_calculator, shear_strains=[])
    with pytest.raises(ValueError, match="norm_strains is empty"):
        ElasticityCalc(pes_calculator, norm_strains=[])

    with pytest.raises(ValueError, match="strains must be non-zero"):
        ElasticityCalc(pes_calculator, norm_strains=[0.0, 0.1])
    with pytest.raises(ValueError, match="strains must be non-zero"):
        ElasticityCalc(pes_calculator, shear_strains=[0.0, 0.1])
