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
    m3gnet_calculator: PESCalculator,
    relax_deformed_structures: bool,  # noqa: FBT001
) -> None:
    """Tests for ElasticCalc class"""
    elast_calc = ElasticityCalc(
        m3gnet_calculator,
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
    assert results["structure"].lattice.a == pytest.approx(3.291071792359756, rel=1e-1)

    assert results["elastic_tensor"][0][1][1][0] == pytest.approx(0.3121514513622968, rel=1e-1)
    assert results["bulk_modulus_vrh"] == pytest.approx(0.41534028838780773, rel=1e-1)
    assert results["shear_modulus_vrh"] == pytest.approx(0.25912319676768314, rel=1e-1)
    assert results["youngs_modulus"] == pytest.approx(643538946.776407, rel=1e-1)
    assert results["residuals_sum"] == pytest.approx(1.4675954664743306e-08, rel=1e-1)

    # Test Li2O without the equilibrium structure
    elast_calc = ElasticityCalc(
        m3gnet_calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=False,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    results = elast_calc.calc(Li2O)
    assert results["residuals_sum"] == pytest.approx(1.1166845725443057e-08, rel=1e-1)

    # Test Li2O with float
    elast_calc = ElasticityCalc(
        m3gnet_calculator,
        fmax=0.1,
        norm_strains=0.004,
        shear_strains=0.004,
        use_equilibrium=True,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    results = elast_calc.calc(Li2O)
    assert results["residuals_sum"] == 0.0
    assert results["bulk_modulus_vrh"] == pytest.approx(0.40877813076228825, rel=1e-1)


def test_elastic_calc_invalid_states(m3gnet_calculator: PESCalculator) -> None:
    with pytest.raises(ValueError, match="shear_strains is empty"):
        ElasticityCalc(m3gnet_calculator, shear_strains=[])
    with pytest.raises(ValueError, match="norm_strains is empty"):
        ElasticityCalc(m3gnet_calculator, norm_strains=[])

    with pytest.raises(ValueError, match="strains must be non-zero"):
        ElasticityCalc(m3gnet_calculator, norm_strains=[0.0, 0.1])
    with pytest.raises(ValueError, match="strains must be non-zero"):
        ElasticityCalc(m3gnet_calculator, shear_strains=[0.0, 0.1])
