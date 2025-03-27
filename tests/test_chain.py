"""Tests for ElasticCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.filters import ExpCellFilter

from matcalc.base import ChainedCalc
from matcalc.elasticity import ElasticityCalc
from matcalc.relaxation import RelaxCalc
from matcalc.stability import EnergeticsCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_chain_calc(
    Li2O: Structure,
    m3gnet_calculator: PESCalculator,
) -> None:
    """Tests for ElasticCalc class"""

    relax_calc = RelaxCalc(
        m3gnet_calculator,
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
    )
    energetics_calc = EnergeticsCalc(m3gnet_calculator, relax_structure=False)
    elast_calc = ElasticityCalc(
        m3gnet_calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=True,
        relax_structure=False,
        relax_deformed_structures=True,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )
    calc = ChainedCalc([relax_calc, energetics_calc, elast_calc])
    # Test Li2O with equilibrium structure
    results = calc.calc(Li2O)
    assert results["elastic_tensor"].shape == (3, 3, 3, 3)
    assert results["structure"].lattice.a == pytest.approx(3.291071792359756, rel=1e-1)

    assert results["elastic_tensor"][0][1][1][0] == pytest.approx(0.3121514513622968, rel=1e-1)
    assert results["bulk_modulus_vrh"] == pytest.approx(0.41534028838780773, rel=1e-1)
    assert results["shear_modulus_vrh"] == pytest.approx(0.25912319676768314, rel=1e-1)
    assert results["youngs_modulus"] == pytest.approx(643538946.776407, rel=1e-1)
    assert results["residuals_sum"] == pytest.approx(1.4675954664743306e-08, rel=1e-1)
    # A chained calculation has results from all steps.

    assert results["energy"] == pytest.approx(-14.176680, rel=1e-1)
    assert results["a"] == pytest.approx(3.291072, rel=1e-1)
    assert results["alpha"] == pytest.approx(60, abs=5)

    assert results["formation_energy_per_atom"] == pytest.approx(-1.8127431869506836, abs=1e-3)

    results = list(calc.calc_many([Li2O] * 3))
    assert len(results) == 3
    assert results[0]["energy"] == pytest.approx(-14.176680, rel=1e-1)
