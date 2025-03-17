"""Tests for PhononCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ase.filters import ExpCellFilter
from matcalc.eos import EOSCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_eos_calc(
    Li2O: Structure,
    LiFePO4: Structure,
    pes_calculator: PESCalculator,
) -> None:
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    eos_calc = EOSCalc(pes_calculator, fmax=0.1, relax_calc_kwargs={"cell_filter": ExpCellFilter})
    result = eos_calc.calc(Li2O)

    assert {*result} == {"eos", "r2_score_bm", "bulk_modulus_bm"}
    assert result["bulk_modulus_bm"] == pytest.approx(65.57980045603279, rel=1e-2)
    assert {*result["eos"]} == {"volumes", "energies"}
    assert result["eos"]["volumes"] == pytest.approx(
        [18.38, 19.63, 20.94, 22.3, 23.73, 25.21, 26.75, 28.36, 30.02, 31.76, 33.55],
        rel=1e-3,
    )
    assert result["eos"]["energies"] == pytest.approx(
        [
            -13.52,
            -13.77,
            -13.94,
            -14.08,
            -14.15,
            -14.18,
            -14.16,
            -14.11,
            -14.03,
            -13.94,
            -13.83,
        ],
        rel=1e-3,
    )
    eos_calc = EOSCalc(pes_calculator, relax_structure=False)
    results = list(eos_calc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus_bm"] == pytest.approx(54.5953851822073, rel=1e-2)
