"""Tests for PhononCalc class"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc.phonon import PhononCalc

if TYPE_CHECKING:
    from matgl.ext.ase import M3GNetCalculator
    from pymatgen.core import Structure


def test_phonon_calc(Li2O: Structure, M3GNetCalc: M3GNetCalculator) -> None:
    """Tests for PhononCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    phonon_calc = PhononCalc(
        M3GNetCalc, supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)), fmax=0.1, t_step=50, t_max=1000
    )
    result = phonon_calc.calc(Li2O)

    # Test values at 100 K
    ind = result["thermal_properties"]["temperatures"].tolist().index(300)
    assert result["thermal_properties"]["heat_capacity"][ind] == pytest.approx(58.42898370395005, rel=1e-2)
    assert result["thermal_properties"]["entropy"][ind] == pytest.approx(49.3774618162247, rel=1e-2)
    assert result["thermal_properties"]["free_energy"][ind] == pytest.approx(13.245478097108784, rel=1e-2)

    results = list(phonon_calc.calc_many([Li2O, Li2O]))
    assert len(results) == 2
