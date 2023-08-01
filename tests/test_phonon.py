from __future__ import annotations

from matcalc.phonon import PhononCalc


def test_PhononCalc(M3GNetUPCalc):
    calc = PhononCalc(M3GNetUPCalc)
    assert True
