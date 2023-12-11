from __future__ import annotations

import pytest
from ase.calculators.calculator import Calculator

from matcalc.util import UNIVERSAL_CALCULATORS, get_universal_calculator


def test_get_universal_calculator() -> None:
    for name in UNIVERSAL_CALCULATORS:
        calc = get_universal_calculator(name)
        assert isinstance(calc, Calculator)
        same_calc = get_universal_calculator(calc)  # test ASE Calculator classes are returned as-is
        assert calc is same_calc

    name = "whatever"
    with pytest.raises(ValueError, match=f"Unrecognized {name=}") as exc:
        get_universal_calculator(name)
    assert str(exc.value) == f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}"
