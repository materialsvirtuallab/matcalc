from __future__ import annotations

import pytest
from ase.calculators.calculator import Calculator

from matcalc.utils import UNIVERSAL_CALCULATORS, get_universal_calculator


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

    # cover edge case like https://github.com/materialsvirtuallab/matcalc/issues/14
    # where non-str and non-ASE Calculator instances are passed in
    assert get_universal_calculator(42) == 42  # test non-str input is returned as-is
