import pytest

from matcalc.util import get_universal_calculator
from ase.calculators.calculator import Calculator


def test_get_calculator():
    for name in ("M3GNet", "M3GNet-MP-2021.2.8-PES", "CHGNet"):
        calc = get_universal_calculator(name)
        assert isinstance(calc, Calculator)
    with pytest.raises(ValueError, match="Unsupported model name"):
        get_universal_calculator("whatever")
