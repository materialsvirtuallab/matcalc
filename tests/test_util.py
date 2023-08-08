import pytest

from matcalc.util import get_calculator
from ase.calculators.calculator import Calculator


def test_get_calculator():
    for name in ("M3GNet-MP-2021.2.8-DIRECT-PES", "CHGNet"):
        calc = get_calculator(name)
        assert isinstance(calc, Calculator)
    with pytest.raises(ValueError, match="Unsupported model name"):
        get_calculator("whatever")
