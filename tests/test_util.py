import pytest

from matcalc.util import get_universal_calculator, UNIVERSAL_CALCULATORS
from ase.calculators.calculator import Calculator


def test_get_calculator():
    for name in UNIVERSAL_CALCULATORS:
        calc = get_universal_calculator(name)
        assert isinstance(calc, Calculator)

    with pytest.raises(ValueError, match="Unsupported name='whatever'"):
        get_universal_calculator("whatever")
