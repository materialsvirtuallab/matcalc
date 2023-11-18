import pytest

from matcalc.util import get_universal_calculator, UNIVERSAL_CALCULATORS
from ase.calculators.calculator import Calculator


def test_get_universal_calculator():
    # skip MACE until https://github.com/ACEsuit/mace/pull/230 is merged
    # maybe even permanently, since the checkpoint downloaded from figshare is 130MB+
    # and hence would slow down test a lot
    for name in {*UNIVERSAL_CALCULATORS} - {"MACE"}:
        calc = get_universal_calculator(name)
        assert isinstance(calc, Calculator)
        same_calc = get_universal_calculator(calc)  # test ASE Calculator classes are returned as-is
        assert calc is same_calc

    with pytest.raises(ValueError) as exc:
        get_universal_calculator("whatever")
    assert str(exc.value) == f"Unrecognized name='whatever', must be one of {UNIVERSAL_CALCULATORS}"
