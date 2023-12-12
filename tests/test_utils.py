from __future__ import annotations

import ase.optimize
import pytest
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

from matcalc.utils import (
    UNIVERSAL_CALCULATORS,
    VALID_OPTIMIZERS,
    get_ase_optimizer,
    get_universal_calculator,
    is_ase_optimizer,
)


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


def test_get_ase_optimizer() -> None:
    for name in dir(ase.optimize):
        if is_ase_optimizer(name):
            optimizer = get_ase_optimizer(name)
            assert issubclass(optimizer, Optimizer)
            same_optimizer = get_ase_optimizer(optimizer)  # test ASE Optimizer classes are returned as-is
            assert optimizer is same_optimizer

    for optimizer in ("whatever", 42):
        with pytest.raises(ValueError, match=f"Unknown {optimizer=}") as exc:
            get_ase_optimizer(optimizer)
        assert str(exc.value) == f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}"
