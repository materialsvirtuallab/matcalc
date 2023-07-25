import pytest
import matgl

from matgl.ext.ase import M3GNetCalculator

from matcalc.relaxation import RelaxCalc


def test_RelaxCalc(LiFePO4):
    potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)

    calc = RelaxCalc(calculator)
    results = calc.calc(LiFePO4)
    assert results["a"] == pytest.approx(4.755711375217371)
    assert results["b"] == pytest.approx(6.131614236614623)
    assert results["c"] == pytest.approx(10.43859339794175)
