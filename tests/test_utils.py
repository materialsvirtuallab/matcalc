from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import ase.optimize
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

from matcalc.utils import (
    MODEL_ALIASES,
    UNIVERSAL_CALCULATORS,
    VALID_OPTIMIZERS,
    PESCalculator,
    get_ase_optimizer,
    get_universal_calculator,
    is_ase_optimizer,
)

DIR = Path(__file__).parent.absolute()


class TestPESCalculator:
    @pytest.mark.parametrize(
        ("expected_unit", "expected_weight"),
        [
            ("eV/A3", 0.006241509125883258),
            ("GPa", 1.0),
        ],
    )
    @pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
    def test_pescalculator_load_mtp(self, expected_unit: str, expected_weight: float) -> None:
        calc = PESCalculator.load_mtp(
            filename=DIR / "pes" / "MTP-Cu-2020.1-PES" / "fitted.mtp",
            elements=["Cu"],
        )
        assert isinstance(calc, Calculator)
        assert PESCalculator(
            potential=calc.potential,
            stress_unit=expected_unit,
        ).stress_weight == pytest.approx(expected_weight)
        with pytest.raises(ValueError, match="Unsupported stress_unit: Pa. Must be 'GPa' or 'eV/A3'."):
            PESCalculator(potential=calc.potential, stress_unit="Pa")

    @pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
    def test_pescalculator_load_gap(self) -> None:
        calc = PESCalculator.load_gap(filename=DIR / "pes" / "GAP-NiMo-2020.3-PES" / "gap.xml")
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
    def test_pescalculator_load_nnp(self) -> None:
        calc = PESCalculator.load_nnp(
            input_filename=DIR / "pes" / "NNP-Cu-2020.1-PES" / "input.nn",
            scaling_filename=DIR / "pes" / "NNP-Cu-2020.1-PES" / "scaling.data",
            weights_filenames=[DIR / "pes" / "NNP-Cu-2020.1-PES" / "weights.029.data"],
        )
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
    @pytest.mark.skipif(not find_spec("lammps"), reason="lammps is not installed")
    def test_pescalculator_load_snap(self) -> None:
        for name in ("SNAP", "qSNAP"):
            calc = PESCalculator.load_snap(
                param_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapparam",
                coeff_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapcoeff",
            )
            assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("pyace"), reason="pyace is not installed")
    def test_pescalculator_load_ace(self) -> None:
        calc = PESCalculator.load_ace(basis_set=DIR / "pes" / "ACE-Cu-2021.5.15-PES" / "Cu-III.yaml")
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("nequip"), reason="nequip is not installed")
    def test_pescalculator_load_nequip(self) -> None:
        calc = PESCalculator.load_nequip(model_path=DIR / "pes" / "NequIP-am-Al2O3-2023.9-PES" / "default.pth")
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("matgl"), reason="matgl is not installed")
    def test_pescalculator_load_matgl(self) -> None:
        calc = PESCalculator.load_matgl(path=DIR / "pes" / "M3GNet-MP-2021.2.8-PES")
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("matgl"), reason="matgl is not installed")
    @pytest.mark.skipif(not find_spec("chgnet"), reason="chgnet is not installed")
    @pytest.mark.skipif(not find_spec("mace"), reason="mace is not installed")
    @pytest.mark.skipif(not find_spec("sevenn"), reason="sevenn is not installed")
    def test_pescalculator_load_universal(self) -> None:
        for name in UNIVERSAL_CALCULATORS:
            calc = PESCalculator.load_universal(name)
            assert isinstance(calc, Calculator)
            same_calc = PESCalculator.load_universal(calc)  # test ASE Calculator classes are returned as-is
            assert calc is same_calc

        name = "whatever"
        with pytest.raises(ValueError, match=f"Unrecognized {name=}") as exc:
            get_universal_calculator(name)
        assert str(exc.value) == f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}"

        # cover edge case like https://github.com/materialsvirtuallab/matcalc/issues/14
        # where non-str and non-ASE Calculator instances are passed in
        assert get_universal_calculator(42) == 42  # test non-str input is returned as-is

    @pytest.mark.skipif(not find_spec("lammps"), reason="lammps is not installed")
    def test_pescalculator_calculate(self) -> None:
        calc = PESCalculator.load_snap(
            param_file=DIR / "pes" / "SNAP-Cu-2020.1-PES" / "SNAPotential.snapparam",
            coeff_file=DIR / "pes" / "SNAP-Cu-2020.1-PES" / "SNAPotential.snapcoeff",
        )

        atoms = Atoms(
            "Cu4",
            scaled_positions=[(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
            cell=[3.57743067] * 3,
            pbc=True,
        )

        atoms.set_calculator(calc)

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stresses = atoms.get_stress()

        assert isinstance(energy, float)
        assert list(forces.shape) == [4, 3]
        assert list(stresses.shape) == [6]


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


def test_is_ase_optimizer() -> None:
    assert is_ase_optimizer(ase.optimize.BFGS)
    assert is_ase_optimizer(Optimizer)
    assert not is_ase_optimizer(Calculator)

    for name in ("whatever", 42, -3.14):
        assert not is_ase_optimizer(name)


def test_aliases() -> None:
    # Ensures that model aliases always point to valid models.
    for v in MODEL_ALIASES.values():
        assert v in UNIVERSAL_CALCULATORS
