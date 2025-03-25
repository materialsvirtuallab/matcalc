from __future__ import annotations

import os
from importlib.util import find_spec

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

DIR = os.path.abspath(os.path.dirname(__file__))


def _map_calculators_to_packages(calculators: list[str]) -> dict[str, str]:
    prefix_package_map: list[tuple[tuple[str, ...], str]] = [
        (("m3gnet", "chgnet", "tensornet", "pbe", "r2scan"), "matgl"),
        (("mace",), "mace"),
        (("sevennet",), "sevenn"),
        (("grace", "tensorpotential"), "tensorpotential"),
        (("orb",), "orb_models"),
        (("mattersim",), "mattersim"),
        (("fairchem",), "fairchem"),
        (("petmad",), "pet_mad"),
        (("deepmd",), "deepmd"),
    ]

    calculator_to_package: dict[str, str] = {}

    for calc in calculators:
        lower_calc = calc.lower()
        for prefixes, package in prefix_package_map:
            if any(lower_calc.startswith(prefix) for prefix in prefixes):
                calculator_to_package[calc] = package
                break
    return calculator_to_package


UNIVERSAL_TO_PACKAGE = _map_calculators_to_packages(UNIVERSAL_CALCULATORS)


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
            filename=os.path.join(DIR, "pes/MTP-Cu-2020.1-PES", "fitted.mtp"),
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
        calc = PESCalculator.load_gap(filename=os.path.join(DIR, "pes/GAP-NiMo-2020.3-PES", "gap.xml"))
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
    def test_pescalculator_load_nnp(self) -> None:
        calc = PESCalculator.load_nnp(
            input_filename=os.path.join(DIR, "pes/NNP-Cu-2020.1-PES", "input.nn"),
            scaling_filename=os.path.join(DIR, "pes/NNP-Cu-2020.1-PES", "scaling.data"),
            weights_filenames=[os.path.join(DIR, "pes/NNP-Cu-2020.1-PES", "weights.029.data")],
        )
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
    @pytest.mark.skipif(not find_spec("lammps"), reason="lammps is not installed")
    def test_pescalculator_load_snap(self) -> None:
        for name in ("SNAP", "qSNAP"):
            calc = PESCalculator.load_snap(
                param_file=os.path.join(DIR, f"pes/{name}-Cu-2020.1-PES", "SNAPotential.snapparam"),
                coeff_file=os.path.join(DIR, f"pes/{name}-Cu-2020.1-PES", "SNAPotential.snapcoeff"),
            )
            assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("pyace"), reason="pyace is not installed")
    def test_pescalculator_load_ace(self) -> None:
        calc = PESCalculator.load_ace(basis_set=os.path.join(DIR, "pes/ACE-Cu-2021.5.15-PES", "Cu-III.yaml"))
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("nequip"), reason="nequip is not installed")
    def test_pescalculator_load_nequip(self) -> None:
        calc = PESCalculator.load_nequip(model_path=os.path.join(DIR, "pes/NequIP-am-Al2O3-2023.9-PES", "default.pth"))
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("matgl"), reason="matgl is not installed")
    def test_pescalculator_load_matgl(self) -> None:
        calc = PESCalculator.load_matgl(path=os.path.join(DIR, "pes/M3GNet-MP-2021.2.8-PES"))
        assert isinstance(calc, Calculator)

    @pytest.mark.skipif(not find_spec("deepmd"), reason="deepmd-kit is not installed")
    def test_pescalculator_load_deepmd(self) -> None:
        calc = PESCalculator.load_deepmd(
            model_path=os.path.join(DIR, "pes/DPA2-medium-28-10M-rc0-MPTraj-PES", "frozen_model.pth")
        )
        assert isinstance(calc, Calculator)

    @pytest.mark.parametrize("name", UNIVERSAL_CALCULATORS)
    def test_pescalculator_load_universal(self, name: str) -> None:
        if name not in UNIVERSAL_TO_PACKAGE:
            pytest.fail(f"No package mapping found for {name}. Please add it to UNIVERSAL_TO_PACKAGE.")

        pkg_name = UNIVERSAL_TO_PACKAGE[name]

        if not find_spec(pkg_name):
            pytest.skip(f"{pkg_name} is not installed. Skipping {name} test.")

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
            param_file=os.path.join(DIR, "pes/SNAP-Cu-2020.1-PES", "SNAPotential.snapparam"),
            coeff_file=os.path.join(DIR, "pes/SNAP-Cu-2020.1-PES", "SNAPotential.snapcoeff"),
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


@pytest.mark.skipif(not find_spec("matgl"), reason="matgl is not installed")
def test_aliases() -> None:
    # Ensures that model aliases always point to valid models.
    for v in MODEL_ALIASES.values():
        assert v in UNIVERSAL_CALCULATORS
