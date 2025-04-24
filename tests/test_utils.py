from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from matcalc import RelaxCalc
from matcalc.utils import (
    MODEL_ALIASES,
    UNIVERSAL_CALCULATORS,
    PESCalculator,
)

DIR = Path(__file__).parent.absolute()

if TYPE_CHECKING:
    from pymatgen.core import Structure


def _map_calculators_to_packages(calculators: UNIVERSAL_CALCULATORS) -> dict[str, str]:  # Think
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
        lower_calc = calc.name.lower()
        for prefixes, package in prefix_package_map:
            if any(lower_calc.startswith(prefix) for prefix in prefixes):
                calculator_to_package[calc.name] = package
                break
    return calculator_to_package


UNIVERSAL_TO_PACKAGE = _map_calculators_to_packages(UNIVERSAL_CALCULATORS)


@pytest.mark.parametrize(
    ("expected_unit", "expected_weight"),
    [
        ("eV/A3", 0.006241509125883258),
        ("GPa", 1.0),
    ],
)
@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
def test_pescalculator_load_mtp(expected_unit: str, expected_weight: float) -> None:
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
def test_pescalculator_load_gap() -> None:
    calc = PESCalculator.load_gap(filename=DIR / "pes" / "GAP-NiMo-2020.3-PES" / "gap.xml")
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
def test_pescalculator_load_nnp() -> None:
    calc = PESCalculator.load_nnp(
        input_filename=DIR / "pes" / "NNP-Cu-2020.1-PES" / "input.nn",
        scaling_filename=DIR / "pes" / "NNP-Cu-2020.1-PES" / "scaling.data",
        weights_filenames=[DIR / "pes" / "NNP-Cu-2020.1-PES" / "weights.029.data"],
    )
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
def test_pescalculator_load_snap() -> None:
    for name in ("SNAP", "qSNAP"):
        calc = PESCalculator.load_snap(
            param_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapparam",
            coeff_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapcoeff",
        )
        assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("pyace"), reason="pyace is not installed")
def test_pescalculator_load_ace() -> None:
    calc = PESCalculator.load_ace(basis_set=DIR / "pes" / "ACE-Cu-2021.5.15-PES" / "Cu-III.yaml")
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("nequip"), reason="nequip is not installed")
def test_pescalculator_load_nequip() -> None:
    calc = PESCalculator.load_nequip(model_path=DIR / "pes" / "NequIP-am-Al2O3-2023.9-PES" / "default.pth")
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("matgl"), reason="matgl is not installed")
def test_pescalculator_load_matgl() -> None:
    calc = PESCalculator.load_matgl(path=DIR / "pes" / "M3GNet-MP-2021.2.8-PES")
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("deepmd"), reason="deepmd is not installed")
def test_pescalculator_load_deepmd() -> None:
    calc = PESCalculator.load_deepmd(
        model_path=(DIR / "pes" / "DPA3-LAM-2025.3.14-PES" / "2025-03-14-dpa3-openlam.pth")
    )
    assert isinstance(calc, Calculator)


@pytest.mark.parametrize("name", (c.name for c in UNIVERSAL_CALCULATORS))
def test_pescalculator_load_universal(Li2O: Structure, name: str) -> None:
    if name not in UNIVERSAL_TO_PACKAGE:
        pytest.fail(f"No package mapping found for {name}. Please add it to UNIVERSAL_TO_PACKAGE.")

    pkg_name = UNIVERSAL_TO_PACKAGE[name]

    if not find_spec(pkg_name):
        pytest.skip(f"{pkg_name} is not installed. Skipping {name} test.")

    calc = PESCalculator.load_universal(name)
    assert isinstance(calc, Calculator)
    same_calc = PESCalculator.load_universal(calc)  # test ASE Calculator classes are returned as-is
    assert calc is same_calc
    # We run a basic relaxation calc to make sure that all calculators work.
    relaxed_calc = RelaxCalc(calc, relax_cell=False, relax_atoms=False)
    result = relaxed_calc.calc(Li2O)
    assert isinstance(result, dict)

    name = "whatever"
    with pytest.raises(ValueError, match=f"Unrecognized {name=}") as exc:
        PESCalculator.load_universal(name)
    assert str(exc.value) == f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}"


def test_pescalculator_calculate() -> None:
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


def test_aliases() -> None:
    # Ensures that model aliases always point to valid models.
    for v in MODEL_ALIASES.values():
        assert v in UNIVERSAL_CALCULATORS
