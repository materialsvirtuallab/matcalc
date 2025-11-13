from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import NEBCalc, RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_neb_calc(LiFePO4: Structure, matpes_calculator: PESCalculator, tmp_path: Path) -> None:
    """Tests for NEBCalc class"""
    relax = RelaxCalc(matpes_calculator, fmax=0.5, relax_cell=False)
    image_start = LiFePO4.copy()
    image_start.remove_sites([2])
    image_start = relax.calc(image_start)["final_structure"]
    image_end = LiFePO4.copy()
    image_end.remove_sites([3])
    image_end = relax.calc(image_end)["final_structure"]
    neb_calc = NEBCalc(matpes_calculator, traj_folder=tmp_path, fmax=0.5)
    barriers = neb_calc.calc_images(image_start, image_end, n_images=5)
    assert barriers["barrier"] == pytest.approx(0.17913818359375044, rel=0.002)
    assert barriers["force"] == pytest.approx(-0.0042724609375, rel=0.002)
    assert isinstance(barriers["mep"], dict), "barriers['mep'] should be a dictionary"
    for key, value in barriers["mep"].items():
        assert "structure" in value, f"Missing 'structure' key in barriers['mep'][{key}]"
        assert "energy" in value, f"Missing 'energy' key in barriers['mep'][{key}]"
    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        NEBCalc(matpes_calculator, optimizer="invalid")
