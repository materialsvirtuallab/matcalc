from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import NEBCalc

if TYPE_CHECKING:
    from pathlib import Path

    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_neb_calc(LiFePO4: Structure, m3gnet_calculator: PESCalculator, tmp_path: Path) -> None:
    """Tests for NEBCalc class"""
    image_start = LiFePO4.copy()
    image_start.remove_sites([2])
    image_end = LiFePO4.copy()
    image_end.remove_sites([3])
    neb_calc = NEBCalc(m3gnet_calculator, traj_folder=tmp_path, fmax=0.5)
    barriers = neb_calc.calc_images(image_start, image_end, n_images=5)
    assert barriers["barrier"] == pytest.approx(0.0184783935546875, rel=0.002)
    assert barriers["force"] == pytest.approx(0.0018920898, rel=0.002)
    assert isinstance(barriers["mep"], dict), "barriers['mep'] should be a dictionary"
    for key, value in barriers["mep"].items():
        assert "structure" in value, f"Missing 'structure' key in barriers['mep'][{key}]"
        assert "energy" in value, f"Missing 'energy' key in barriers['mep'][{key}]"
    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        NEBCalc(m3gnet_calculator, optimizer="invalid")
