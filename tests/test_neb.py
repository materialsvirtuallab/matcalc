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
    neb_calc = NEBCalc.from_end_images(m3gnet_calculator, image_start, image_end, n_images=5, traj_folder=tmp_path)
    barriers = neb_calc.calc(fmax=0.5)
    assert len(neb_calc.neb.images) == 7
    assert barriers[0] == pytest.approx(0.0184783935546875, rel=0.002)
    assert barriers[1] == pytest.approx(0.0018920898, rel=0.002)

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        neb_calc.from_end_images(m3gnet_calculator, image_start, image_end, optimizer="invalid")
