from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc.neb import NEBCalc

if TYPE_CHECKING:
    from pathlib import Path

    from matgl.ext.ase import M3GNetCalculator
    from pymatgen.core import Structure


def test_neb_calc(LiFePO4: Structure, M3GNetCalc: M3GNetCalculator, tmp_path: Path) -> None:
    """Tests for NEBCalc class"""
    image_start = LiFePO4.copy()
    image_start.remove_sites([2])
    image_end = LiFePO4.copy()
    image_end.remove_sites([3])
    NEBcalc = NEBCalc.from_end_images(image_start, image_end, M3GNetCalc, n_images=5, traj_folder=tmp_path)
    barriers = NEBcalc.calc(fmax=0.5)
    print(barriers)
    assert len(NEBcalc.neb.images) == 7
    assert barriers[0] == pytest.approx(0.0184783935546875, rel=0.002)
    assert barriers[1] == pytest.approx(0.0018920898, rel=0.002)

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        NEBcalc.from_end_images(image_start, image_end, M3GNetCalc, optimizer="invalid")
