from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import MEP, NEBCalc, RelaxCalc

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
    assert isinstance(barriers["mep"], MEP), "barriers['mep'] should be an MEP instance"
    mep = barriers["mep"]
    assert len(mep.labels) > 0, "MEP should have labels"
    assert len(mep.frac_coords) == 7, "MEP should have 6 images (n_images=5 + 2 endpoints)"
    assert len(mep.energies) == 7, "MEP should have 6 energies"
    assert mep.lattice.shape == (3, 3), "MEP should have a 3x3 lattice matrix"
    # Test as_dict method
    mep_dict = mep.as_dict()
    assert "labels" in mep_dict, "as_dict() should contain 'labels'"
    assert "lattice" in mep_dict, "as_dict() should contain 'lattice'"
    assert "images" in mep_dict, "as_dict() should contain 'images'"
    assert len(mep_dict["images"]) == 7, "as_dict() should have 6 images"
    for img in mep_dict["images"]:
        assert "frac_coords" in img, "Each image should have 'frac_coords'"
        assert "energy" in img, "Each image should have 'energy'"

    # Test from_dict method
    mep_reconstructed = MEP.from_dict(mep_dict)
    assert len(mep_reconstructed.labels) == len(mep.labels), "Reconstructed MEP should have same number of labels"
    assert len(mep_reconstructed.frac_coords) == len(mep.frac_coords), (
        "Reconstructed MEP should have same number of images"
    )
    assert len(mep_reconstructed.energies) == len(mep.energies), "Reconstructed MEP should have same number of energies"
    assert mep_reconstructed.lattice.shape == mep.lattice.shape, "Reconstructed MEP should have same lattice shape"
    # Check that energies match
    for orig_energy, recon_energy in zip(mep.energies, mep_reconstructed.energies, strict=False):
        assert orig_energy == pytest.approx(recon_energy), "Reconstructed energies should match"

    # Test get_structures method
    structures = mep.get_structures()
    assert len(structures) == 7, "get_structures() should return 6 structures"
    for i, struct in enumerate(structures):
        assert struct.num_sites == len(mep.labels), f"Structure {i} should have same number of sites as labels"
        # Check that fractional coordinates match (within numerical precision)
        assert struct.frac_coords.shape == mep.frac_coords[i].shape, (
            f"Structure {i} should have matching frac_coords shape"
        )

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        NEBCalc(matpes_calculator, optimizer="invalid")
