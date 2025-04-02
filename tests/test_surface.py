"""Tests for the SurfaceCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import SurfaceCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_surface_calc_basic(Si: Structure, m3gnet_calculator: PESCalculator) -> None:
    """
    Test the basic workflow:
      1) calc_slabs on a known structure
      2) calc on the resulting slabs
      3) Check the final results
    """
    surf_calc = SurfaceCalc(
        calculator=m3gnet_calculator,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        relax_bulk=True,
        relax_slab=True,
        fmax=0.1,
        max_steps=100,
    )

    slabs = surf_calc.calc_slabs(Si)
    assert len(slabs) == 2, "Expected two slabs for Silicon (111)."

    results = surf_calc.calc(slabs)
    assert len(results) == len(slabs), "Should return a result for each slab key."

    it = iter(results)
    first_key = next(it)
    slab_res = results[first_key]

    assert "bulk_energy" in slab_res
    assert "final_bulk_structure" in slab_res
    assert "slab_energy" in slab_res
    assert "surface_energy" in slab_res
    assert "final_slab_structure" in slab_res
    assert "initial_slab_structure" in slab_res

    assert slab_res["bulk_energy"] == pytest.approx(-43.35231018066406, rel=1e-1)
    assert slab_res["slab_energy"] == pytest.approx(-42.81388473510742, rel=1e-1)
    assert slab_res["surface_energy"] == pytest.approx(0.020886063055827422, rel=1e-1)

    second_key = next(it)
    slab_res = results[second_key]

    assert slab_res["bulk_energy"] == pytest.approx(-43.35231018066406, rel=1e-1)
    assert slab_res["slab_energy"] == pytest.approx(-39.40110397338867, rel=1e-1)
    assert slab_res["surface_energy"] == pytest.approx(0.15327125170767797, rel=1e-1)


def test_surface_calc_no_bulk_data_error(Si: Structure, m3gnet_calculator: PESCalculator) -> None:
    """
    Ensure that calling calc(...) before calc_slabs(...)
    raises an error because bulk data is None.
    """
    surf_calc = SurfaceCalc(calculator=m3gnet_calculator)
    with pytest.raises(ValueError, match="Bulk energy, number of atoms, or structure is not initialized."):
        surf_calc.calc({"slab_00": Si})


def test_surface_calc_invalid_input(Si: Structure, m3gnet_calculator: PESCalculator) -> None:
    """
    If the user passes a non-dict to calc, it should raise ValueError.
    """
    surf_calc = SurfaceCalc(calculator=m3gnet_calculator)
    with pytest.raises(
        ValueError,
        match="For surface calculations, \
                    structure must be a dict containing the images with keys slab_00, slab_01, etc.",
    ):
        surf_calc.calc(Si)


def test_surface_calc_many(Si: Structure, m3gnet_calculator: PESCalculator) -> None:
    """
    Test parallel usage by calling calc_many on multiple slabs.
    """
    surf_calc = SurfaceCalc(
        calculator=m3gnet_calculator,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        relax_bulk=True,
        relax_slab=True,
        fmax=0.1,
        max_steps=100,
    )

    slabs = surf_calc.calc_slabs(Si)
    assert len(slabs) == 2, "Expected two slabs for Silicon (111)."

    results_gen = surf_calc.calc_many([slabs], n_jobs=2, allow_errors=False)
    results_list = list(results_gen)

    assert len(results_list) == len(slabs)

    combined = (results_list[0] or {}) | (results_list[1] or {})
    assert len(combined) == len(slabs)

    slab_data = (
        (-43.35231018066406, -42.81388473510742, 0.020886063055827422),
        (-43.35231018066406, -39.40110397338867, 0.15327125170767797),
    )

    for i, k in enumerate(combined):
        assert combined[k]["bulk_energy"] == pytest.approx(slab_data[i][0], rel=1e-1)
        assert combined[k]["slab_energy"] == pytest.approx(slab_data[i][1], rel=1e-1)
        assert combined[k]["surface_energy"] == pytest.approx(slab_data[i][2], rel=1e-1)


def test_surface_calc_many_invalid_input(Si: Structure, m3gnet_calculator: PESCalculator) -> None:
    """
    If the user passes a invalid input to calc_many, it should raise ValueError.
    """
    surf_calc = SurfaceCalc(calculator=m3gnet_calculator)
    with pytest.raises(
        ValueError,
        match="structures must be a sequence containing one dict",
    ):
        surf_calc.calc_many([Si, Si])
