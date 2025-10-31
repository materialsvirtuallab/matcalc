"""Tests for the SurfaceCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

from matcalc import AdsorptionCalc
from matcalc.utils import to_ase_atoms

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure
    from pymatgen.core.surface import Slab
    from ase import Atoms

@pytest.fixture(scope="module")
def Pt_bulk() -> Structure:
    """Pt bulk as module-scoped fixture."""
    struct = Structure(
        Lattice.cubic(3.924), ["Pt"]*4,
        [[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]
    )
    return struct

@pytest.fixture(scope="module")
def Pt_slab(Pt_bulk: Structure) -> Slab:
    """Pt slab as module-scoped fixture."""
    slabgen = SlabGenerator(Pt_bulk, (1,1,1), 10, 10)
    return slabgen.get_slab()

@pytest.fixture(scope="module")
def Pt_slab_atoms(Pt_slab: Slab) -> Atoms:
    """Si slab as session-scoped fixture."""
    return to_ase_atoms(Pt_slab)

@pytest.fixture(scope="module")
def CO2() -> Molecule:
    return Molecule("COO", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])

@pytest.fixture(scope="module")
def Pt_adslab(Pt_slab: Slab, CO2: Molecule):
    asf = AdsorbateSiteFinder(Pt_slab)
    adsite = asf.find_adsorption_sites()['all'][0]
    return asf.add_adsorbate(CO2, adsite)

@pytest.mark.parametrize("test_input, expected",[
    ({'slab_energy_per_atom': 1.0}, (1.0, -22.68405)),
    ({'adsorbate_energy': 4.0}, (-5.8437, 4.0)),
    ({'slab': Pt_slab_atoms, 'slab_energy_per_atom': 1.0, 'adsorbate_energy': 4.0}, (1.0, 4.0))
])
def test_adsorption_calc_slab_inputs(
    Pt_slab: Slab,
    CO2: Molecule,
    Pt_adslab: Structure,
    m3gnet_calculator: PESCalculator,
    test_input: dict,
    expected: float,
) -> None:

    ad_calc = AdsorptionCalc(
        calculator=m3gnet_calculator,
        relax_slab=False,
        relax_bulk=False,
        relax_adsorbate=False,
    )
    structure = {
        "slab": Pt_slab,
        "adsorbate": CO2,
        "adslab": Pt_adslab,
    } | test_input

    results = ad_calc.calc(structure)

    assert(results['slab_energy_per_atom']) == pytest.approx(expected[0], rel=1e-1)
    assert(results['adsorbate_energy']) == pytest.approx(expected[1], rel=1e-1)

def test_adsorption_calc_adslabs(
    Pt_bulk: Slab,
    CO2: Molecule,
    m3gnet_calculator: PESCalculator,
) -> None:
    
    ad_calc = AdsorptionCalc(
        calculator=m3gnet_calculator,
        relax_slab=True,
        relax_bulk=True,
        relax_adsorbate=False,
    )

    results = ad_calc.calc_adslabs(
        adsorbate=CO2,
        adsorbate_energy=1.0,
        bulk=Pt_bulk,
        miller_index=(1,1,1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        inplane_supercell=(3, 3),
        slab_gen_kwargs={
            "center_slab": True,
        },
        adsorption_sites="hollow",
        fixed_height=4.0,
        dry_run=True,
    )

    assert len(results) == 2, "There are two hollow sites on Pt (111) surface."
    assert "adslab" in results[0]
    assert results[0]["adsorption_site"] == "hollow"
    assert results[1]["adsorption_site_index"] == 1
    assert len(results[0]["adslab"]) == 54 + 3
    assert "Pt" in results[0]["adslab"].symbol_set
    assert "C" in results[0]["adslab"].symbol_set
    assert "O" in results[0]["adslab"].symbol_set
    assert "selective_dynamics" in results[0]["adslab"].site_properties
    assert results[0]["adsorbate_energy"] == pytest.approx(1.0, rel=1e-1)
    assert results[0]["slab_energy_per_atom"] == pytest.approx(-6.016221)
