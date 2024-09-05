"""
This file defines commonly used test fixtures. These are meant to be reused in unit tests.
- Fixtures that are formulae (e.g., LiFePO4) returns the appropriate pymatgen Structure or Molecule based on the most
  commonly known structure.
- Fixtures that are prefixed with `graph_` returns a (structure, graph, state) tuple.

Given that the fixtures are unlikely to be modified by the underlying code, the fixtures are set with a scope of
"session". In the event that future tests are written that modifies the fixtures, these can be set to the default scope
of "function".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matgl
import pytest
from matgl.ext.ase import M3GNetCalculator
from pymatgen.util.testing import PymatgenTest

if TYPE_CHECKING:
    from pymatgen.core import Structure

matgl.clear_cache(confirm=False)


@pytest.fixture(scope="session")
def LiFePO4() -> Structure:
    """LiFePO4 structure as session-scoped fixture (don't modify in-place,
    will affect other tests).
    """
    return PymatgenTest.get_structure("LiFePO4")


@pytest.fixture(scope="session")
def Li2O() -> Structure:
    """Li2O structure as session-scoped fixture."""
    return PymatgenTest.get_structure("Li2O")


@pytest.fixture(scope="session")
def M3GNetCalc() -> M3GNetCalculator:
    """M3GNet calculator as session-scoped fixture."""
    potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    return M3GNetCalculator(potential=potential, stress_weight=0.01)
