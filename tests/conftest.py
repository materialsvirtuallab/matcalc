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

import pytest
from pymatgen.util.testing import PymatgenTest

import matcalc
from matcalc.utils import PESCalculator

if TYPE_CHECKING:
    from pymatgen.core import Structure

import matgl

matgl.clear_cache(confirm=False)
matcalc.clear_cache(confirm=False)


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
def m3gnet_calculator() -> PESCalculator:
    """M3GNet calculator as session-scoped fixture."""
    return PESCalculator.load_matgl("M3GNet-MP-2021.2.8-PES")


@pytest.fixture(scope="session")
def matpes_calculator() -> PESCalculator:
    """TensorNet calculator as session-scoped fixture."""
    return PESCalculator.load_matgl("TensorNet-MatPES-PBE-v2025.1-PES")
