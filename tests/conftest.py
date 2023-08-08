"""
Define commonly used text fixtures. These are meant to be reused in unittests.
- Fixtures that are formulae (e.g., LiFePO4) returns the appropriate pymatgen Structure or Molecule based on the most
  commonly known structure.
- Fixtures that are prefixed with `graph_` returns a (structure, graph, state) tuple.

Given that the fixtures are unlikely to be modified by the underlying code, the fixtures are set with a scope of
"session". In the event that future tests are written that modifies the fixtures, these can be set to the default scope
of "function".
"""
from __future__ import annotations

import pytest
import matgl
import m3gnet

from pymatgen.util.testing import PymatgenTest
from matgl.ext.ase import M3GNetCalculator
from m3gnet.models import M3GNetCalculator as M3GNetCalculator_tf
from m3gnet.models import M3GNet as M3GNet_tf
from m3gnet.models._base import Potential as Potential_tf

matgl.clear_cache(confirm=False)


@pytest.fixture(scope="session")
def LiFePO4():
    return PymatgenTest.get_structure("LiFePO4")


@pytest.fixture(scope="session")
def Li2O():
    return PymatgenTest.get_structure("Li2O")


@pytest.fixture(scope="session")
def M3GNetUPCalc():
    potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    return M3GNetCalculator(potential=potential, stress_weight=0.01)


# @pytest.fixture(scope="session")
# def M3GNetUPCalc_tf():
#     potential = Potential_tf(M3GNet_tf.load())
#     return M3GNetCalculator_tf(potential=potential, stress_weight=0.01)
