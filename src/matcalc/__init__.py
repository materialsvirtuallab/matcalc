"""Calculators for materials properties."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matcalc")
except PackageNotFoundError:
    pass  # package not installed

from ._alamode import AlamodeCalc
from ._base import ChainedCalc, PropCalc
from ._elasticity import ElasticityCalc
from ._eos import EOSCalc
from ._fourphonon import FourPhononCalc
from ._lammps import LAMMPSMDCalc
from ._md import MDCalc
from ._neb import NEBCalc
from ._pheasy import PheasyCalc
from ._phonon import PhononCalc
from ._phonon3 import Phonon3Calc
from ._qha import QHACalc
from ._relaxation import RelaxCalc
from ._stability import EnergeticsCalc
from ._surface import SurfaceCalc
from .config import SIMULATION_BACKEND, clear_cache
from .utils import UNIVERSAL_CALCULATORS, PESCalculator

# Provide an alias for loading calculators quickly.
load_up = PESCalculator.load_universal
load_fp = PESCalculator.load_universal
