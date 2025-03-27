"""Calculators for materials properties."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matcalc")
except PackageNotFoundError:
    pass  # package not installed

from .base import ChainedCalc, PropCalc
from .config import clear_cache
from .elasticity import ElasticityCalc
from .eos import EOSCalc
from .neb import NEBCalc
from .phonon import PhononCalc
from .qha import QHACalc
from .relaxation import RelaxCalc
from .stability import EnergeticsCalc
from .utils import UNIVERSAL_CALCULATORS, PESCalculator
