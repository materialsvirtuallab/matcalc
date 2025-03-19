"""Calculators for materials properties."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matcalc")
except PackageNotFoundError:
    pass  # package not installed

from .base import PropCalc
from .elasticity import ElasticityCalc
from .eos import EOSCalc
from .neb import NEBCalc
from .phonon import PhononCalc
from .qha import QHACalc
from .relaxation import RelaxCalc
from .utils import PESCalculator
