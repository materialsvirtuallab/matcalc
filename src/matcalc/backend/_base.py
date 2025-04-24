from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    from pymatgen.core.structure import Structure


class PESResult(NamedTuple):
    """Container for results from PES calculators."""

    structure: Structure
    energy: float
    forces: np.ndarray
    stress: np.ndarray
