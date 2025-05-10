from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    from pymatgen.core.structure import Structure


class SimulationResult(NamedTuple):
    """Container for results from PES calculators."""

    structure: Structure
    potential_energy: float
    kinetic_energy: float
    energy: float
    forces: np.typing.NDArray[np.float64]
    stress: np.typing.NDArray[np.float64]
