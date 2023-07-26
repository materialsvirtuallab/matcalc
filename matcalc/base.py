"""Define basic API."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from joblib import Parallel, delayed

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from pymatgen.core import Structure


class PropCalc(metaclass=abc.ABCMeta):
    """API for a property calculator."""

    @abc.abstractmethod
    def calc(self, structure: Structure) -> dict:
        """
        All PropCalc should implement a calc method that takes in a pymatgen structure and returns a dict. Note that
        the method can return more than one property.

        Args:
            structure: Pymatgen structure.

        Returns: {"prop name": value}
        """

    def calc_many(self, structures: Sequence[Structure], n_jobs: None | int = None) -> Generator[dict, None, None]:
        """
        Performs calc on many structures. The return type is a generator given that the calc method can potentially be
        reasonably expensive. It is trivial to convert the generator to a list/tuple.

        Args:
            structures: List or generator of Structures.
            n_jobs: Number of processes to use for multiprocessing.Pool. Defaults to None, i.e., all CPUs.

        Returns:
            Generator of dicts.
        """
        parallel = Parallel(n_jobs=n_jobs, return_as="generator")
        return parallel(delayed(self.calc)(s) for s in structures)
