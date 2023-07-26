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

    def calc_many(
        self, structures: Sequence[Structure], n_jobs: None | int = None, **kwargs
    ) -> Generator[dict, None, None]:
        """
        Performs calc on many structures. The return type is a generator given that the calc method can potentially be
        reasonably expensive. It is trivial to convert the generator to a list/tuple.

        Args:
            structures: List or generator of Structures.
            n_jobs: The maximum number of concurrently running jobs, such as the number of Python worker processes when
                backend=”multiprocessing” or the size of the thread-pool when backend=”threading”. If -1 all CPUs are
                used. If 1 is given, no parallel computing code is used at all, and the behavior amounts to a simple
                python for loop. This mode is not compatible with timeout. For n_jobs below -1, (n_cpus + 1 + n_jobs)
                are used. Thus for n_jobs = -2, all CPUs but one are used. None is a marker for `unset` that will be
                interpreted as n_jobs=1 unless the call is performed under a parallel_config() context manager that
                sets another value for n_jobs.
            **kwargs: Passthrough to joblib.Parallel.

        Returns:
            Generator of dicts.
        """
        parallel = Parallel(n_jobs=n_jobs, return_as="generator", **kwargs)
        return parallel(delayed(self.calc)(s) for s in structures)
