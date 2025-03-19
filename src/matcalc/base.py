"""Define basic API."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from pymatgen.core import Structure


class PropCalc(abc.ABC):
    """API for a property calculator."""

    @abc.abstractmethod
    def calc(self, structure: Structure) -> dict:
        """All PropCalc subclasses should implement a calc method that takes in a pymatgen structure
        and returns a dict. The method can return more than one property.

        Args:
            structure: Pymatgen structure.

        Returns:
            dict[str, Any]: In the form {"prop_name": value}.
        """

    def calc_many(
        self,
        structures: Sequence[Structure],
        n_jobs: None | int = None,
        allow_errors: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> Generator[dict | None]:
        """Performs calc on many structures. The return type is a generator given that the calc method can
        potentially be expensive. It is trivial to convert the generator to a list/tuple.

        Args:
            structures: List or generator of Structures.
            n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used. None is a marker for `unset` that will be interpreted as n_jobs=1
                unless the call is performed under a parallel_config() context manager that sets another value for
                n_jobs.
            allow_errors: Whether to skip failed calculations. For these calculations, None will be returned. For
                large scale calculations, you may want this to be True to avoid the entire calculation failing.
                Defaults to False.
            **kwargs: Passthrough to joblib.Parallel.

        Returns:
            Generator of dicts.
        """
        parallel = Parallel(n_jobs=n_jobs, return_as="generator", **kwargs)
        if allow_errors:

            def _func(s: Structure) -> dict | None:
                try:
                    return self.calc(s)
                except Exception:  # noqa: BLE001
                    return None

        else:
            _func = self.calc  # type: ignore[assignment]
        return parallel(delayed(_func)(s) for s in structures)
