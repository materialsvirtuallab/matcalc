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
    def calc(self, structure: Structure | dict[str, Any]) -> dict[str, Any]:
        """All PropCalc subclasses should implement a calc method that takes in a pymatgen structure
        and returns a dict. The method can return more than one property. Generally, subclasses should have a super()
        call to the abstract base method to obtain an initial result dict.

        Args:
            structure: Pymatgen structure or a dict containing a pymatgen Structure under a "final_structure" or
                "structure" key. Allowing dicts provide the means to chain calculators, e.g., do a relaxation followed
                by an elasticity calculation.

        Returns:
            dict[str, Any]: In the form {"prop_name": value}.
        """
        if isinstance(structure, dict):
            if "final_structure" in structure:
                return structure
            if "structure" in structure:
                return structure | {"final_structure": structure["structure"]}

            raise ValueError(
                "Structure must be either a pymatgen Structure or a dict containing a Structure in the"
                "final_structure or structure"
            )
        return {"final_structure": structure}

    def calc_many(
        self,
        structures: Sequence[Structure | dict[str, Any]],
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


class ChainedCalc(PropCalc):
    """A chained calculator that runs a series of PropCalcs on a structure or set of structures.

    Often, you may want to obtain multiple properties at once, e.g., perform a relaxation with a formation energy
    computation and a elasticity calculation. This can be done using this class by supplying a list of calculators.
    Note that it is likely
    """

    def __init__(self, prop_calcs: Sequence[PropCalc]) -> None:
        """
        Initialize a chained calculator.

        Args:
            prop_calcs: Sequence of prop calcs.
        """
        self.prop_calcs = tuple(prop_calcs)

    def calc(self, structure: Structure | dict[str, Any]) -> dict[str, Any]:
        """
        Runs the series of PropCalcs on a structure.

        Args:
            structure: Pymatgen structure or a dict containing a pymatgen Structure under a "final_structure" or
                "structure" key. Allowing dicts provide the means to chain calculators, e.g., do a relaxation followed
                by an elasticity calculation.

        Returns:
            dict[str, Any]: In the form {"prop_name": value}.
        """
        results = structure  # type:ignore[assignment]
        for prop_calc in self.prop_calcs:
            results = prop_calc.calc(results)  # type:ignore[assignment]
        return results  # type:ignore[return-value]

    def calc_many(
        self,
        structures: Sequence[Structure | dict[str, Any]],
        n_jobs: None | int = None,
        allow_errors: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> Generator[dict | None]:
        """Runs the sequence of PropCalc on many structures.

        Args:
            structures: List or generator of Structures.
            n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used. None is a marker for `unset` that will be interpreted as n_jobs=1
                unless the call is performed under a parallel_config() context manager that sets another value for
                n_jobs.
            allow_errors: Whether to skip failed calculations. For these calculations, None will be returned. For
                large scale calculations, you may want this to be True to avoid the entire calculation failing.
                Defaults to False.
            **kwargs: Passthrough to calc_many method of all PropCalcs.

        Returns:
            Generator of dicts.
        """
        results = structures  # type:ignore[assignment]
        for prop_calc in self.prop_calcs:
            results = prop_calc.calc_many(results, n_jobs=n_jobs, allow_errors=allow_errors, **kwargs)  # type:ignore[assignment]
        return results  # type:ignore[return-value]
