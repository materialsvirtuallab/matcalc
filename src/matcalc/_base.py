"""Define basic API."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed

from .utils import PESCalculator

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


class PropCalc(abc.ABC):
    """
    Abstract base class for property calculations.

    This class defines the interface for performing property calculations on
    structures (using pymatgen's Structure objects or a dictionary containing a
    pymatgen structure). Subclasses are expected to implement the `calc` method
    to define specific property calculation logic. Additionally, this class provides
    an implementation of the `calc_many` method, which enables concurrent calculations
    on multiple structures using joblib.
    """

    @property
    def calculator(self) -> Calculator:
        """
        This method returns the Calculator object associated with the current instance.

        Parameters:
            None

        Returns:
            Calculator: The Calculator object associated with the current instance.
        """
        return self._pes_calculator

    @calculator.setter
    def calculator(self, val: str | Calculator) -> None:
        """
        Set the calculator for PES calculation.

        Parameters:
            val (str | Calculator): A string path to load a PESCalculator or a PESCalculator object.

        Return:
            None

        """
        self._pes_calculator = PESCalculator.load_universal(val) if isinstance(val, str) else val

    @abc.abstractmethod
    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Abstract method to calculate and return a standardized format of structural data.

        This method processes input structural data, which could either be a dictionary
        or a pymatgen Structure object, and returns a dictionary representation. If a
        dictionary is provided, it must include either the key ``final_structure`` or
        ``structure``. For a pymatgen Structure input, it will be converted to a dictionary
        with the key ``final_structure``. To support chaining, a super() call should be made
        by subclasses to ensure that the input dictionary is standardized.

        :param structure: A pymatgen Structure object or a dictionary containing structural
            data with keys such as ``final_structure`` or ``structure``.
        :type structure: Structure | dict[str, Any]

        :return: A dictionary with the key ``final_structure`` mapping to the corresponding
            structural data.
        :rtype: dict[str, Any]

        :raises ValueError: If the input dictionary does not include the required keys
            ``final_structure`` or ``structure``.
        """
        if isinstance(structure, dict):
            if "final_structure" in structure:
                return structure
            if "structure" in structure:
                return structure | {"final_structure": structure["structure"]}

            raise ValueError(
                "Structure must be either a pymatgen Structure, ASE Atoms, or a dict containing a Structure/Atoms in "
                "the final_structure or structure"
            )
        return {"final_structure": structure}

    def calc_many(
        self,
        structures: Sequence[Structure | dict[str, Any] | Atoms],
        n_jobs: None | int = None,
        allow_errors: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> Generator[dict | None, None, None]:
        """
        Calculate properties for multiple structures concurrently.

        This method leverages parallel processing to compute properties for a
        given sequence of structures. It uses the `joblib.Parallel` library to
        support multi-job execution and manage error handling behavior based
        on user configuration.

        :param structures: A sequence of `Structure` or `Atoms` objects or dictionaries
            representing the input structures to be processed. Each entry in
            the sequence is processed independently.
        :param n_jobs: The number of jobs to run in parallel. If set to None,
            joblib will determine the optimal number of jobs based on the
            system's CPU configuration.
        :param allow_errors: A boolean flag indicating whether to tolerate
            exceptions during processing. When set to True, any failed
            calculation will result in a `None` value for that structure
            instead of raising an exception.
        :param kwargs: Additional keyword arguments passed directly to
            `joblib.Parallel`, which allows customization of parallel
            processing behavior.
        :return: A generator yielding dictionaries with computed properties
            for each structure or `None` if an error occurred (depending on
            the `allow_errors` flag).
        """
        parallel = Parallel(n_jobs=n_jobs, return_as="generator", **kwargs)

        def _func(s: Structure | Atoms) -> dict | None:
            try:
                return self.calc(s)
            except Exception as ex:
                if allow_errors:
                    return None
                raise ex  # noqa:TRY201

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

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
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
        structures: Sequence[Structure | Atoms | dict[str, Any]],
        n_jobs: None | int = None,
        allow_errors: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> Generator[dict | None, None, None]:
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
