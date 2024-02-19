"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

import functools
from inspect import isclass
from typing import TYPE_CHECKING, Any

import ase.optimize
from ase.optimize.optimize import Optimizer

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

# Listing of supported universal calculators.
UNIVERSAL_CALCULATORS = (
    "M3GNet",
    "M3GNet-MP-2021.2.8-PES",
    "M3GNet-MP-2021.2.8-DIRECT-PES",
    "CHGNet",
    "MACE",
)


@functools.lru_cache
def get_universal_calculator(name: str | Calculator, **kwargs: Any) -> Calculator:
    """
    Helper method to get some well-known **universal** calculators.
    Imports should be inside if statements to ensure that all models are optional dependencies.
    All calculators must be universal, i.e. encompass a wide swath of the periodic table.
    Though matcalc can be used with any MLIP, even custom ones, this function is not meant as
        a list of all MLIPs.

    Args:
        name (str): Name of calculator.
        **kwargs: Passthrough to calculator init.

    Raises:
        ValueError: on unrecognized model name.

    Returns:
        Calculator
    """
    if not isinstance(name, str):  # e.g. already an ase Calculator instance
        return name

    if name.lower().startswith("m3gnet"):
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        # M3GNet is shorthand for latest M3GNet based on DIRECT sampling.
        name = {"m3gnet": "M3GNet-MP-2021.2.8-DIRECT-PES"}.get(name.lower(), name)
        model = matgl.load_model(name)
        kwargs.setdefault("stress_weight", 1 / 160.21766208)
        return M3GNetCalculator(potential=model, **kwargs)

    if name.lower() == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(**kwargs)

    if name.lower() == "mace":
        from mace.calculators import mace_mp

        return mace_mp(**kwargs)

    raise ValueError(f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}")


def is_ase_optimizer(key: str | Optimizer) -> bool:
    """Check if key is the name of an ASE optimizer class."""
    if isclass(key) and issubclass(key, Optimizer):
        return True
    if isinstance(key, str):
        return isclass(obj := getattr(ase.optimize, key, None)) and issubclass(obj, Optimizer)
    return False


VALID_OPTIMIZERS = [key for key in dir(ase.optimize) if is_ase_optimizer(key)]


def get_ase_optimizer(optimizer: str | Optimizer) -> Optimizer:
    """Validate optimizer is a valid ASE Optimizer.

    Args:
        optimizer (str | Optimizer): The optimization algorithm.

    Raises:
        ValueError: on unrecognized optimizer name.

    Returns:
        Optimizer: ASE Optimizer class.
    """
    if isclass(optimizer) and issubclass(optimizer, Optimizer):
        return optimizer

    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}")

    return getattr(ase.optimize, optimizer) if isinstance(optimizer, str) else optimizer
