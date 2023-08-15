"""Some utility methods, e.g., for getting calculators from well-known sources."""
from __future__ import annotations

import functools

from ase.calculators.calculator import Calculator

# Listing of supported universal calculators.
UNIVERSAL_CALCULATORS = ("M3GNet-MP-2021.2.8-PES", "M3GNet-MP-2021.2.8-DIRECT-PES", "CHGNet")


@functools.lru_cache
def get_universal_calculator(name: str | Calculator, **kwargs) -> Calculator:
    """
    Helper method to get some well-known **universal** calculators.
    Imports should be inside if statements to ensure that all models are optional dependencies.
    All calculators must be universal, i.e. encompass a wide swath of the periodic table.
    Though matcalc can be used with any MLIP, even custom ones, this function is not meant as
        a list of all MLIPs.

    Args:
        name (str): Name of calculator.
        **kwargs: Passthrough to calculator init.

    Returns:
        Calculator
    """
    if isinstance(name, Calculator):
        return name

    if name in ("M3GNet", "M3GNet-MP-2021.2.8-PES", "M3GNet-MP-2021.2.8-DIRECT-PES"):
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        # M3GNet is shorthand for latest M3GNet based on DIRECT sampling.
        name = {"M3GNet": "M3GNet-MP-2021.2.8-PES"}.get(name, name)
        model = matgl.load_model(name)
        kwargs.setdefault("stress_weight", 0.01)
        return M3GNetCalculator(potential=model, **kwargs)

    if name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(**kwargs)

    raise ValueError(f"Unsupported {name=}")
