"""Some utility methods, e.g., for getting calculators from well-known sources."""
from __future__ import annotations

import functools


@functools.lru_cache
def get_calculator(name: str, **kwargs):
    """
    Helper method to get some well-known calculators. Note that imports should be within the if statements to ensure
    that all these are optional.

    Args:
        name (str): Name of calculator.
        **kwargs: Passthrough to calculator init.

    Returns:
        Calculator
    """
    if name in ("M3GNet", "M3GNet-MP-2021.2.8-PES", "M3GNet-MP-2021.2.8-DIRECT-PES"):
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        if name == "M3GNet":
            # M3GNet is shorthand for latest M3GNet based on DIRECT sampling.
            name = "M3GNet-MP-2021.2.8-DIRECT-PES"
        potential = matgl.load_model(name)
        return M3GNetCalculator(potential=potential, stress_weight=0.01, **kwargs)

    if name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet

        return CHGNetCalculator(CHGNet.load(), stress_weight=0.01, **kwargs)

    raise ValueError(f"Unsupported model name: {name}")
