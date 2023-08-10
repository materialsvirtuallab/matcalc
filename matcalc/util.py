"""Some utility methods, e.g., for getting calculators from well-known sources."""
from __future__ import annotations

import functools

# Listing of supported universal calculators.
UNIVERSAL_CALCULATORS = ("M3GNet-MP-2021.2.8-PES", "M3GNet-MP-2021.2.8-DIRECT-PES", "CHGNet")


@functools.lru_cache
def get_universal_calculator(name: str, **kwargs):
    """
    Helper method to get some well-known **universal** calculators. Note that imports should be within the if
    statements to ensure that all these are optional. It should be stressed that this method is for **universal**
    calculators encompassing a wide swath of the periodic table only. Though matcalc can be used with any MLIP, even
    custom ones, it is not the intention for this method to provide a listing of all MLIPs.

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
