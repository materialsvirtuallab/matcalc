"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

import functools
from inspect import isclass
from typing import TYPE_CHECKING, Any

import ase.optimize
from ase.optimize.optimize import Optimizer

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

# Listing of supported customized calculators.
CUSTOMIZED_CALCULATORS = (
    "MatGL",
    "MTP",
    "GAP",
    "NNP",
    "SNAP",
    "QSNAP",
    "ACE",
)

# Listing of supported universal calculators.
UNIVERSAL_CALCULATORS = (
    "M3GNet",
    "M3GNet-MP-2021.2.8-PES",
    "M3GNet-MP-2021.2.8-DIRECT-PES",
    "CHGNet",
    "MACE",
    "SevenNet",
)

def get_customized_calculator(name: str | Calculator, **kwargs: Any) -> Calculator:
    """Helper method to get some well-known **customized** calculators.
    Imports should be inside if statements to ensure that all models are optional dependencies.

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

    if name.lower().startswith("matgl"):
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        model = matgl.load_model(path=kwargs.get("path"))
        kwargs.setdefault("stress_weight", 1 / 160.21766208)
        return M3GNetCalculator(potential=model, **kwargs)

    if name.lower().startswith("mtp"):
        from maml.apps.pes import MTPotential

        model = MTPotential.from_config(filename=kwargs.get("filename"),
                                        elements=kwargs.get("elements"),
                                       )
        return PotentialCalculator(potential=model, **kwargs)

    if name.lower().startswith("gap"):
        from maml.apps.pes import GAPotential

        model = GAPotential.from_config(filename=kwargs.get("filename"))
        return PotentialCalculator(potential=model, **kwargs)

    if name.lower().startswith("nnp"):
        from maml.apps.pes import NNPotential

        model = NNPotential.from_config(input_filename=kwargs.get("input_filename"),
                                        scaling_filename=kwargs.get("scaling_filename"),
                                        weights_filenames=kwargs.get("weights_filenames"),
                                       )
        return PotentialCalculator(potential=model, **kwargs)

    if name.lower().startswith(("qsnap", "snap")):
        from maml.apps.pes import SNAPotential

        model = SNAPotential.from_config(param_file=kwargs.get("param_file"),
                                         coeff_file=kwargs.get("coeff_file"),
                                        )
        return PotentialCalculator(potential=model, **kwargs)

    if name.lower().startswith("ace"):
        from pyace import PyACECalculator

        return PyACECalculator(**kwargs)

    raise ValueError(f"Unrecognized {name=}, must be one of {CUSTOMIZED_CALCULATORS}")

@functools.lru_cache
def get_universal_calculator(name: str | Calculator, **kwargs: Any) -> Calculator:
    """Helper method to get some well-known **universal** calculators.
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

    if name.lower() == "sevennet":
        from sevenn.sevennet_calculator import SevenNetCalculator

        return SevenNetCalculator(**kwargs)

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

class PotentialCalculator(Calculator):
    """Potential calculator for ASE."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(self, potential, stress_weight: float = 1 / 160.21766208, **kwargs: Any):
        """
        Init PotentialCalculator with a Potential from maml.

        Args:
            potential (Potential): maml.apps.pes.Potential
            stress_weight (float): conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa.
                Default to 1 / 160.21766208.
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.stress_weight = stress_weight
    def calculate(self, atoms: Atoms | None = None, properties: list | None = None, system_changes: list | None = None):
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        from ase.calculators.calculator import all_changes, all_properties
        from pymatgen.io.ase import AseAtomsAdaptor

        from maml.apps.pes import EnergyForceStress

        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        structure = AseAtomsAdaptor.get_structure(atoms)
        efs_calculator = EnergyForceStress(ff_settings=self.potential)
        energy, forces, stresses = efs_calculator.calculate([structure])[0]

        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stresses*self.stress_weight,
        }
