"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING, Any, Literal

import ase.optimize
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from monty.dev import deprecated

from .units import eVA3ToGPa

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from maml.apps.pes._lammps import LMPStaticCalculator
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration


# Listing of supported universal calculators.
# If you update UNIVERSAL_CALCULATORS, you must also update the mapping in
# map_calculators_to_packages in test_utils.py, unless already covered.
UNIVERSAL_CALCULATORS = [
    "M3GNet",
    "CHGNet",
    "MACE",
    "SevenNet",
    "TensorNet",
    "GRACE",
    "TensorPotential",
    "ORB",
    "PBE",
    "r2SCAN",
    "Mattersim",
    "FairChem",
    "PETMAD",
    "DeepMD",
]

try:
    # Auto-load all available PES models from matgl if installed.
    import matgl

    UNIVERSAL_CALCULATORS += [m for m in matgl.get_available_pretrained_models() if "PES" in m]
    UNIVERSAL_CALCULATORS = sorted(set(UNIVERSAL_CALCULATORS))
except ImportError:
    pass

# Provide simple aliases for some common models. The key in MODEL_ALIASES must be lower case.
MODEL_ALIASES = {
    "tensornet": "TensorNet-MatPES-PBE-v2025.1-PES",
    "m3gnet": "M3GNet-MatPES-PBE-v2025.1-PES",
    "chgnet": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "pbe": "TensorNet-MatPES-PBE-v2025.1-PES",
    "r2scan": "TensorNet-MatPES-r2SCAN-v2025.1-PES",
}


class PESCalculator(Calculator):
    """
    Potential calculator for ASE, supporting both **universal** and **customized** potentials, including:
        Customized potentials: MatGL(M3GNet, CHGNet, TensorNet and SO3Net), MAML(MTP, GAP, NNP, SNAP and qSNAP) and ACE.
        Universal potentials: M3GNet, CHGNet, MACE and SevenNet.
    Though MatCalc can be used with any MLIP, this method does not yet cover all MLIPs.
    Imports should be inside if statements to ensure that all models are optional dependencies.
    """

    implemented_properties = ("energy", "forces", "stress")

    def __init__(
        self,
        potential: LMPStaticCalculator,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PESCalculator with a potential from maml.

        Args:
            potential (LMPStaticCalculator): maml.apps.pes._lammps.LMPStaticCalculator
            stress_unit (str): The unit of stress. Default to "GPa"
            stress_weight (float): The conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa.
                Default to 1.0.
            **kwargs: Additional keyword arguments passed to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential

        # Handle stress unit conversion
        if stress_unit == "eV/A3":
            conversion_factor = 1 / eVA3ToGPa  # Conversion factor from GPa to eV/A^3
        elif stress_unit == "GPa":
            conversion_factor = 1.0  # No conversion needed if stress is already in GPa
        else:
            raise ValueError(f"Unsupported stress_unit: {stress_unit}. Must be 'GPa' or 'eV/A3'.")

        self.stress_weight = stress_weight * conversion_factor

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): The list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        from ase.calculators.calculator import all_changes, all_properties
        from maml.apps.pes import EnergyForceStress
        from pymatgen.io.ase import AseAtomsAdaptor

        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        structure = AseAtomsAdaptor.get_structure(atoms)
        efs_calculator = EnergyForceStress(ff_settings=self.potential)
        energy, forces, stresses = efs_calculator.calculate([structure])[0]

        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stresses * self.stress_weight,
        }

    @staticmethod
    def load_matgl(path: str | Path, **kwargs: Any) -> Calculator:
        """
        Load the MatGL model for use in ASE as a calculator.

        Args:
            path (str | Path): The path to the folder storing model.
            **kwargs (Any): Additional keyword arguments for the M3GNetCalculator.

        Returns:
            Calculator: ASE calculator compatible with the MatGL model.
        """
        import matgl
        from matgl.ext.ase import PESCalculator as PESCalculator_

        model = matgl.load_model(path=path)
        kwargs.setdefault("stress_unit", "eV/A3")
        return PESCalculator_(potential=model, **kwargs)

    @staticmethod
    def load_mtp(filename: str | Path, elements: list, **kwargs: Any) -> Calculator:
        """
        Load the MTP model for use in ASE as a calculator.

        Args:
            filename (str | Path): The file storing parameters of potentials, filename should ends with ".mtp".
            elements (list): The list of elements.
            **kwargs (Any): Additional keyword arguments for the PESCalculator.

        Returns:
            Calculator: ASE calculator compatible with the MTP model.
        """
        from maml.apps.pes import MTPotential

        model = MTPotential.from_config(filename=filename, elements=elements)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_gap(filename: str | Path, **kwargs: Any) -> Calculator:
        """
        Load the GAP model for use in ASE as a calculator.

        Args:
            filename (str | Path): The file storing parameters of potentials, filename should ends with ".xml".
            **kwargs (Any): Additional keyword arguments for the PESCalculator.

        Returns:
            Calculator: ASE calculator compatible with the GAP model.
        """
        from maml.apps.pes import GAPotential

        model = GAPotential.from_config(filename=filename)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_nnp(
        input_filename: str | Path, scaling_filename: str | Path, weights_filenames: list, **kwargs: Any
    ) -> Calculator:
        """
        Load the NNP model for use in ASE as a calculator.

        Args:
                input_filename (str | Path): The file storing the input configuration of
                    Neural Network Potential.
                scaling_filename (str | Path): The file storing scaling info of
                    Neural Network Potential.
                weights_filenames (list | Path): List of files storing weights of each specie in
                    Neural Network Potential.
                **kwargs (Any): Additional keyword arguments for the PESCalculator.

        Returns:
            Calculator: ASE calculator compatible with the NNP model.
        """
        from maml.apps.pes import NNPotential

        model = NNPotential.from_config(
            input_filename=input_filename,
            scaling_filename=scaling_filename,
            weights_filenames=weights_filenames,
        )
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_snap(param_file: str | Path, coeff_file: str | Path, **kwargs: Any) -> Calculator:
        """
        Load the SNAP or qSNAP model for use in ASE as a calculator.

        Args:
            param_file (str | Path): The file storing the configuration of potentials.
            coeff_file (str | Path): The file storing the coefficients of potentials.
            **kwargs (Any): Additional keyword arguments for the PESCalculator.

        Returns:
            Calculator: ASE calculator compatible with the SNAP or qSNAP model.
        """
        from maml.apps.pes import SNAPotential

        model = SNAPotential.from_config(param_file=param_file, coeff_file=coeff_file)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_ace(
        basis_set: str | Path | ACEBBasisSet | ACECTildeBasisSet | BBasisConfiguration, **kwargs: Any
    ) -> Calculator:
        """
        Load the ACE model for use in ASE as a calculator.

        Args:
            basis_set: The specification of ACE potential, could be in following forms:
                ".ace" potential filename
                ".yaml" potential filename
                ACEBBasisSet object
                ACECTildeBasisSet object
                BBasisConfiguration object
            **kwargs (Any): Additional keyword arguments for the PyACECalculator.

        Returns:
            Calculator: ASE calculator compatible with the ACE model.
        """
        from pyace import PyACECalculator

        return PyACECalculator(basis_set=basis_set, **kwargs)

    @staticmethod
    def load_nequip(model_path: str | Path, **kwargs: Any) -> Calculator:
        """
        Load the NequIP model for use in ASE as a calculator.

        Args:
            model_path (str | Path): The file storing the configuration of potentials, filename should ends with ".pth".
            **kwargs (Any): Additional keyword arguments for the PESCalculator.

        Returns:
            Calculator: ASE calculator compatible with the NequIP model.
        """
        from nequip.ase import NequIPCalculator

        return NequIPCalculator.from_deployed_model(model_path=model_path, **kwargs)

    @staticmethod
    def load_deepmd(model_path: str | Path, **kwargs: Any) -> Calculator:
        """
        Loads the custom deep potential model for use in ASE as a calculator.

        Args:
            model_path (str | Path): The file storing the configuration of
                potential, filename should end with ".pth"
            **kwargs (Any): Additional keyword arguments for the PESCalculator.

        Returns:
            Calculator: ASE calculator compatible with the DeepMD model.
        """
        from deepmd.calculator import DP

        return DP(model=model_path, **kwargs)

    @staticmethod
    def load_universal(name: str | Calculator, **kwargs: Any) -> Calculator:  # noqa: C901
        """
        Load the universal model for use in ASE as a calculator.

        Args:
            name (str | Calculator): The name of universal calculator.
            **kwargs (Any): Additional keyword arguments for universal calculator.

        Returns:
            Calculator: ASE calculator compatible with the universal model.
        """
        result: Calculator

        if not isinstance(name, str):  # e.g. already an ase Calculator instance
            result = name

        elif any(name.lower().startswith(m) for m in ("m3gnet", "chgnet", "tensornet", "pbe", "r2scan")):
            import matgl
            from matgl.ext.ase import PESCalculator as PESCalculator_

            name = MODEL_ALIASES.get(name.lower(), name)
            model = matgl.load_model(name)
            kwargs.setdefault("stress_unit", "eV/A3")
            result = PESCalculator_(potential=model, **kwargs)

        elif name.lower() == "mace":
            from mace.calculators import mace_mp

            result = mace_mp(**kwargs)

        elif name.lower() == "sevennet":
            from sevenn.calculator import SevenNetCalculator

            result = SevenNetCalculator(**kwargs)

        elif name.lower() == "grace" or name.lower() == "tensorpotential":
            from tensorpotential.calculator.foundation_models import grace_fm

            kwargs.setdefault("model", "GRACE-2L-OAM")
            result = grace_fm(**kwargs)

        elif name.lower() == "orb":
            from orb_models.forcefield.calculator import ORBCalculator
            from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS

            model = kwargs.pop("model", "orb-v2")
            device = kwargs.get("device", "cpu")

            orbff = ORB_PRETRAINED_MODELS[model](device=device)
            result = ORBCalculator(orbff, **kwargs)

        elif name.lower() == "mattersim":
            from mattersim.forcefield import MatterSimCalculator

            result = MatterSimCalculator(**kwargs)

        elif name.lower() == "fairchem":
            from fairchem.core import OCPCalculator

            # Technically, this supports all models that are in fairchem,
            # not just equiformer.
            kwargs.setdefault("model_name", "EquiformerV2-31M-S2EF-OC20-All+MD")
            kwargs.setdefault("local_cache", "./pretrained_models")
            result = OCPCalculator(**kwargs)

        elif name.lower() == "petmad":
            from pet_mad.calculator import PETMADCalculator

            result = PETMADCalculator(**kwargs)

        elif name.lower().startswith("deepmd"):
            import os
            from deepmd.calculator import DP

            cwd = os.path.abspath(os.path.dirname(__file__))
            model_path = os.path.join(cwd, "../tests/pes/DPA3-LAM-2025.3.14-PES", "2025-03-14-dpa3-openlam.pth")
            model_path = os.path.abspath(model_path)
            kwargs.setdefault("model", model_path)
            result = DP(**kwargs)

        else:
            raise ValueError(f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}")

        return result


@deprecated(PESCalculator, "Use PESCalculator.load_universal instead.")
def get_universal_calculator(name: str | Calculator, **kwargs: Any) -> Calculator:  # noqa: C901
    """Helper method to get some well-known **universal** calculators.
    Imports should be inside if statements to ensure that all models are optional dependencies.
    All calculators must be universal, i.e. encompass a wide swath of the periodic table.
    Though MatCalc can be used with any MLIP, even custom ones, this function is not meant as
        a list of all MLIPs.

    Args:
        name (str): Name of calculator.
        **kwargs: Passthrough to calculator init.

    Raises:
        ValueError: on unrecognized model name.

    Returns:
        Calculator
    """
    result: Calculator

    if not isinstance(name, str):  # e.g. already an ase Calculator instance
        result = name

    elif name.lower().startswith("m3gnet") or name.lower().startswith("tensornet"):
        import matgl
        from matgl.ext.ase import PESCalculator as PESCalculator_

        # M3GNet is shorthand for latest M3GNet based on DIRECT sampling.
        # TensorNet is shorthand for latest TensorNet trained on MatPES.
        name = {"m3gnet": "M3GNet-MP-2021.2.8-DIRECT-PES", "tensornet": "TensorNet-MatPES-PBE-v2025.1-PES"}.get(
            name.lower(), name
        )
        model = matgl.load_model(name)
        kwargs.setdefault("stress_unit", "eV/A3")
        result = PESCalculator_(potential=model, **kwargs)

    elif name.lower() == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        result = CHGNetCalculator(**kwargs)

    elif name.lower() == "mace":
        from mace.calculators import mace_mp

        result = mace_mp(**kwargs)

    elif name.lower() == "sevennet":
        from sevenn.calculator import SevenNetCalculator

        result = SevenNetCalculator(**kwargs)

    elif name.lower() == "grace" or name.lower() == "tensorpotential":
        from tensorpotential.calculator.foundation_models import grace_fm

        kwargs.setdefault("model", "GRACE-2L-OAM")
        result = grace_fm(**kwargs)

    elif name.lower() == "orb":
        from orb_models.forcefield.calculator import ORBCalculator
        from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS

        model = kwargs.pop("model", "orb-v2")
        device = kwargs.get("device", "cpu")

        orbff = ORB_PRETRAINED_MODELS[model](device=device)
        result = ORBCalculator(orbff, **kwargs)

    elif name.lower() == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        result = MatterSimCalculator(**kwargs)

    elif name.lower() == "fairchem":
        from fairchem.core import OCPCalculator

        # Technically, this supports all models that are in fairchem,
        # not just equiformer.
        result = OCPCalculator(**kwargs)

    elif name.lower() == "petmad":
        from pet_mad.calculator import PETMADCalculator

        result = PETMADCalculator(**kwargs)

    elif name.lower().startswith("deepmd"):
        import os

        from deepmd.calculator import DP

        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(cwd, "../tests/pes/DPA3-LAM-2025.3.14-PES", "2025-03-14-dpa3-openlam.pth")
        model_path = os.path.abspath(model_path)
        kwargs.setdefault("model", model_path)
        result = DP(**kwargs)

    else:
        raise ValueError(f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}")

    return result


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
