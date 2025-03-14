"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

import functools
from inspect import isclass
from typing import TYPE_CHECKING, Any

import ase.optimize
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from scipy import constants

eVA3ToGPa = constants.e / (constants.angstrom) ** 3 / constants.giga  # noqa:N816
if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from maml.apps.pes._lammps import LMPStaticCalculator
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration


# Listing of supported universal calculators.
UNIVERSAL_CALCULATORS = (
    "M3GNet",
    "M3GNet-MP-2021.2.8-PES",
    "M3GNet-MP-2021.2.8-DIRECT-PES",
    "CHGNet",
    "MACE",
    "SevenNet",
    "TensorNet-MatPES-PBE-v2025.1-PES",
    "TensorNet-MatPES-r2SCAN-v2025.1-PES",
)


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
        stress_weight: float = 1 / eVA3ToGPa,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PESCalculator with a potential from maml.

        Args:
            potential (LMPStaticCalculator): maml.apps.pes._lammps.LMPStaticCalculator
            stress_weight (float): The conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa.
                Default to 1 / 160.21766208.
            **kwargs: Additional keyword arguments passed to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.stress_weight = stress_weight

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
        kwargs.setdefault("stress_weight", 1 / eVA3ToGPa)
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
    def load_universal(name: str | Calculator, **kwargs: Any) -> Calculator:
        """
        Load the universal model for use in ASE as a calculator.

        Args:
            name (str | Calculator): The name of universal calculator.
            **kwargs (Any): Additional keyword arguments for universal calculator.

        Returns:
            Calculator: ASE calculator compatible with the universal model.
        """
        if not isinstance(name, str):  # e.g. already an ase Calculator instance
            return name

        if name.lower().startswith("m3gnet") or name.lower().startswith("tensornet-matpes"):
            import matgl
            from matgl.ext.ase import PESCalculator as PESCalculator_

            # M3GNet is shorthand for latest M3GNet based on DIRECT sampling.
            name = {"m3gnet": "M3GNet-MP-2021.2.8-DIRECT-PES"}.get(name.lower(), name)
            model = matgl.load_model(name)
            kwargs.setdefault("stress_weight", 1 / eVA3ToGPa)
            return PESCalculator_(potential=model, **kwargs)

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


@functools.lru_cache
def get_universal_calculator(name: str | Calculator, **kwargs: Any) -> Calculator:
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
    import warnings

    warnings.warn(
        "get_universal_calculator() will be deprecated in the future. Use PESCalculator.load_YOUR_MODEL() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(name, str):  # e.g. already an ase Calculator instance
        return name

    if name.lower().startswith("m3gnet") or name.lower().startswith("tensornet-matpes"):
        import matgl
        from matgl.ext.ase import PESCalculator as PESCalculator_

        # M3GNet is shorthand for latest M3GNet based on DIRECT sampling.
        name = {"m3gnet": "M3GNet-MP-2021.2.8-DIRECT-PES"}.get(name.lower(), name)
        model = matgl.load_model(name)
        kwargs.setdefault("stress_weight", 1 / eVA3ToGPa)
        return PESCalculator_(potential=model, **kwargs)

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
