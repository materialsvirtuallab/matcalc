"""This module implements classes for running benchmarks on materials properties."""

from __future__ import annotations

import abc
import json
import logging
import random
import typing
from pathlib import Path
from urllib.parse import urlparse

import fsspec
import numpy as np
import pandas as pd
import requests
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import curve_fit

if typing.TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from .base import PropCalc

from .config import BENCHMARK_DATA_DIR, BENCHMARK_DATA_DOWNLOAD_URL, BENCHMARK_DATA_URL
from .elasticity import ElasticityCalc
from .phonon import PhononCalc
from .relaxation import RelaxCalc
from .units import eVA3ToGPa

logger = logging.getLogger(__name__)


def get_available_benchmarks() -> list[str]:
    """Checks Github for available benchmarks for download.

    Returns:
        List of available benchmarks.
    """
    r = requests.get(BENCHMARK_DATA_URL)  # noqa: S113
    return [d["name"] for d in json.loads(r.content.decode("utf-8")) if d["name"].endswith(".json.gz")]


def get_benchmark_data(name: str) -> pd.DataFrame:
    """
    Retrieve benchmark data as a Pandas DataFrame. Uses fsspec to cache files locally if possible.

    :param name: Name of the benchmark elemental_refs file to be retrieved
    :type name: str
    :return: Benchmark elemental_refs loaded as a Pandas DataFrame
    :rtype: pd.DataFrame
    :raises requests.RequestException: If the benchmark elemental_refs file cannot be
        downloaded from the specified URL
    """
    uri = f"{BENCHMARK_DATA_DOWNLOAD_URL}/{name}"
    parsed = urlparse(str(uri))
    fs = fsspec.filesystem(
        "filecache",
        target_protocol=parsed.scheme,
        cache_storage=str(BENCHMARK_DATA_DIR),
        same_names=True,
    )
    with fs.open(uri, compression="infer") as f:
        return json.load(f, cls=MontyDecoder)


class CheckpointFile:
    """
    CheckpointFile class encapsulates functionality to handle checkpoint files for processing elemental_refs.

    The class constructor initializes the CheckpointFile object with the provided path, all elemental_refs to be
    processed, list of structures, and index name for elemental_refs identification.

    load() method loads a checkpoint file if it exists, filtering the remaining elemental_refs and structures based on
    entries already processed. It returns a tuple containing three lists: already processed records, remaining
    elemental_refs, and remaining structures.

    save() method saves a list of results at the specified checkpoint location.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        """
        Represents an initialization process for handling a filesystem path. The
        provided path is converted into a `Path` object for standardized path
        management in the application.

        :param path: The filesystem path to be managed. Can be provided as a
            string or as a `Path` object.
        """
        self.path = Path(path)

    def load(self, *args: list) -> tuple:
        """
        Loads elemental_refs from a specified path if it exists, returning the loaded elemental_refs along with
        remaining portions of the given input arguments.

        The method checks if the file path exists, and if so, it loads elemental_refs from the specified
        file using a predefined `loadfn` function. It logs the number of loaded entries and
        returns the successfully loaded elemental_refs alongside sliced input arguments based on the
        number of loaded entries. If the file path does not exist, it returns empty results
        and the original input arguments unchanged.

        :param args: List of lists where each list corresponds to additional elemental_refs to
            process in conjunction with the loaded file content.
        :return: A tuple where the first element is the loaded elemental_refs (list) from the specified
            file path (or an empty list if the path does not exist), and subsequent elements
            are the remaining unsliced portions of each input list from `args` or the entire
            original lists if nothing was loaded.
        """
        if self.path.exists():
            results = loadfn(self.path)
            logger.info("Loaded %d entries from %s...", len(results), self.path)
            return results, *[a[len(results) :] for a in args]
        return [], *args

    def save(self, results: list[dict[str, typing.Any]]) -> None:
        """
        Saves a list of results at the specified checkpoint location.

        :param results: A list of dictionaries or objects to be saved.
        :return: None
        """
        logger.info("Saving %d entries to %s...", len(results), self.path)
        dumpfn(results, self.path)


class Benchmark(metaclass=abc.ABCMeta):
    """
    Represents an abstract base class for benchmarking elasticity properties of materials.

    This class provides functionality to process benchmark elemental_refs, create a DataFrame for analysis,
    and run calculations using a specified potential energy surface (PES) calculator. It is designed
    to facilitate benchmarking of bulk and shear moduli against pre-defined ground truth elemental_refs.

    :ivar properties: List of properties to extract and benchmark. These properties
        are key inputs for analysis tasks.
    :type properties: list[str]
    :ivar other_fields: Tuple of additional fields in the benchmark entries to
        include in the processed elemental_refs. Useful for metadata or optional attributes.
    :type other_fields: tuple[str]
    :ivar index_name: Name of the index field in the benchmark dataset. This is used
        as the primary key for identifying entries.
    :type index_name: str
    :ivar structures: List of structures extracted from the benchmark entries. Structures
        are objects describing material geometries stored in the dataset.
    :type structures: list[Structure]
    :ivar kwargs: Additional keywords passed through to the ElasticityCalculator or associated
        processes for extended configuration.
    :type kwargs: dict
    :ivar ground_truth: DataFrame containing the processed benchmark elemental_refs, including ground truth
        reference values for materials properties.
    :type ground_truth: pandas.DataFrame
    """

    def __init__(
        self,
        benchmark_name: str | Path,
        properties: typing.Sequence[str],
        index_name: str,
        other_fields: tuple = (),
        property_rename_map: dict[str, str] | None = None,
        suffix_ground_truth: str = "DFT",
        n_samples: int | None = None,
        seed: int = 42,
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes an instance for processing benchmark elemental_refs and constructing a DataFrame
        representing the ground truth properties of input structures. Additionally, stores
        information about input structures and other auxiliary elemental_refs for further usage.

        :param benchmark_name: The name of the benchmark dataset or a path to a file containing
            the benchmark entries.
        :type benchmark_name: str | Path

        :param properties: A list of property names to extract.
        :type properties: list[str]

        :param index_name: The name of the field used as the index for the resulting DataFrame (typically a unique id
            like mp_id).
        :type index_name: str

        :param other_fields: Additional fields to include in the DataFrame, default is an empty tuple. Useful ones
            are for example formula or metadata.
        :type other_fields: tuple[str]

        :param property_rename_map: A dict used to rename the properties for easier reading.
        :type property_rename_map: dict | None

        :param suffix_ground_truth: The suffix added to the property names in the DataFrame for
            distinguishing ground truth values, default is "DFT".
        :type suffix_ground_truth: str

        :param n_samples: Number of samples to randomly select from the benchmark dataset, or
            ``None`` to include all samples, default is ``None``.
        :type n_samples: int | None

        :param seed: Seed value for random sampling of entries (if `n_samples` is specified), default is ``42``.
        :type seed: int

        :param kwargs: Additional keyword arguments for configuring the PropCalc..
        :type kwargs: dict

        :raises FileNotFoundError: If the provided `benchmark_name` is a path that does not exist.

        :raises ValueError: If invalid or incomplete elemental_refs is encountered in the benchmark entries.
        """
        rows = []
        structures = []
        entries = get_benchmark_data(benchmark_name) if isinstance(benchmark_name, str) else loadfn(benchmark_name)
        if n_samples:
            random.seed(seed)
            entries = random.sample(entries, n_samples)

        # We will first create a DataFrame from the required components from the raw elemental_refs.
        # We also create the list of structures in the order of the entries.
        for entry in entries:
            row = {k: entry[k] for k in [index_name, *other_fields]}
            for prop in properties:
                row[f"{prop}_{suffix_ground_truth}"] = entry[prop]
            rows.append(row)

            structures.append(entry["structure"])

        self.properties = properties
        self.other_fields = other_fields
        self.index_name = index_name
        self.structures = structures
        self.kwargs = kwargs
        self.property_rename_map = property_rename_map or {}
        self.ground_truth = rows

    @abc.abstractmethod
    def get_prop_calc(self, calculator: Calculator, **kwargs: typing.Any) -> PropCalc:
        """
        Abstract method to retrieve a property calculation object using the provided calculator and additional
        parameters.
        This method must be implemented by subclasses and will utilize the provided calculator to create a
        `PropCalc` instance, possibly influenced by additional keyword arguments.

        :param calculator: The calculator instance to be used for generating the property calculation.
        :type calculator: Calculator
        :param kwargs: Additional keyword arguments that can influence the property calculation process.
        :type kwargs: dict
        :return: An instance of `PropCalc` representing the property calculation result.
        :rtype: PropCalc
        """

    @abc.abstractmethod
    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Implements post-processing of results. A default implementation is provided that simply appends the model name
        as a suffix to the key of the input dictionary for all properties. Subclasses can override this method to
        provide more sophisticated processing.

        :param result: Input dictionary containing key-value pairs to be processed.
        :type result: dict
        :param model_name: The name of the model to append to each key as a suffix.
        :type model_name: str
        :return: A new dictionary with modified keys based on the model name suffix.
        :rtype: dict
        """
        return (
            {f"{k}_{model_name}": result[k] for k in self.properties}
            if result is not None
            else {f"{k}_{model_name}": None for k in self.properties}
        )

    def run(
        self,
        calculator: Calculator,
        model_name: str,
        *,
        n_jobs: None | int = -1,
        checkpoint_file: str | Path | None = None,
        checkpoint_freq: int = 1000,
        include_full_results: bool = False,
        **kwargs,  # noqa:ANN003
    ) -> pd.DataFrame:
        """
        Processes a collection of structures using a calculator, saves intermittent
        checkpoints, and returns the results in a DataFrame. This function supports
        parallel computation and allows for error tolerance during processing.

        The function also retrieves a property calculator and utilizes it to calculate
        desired results for the given set of structures. Checkpoints are saved
        periodically based on the specified frequency, ensuring that progress is not
        lost in case of interruptions.

        :param calculator: ASE-compatible calculator instance used to provide PES information for PropCalc.
        :type calculator: Calculator
        :param model_name: Name of the model used for properties' calculation.
            This name is updated in the results DataFrame.
        :type model_name: str
        :param n_jobs: Number of parallel jobs to be used in the computation. Use -1
            to allocate all cores available on the system. Defaults to -1.
        :type n_jobs: int | None
        :param checkpoint_file: File path where checkpoint elemental_refs is saved periodically.
            If None, no checkpoints are saved.
        :type checkpoint_file: str | Path | None
        :param checkpoint_freq: Frequency after which checkpoint elemental_refs is saved.
            Corresponds to the number of structures processed.
        :type checkpoint_freq: int
        :param include_full_results: Whether to save full results from PropCalc.calc for analysis afterwards. For
            instance, the ElasticityProp does not just compute the bulk and shear moduli, but also the full elastic
            tensors, which can be used for other kinds of analysis. Defaults to False.
        :type include_full_results: bool
        :param kwargs: Additional keyword arguments passed to the property calculator,
            for instance, to customize its behavior or computation options.
        :type kwargs: dict
        :return: A pandas DataFrame containing the processed results for the given
            input structures. The DataFrame includes updated results and relevant
            metrics.
        :rtype: pd.DataFrame
        """
        checkpoint = None
        if checkpoint_file:
            checkpoint = CheckpointFile(checkpoint_file)
            results, ground_truth, structures = checkpoint.load(self.ground_truth, self.structures)
        else:
            results = []
            ground_truth = self.ground_truth
            structures = self.structures

        prop_calc = self.get_prop_calc(calculator, **self.kwargs)
        # We make sure of the generator from prop_calc.calc_many to do this in a memory efficient manner.
        # allow_errors typically should be true since some of the calculations may fail.
        for r, d in zip(
            ground_truth,
            prop_calc.calc_many(structures, n_jobs=n_jobs, allow_errors=True, **kwargs),
        ):
            r.update(self.process_result(d, model_name))
            if include_full_results and d is not None:
                r.update({k: v for k, v in d.items() if k not in self.properties})

            results.append(r)
            if checkpoint and len(results) % checkpoint_freq == 0:
                checkpoint.save(results)

        results_df = pd.DataFrame(results)
        if self.property_rename_map:

            def _rename_property(col: str) -> str:
                for k, v in self.property_rename_map.items():
                    col = col.replace(k, v)
                return col

            results_df = results_df.rename(columns=_rename_property)
        return results_df


class RelaxationBenchmark(Benchmark):
    """
    Represents a benchmark for evaluating and analyzing relaxation properties of materials.
    This benchmark utilizes a dataset and provides functionality for property calculation
    and result processing. The class is designed to work with a predefined framework for
    benchmarking relaxation properties. The benchmark dataset contains data such as relaxed
    structures along with additional metadata. This class supports configurability through
    metadata files, index names, and additional benchmark properties. It relies on external
    calculators and utility classes for property computations and result handling.
    """

    def __init__(
        self,
        index_name: str = "material_id",
        benchmark_name: str | Path = "wbm-random-pbe54-equilibrium-2025.1.json.gz",
        folder_name: str = "default_folder",
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes the RelaxationBenchmark instance with specified benchmark metadata and
        configuration parameters. It sets up the benchmark with the necessary properties
        required for relaxation analysis.

        :param index_name: The name of the index used to uniquely identify records in the dataset.
        :type index_name: str
        :param benchmark_name: The path or name of the benchmark file that contains the dataset.
        :type benchmark_name: str | Path
        :param folder_name: The folder name used for file operations related to structure files.
        :type folder_name: str
        :param kwargs: Additional keyword arguments for customization.
        :type kwargs: dict
        """
        self.folder_name = folder_name
        # Define the expected property from the relaxation calculator.
        kwargs.setdefault("properties", ("structure",))
        # Other fields such as the material formula may be included.
        kwargs.setdefault("other_fields", ("formula",))
        super().__init__(benchmark_name, index_name=index_name, **kwargs)

    def get_prop_calc(self, calculator: Calculator, **kwargs: typing.Any) -> PropCalc:
        """
        Returns a property calculation object for performing relaxation calculations.
        This method initializes the relaxation calculator using the provided Calculator
        object and any additional configuration parameters.

        :param calculator: A Calculator object responsible for performing the relaxation calculation.
        :type calculator: Calculator
        :param kwargs: Additional keyword arguments used for configuration.
        :type kwargs: dict
        :return: An initialized PropCalc object configured for relaxation calculations.
        :rtype: PropCalc
        """
        return RelaxCalc(calculator, **kwargs)

    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Processes the result dictionary containing final relaxed structures, formats the keys
        according to the provided model name. If the result is None, default values of
        NaN are returned for final structures.

        :param result:
            A dictionary containing the final relaxed structures under the keys
            'final_structure'. It can also be None to indicate missing elemental_refs.
        :type result: dict or None

        :param model_name:
            A string representing the identifier or name of the model. It will be used
            to format the returned dictionary's keys.
        :type model_name: str

        :return:
            A dictionary containing the specific final relaxed structure prefixed by the model name.
            The values will be NaN if the input result is None.

        :rtype: dict
        """
        return {
            f"structure_{model_name}": (result["final_structure"] if result is not None else float("nan")),
        }


class ElasticityBenchmark(Benchmark):
    """
    Represents a benchmark for evaluating and analyzing mechanical properties such as
    bulk modulus and shear modulus for various materials. The benchmark primarily utilizes
    a dataset and provides functionality for property calculation and result processing.

    The class is designed to work with a predefined framework for benchmarking mechanical
    properties. The benchmark dataset contains values such as bulk modulus and shear
    modulus along with additional metadata. This class supports configurability through
    metadata files, index names, and additional benchmark properties. It relies on
    external calculators and utility classes for property computations and result handling.
    """

    def __init__(
        self,
        index_name: str = "mp_id",
        benchmark_name: str | Path = "mp-binary-pbe-elasticity-2025.1.json.gz",
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes the ElasticityBenchmark instance by taking benchmark metadata and
        additional configuration parameters. Sets up the benchmark framework with
        specified mechanical properties and metadata.

        :param index_name: The name of the index used to uniquely identify records
                           in the benchmark dataset.
        :type index_name: str
        :param benchmark_name: The path or name of the benchmark file that contains
                               the dataset. Can either be a string or a Path object.
        :type benchmark_name: str | Path
        :param kwargs: Additional keyword arguments that may be passed to parent
                       class methods or used for customization.
        :type kwargs: dict
        """
        kwargs.setdefault("properties", ("bulk_modulus_vrh", "shear_modulus_vrh"))
        kwargs.setdefault("property_rename_map", {"bulk_modulus": "K", "shear_modulus": "G"})
        kwargs.setdefault("other_fields", ("formula",))
        super().__init__(
            benchmark_name,
            index_name=index_name,
            **kwargs,
        )

    def get_prop_calc(self, calculator: Calculator, **kwargs: typing.Any) -> PropCalc:
        """
        Calculates and returns a property calculation object based on the provided
        calculator and optional parameters. This is useful for initializing and
        configuring a property calculation.

        :param calculator: A Calculator object responsible for performing numerical
           operations required for property calculations.
        :param kwargs: Additional keyword arguments used for configuring the property
           calculation.
        :return: An initialized `PropCalc` object configured based on the specified
           calculator and keyword arguments.
        :rtype: PropCalc
        """
        return ElasticityCalc(calculator, **kwargs)

    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Processes the result dictionary containing bulk and shear modulus values, adjusts
        them by multiplying with a predefined conversion factor, and formats the keys
        according to the provided model name. If the result is None, default values of
        NaN are returned for both bulk and shear modulus.

        :param result:
            A dictionary containing the bulk and shear modulus values under the keys
            'bulk_modulus_vrh' and 'shear_modulus_vrh' respectively. It can also be
            None to indicate missing elemental_refs.
        :type result: dict or None

        :param model_name:
            A string representing the identifier or name of the model. It will be used
            to format the returned dictionary's keys.
        :type model_name: str

        :return:
            A dictionary containing two entries. The keys will be dynamically created
            by appending the model name to the terms 'bulk_modulus_vrh_' and
            'shear_modulus_vrh_'. The values will either be scaled modulus values or
            NaN if the input result is None.
        :rtype: dict
        """
        return {
            f"bulk_modulus_vrh_{model_name}": (
                result["bulk_modulus_vrh"] * eVA3ToGPa if result is not None else float("nan")
            ),
            f"shear_modulus_vrh_{model_name}": (
                result["shear_modulus_vrh"] * eVA3ToGPa if result is not None else float("nan")
            ),
        }


class PhononBenchmark(Benchmark):
    """
    Manages phonon benchmarking tasks, such as initializing benchmark elemental_refs,
    performing calculations, and processing results.

    This class facilitates constructing and managing phonon benchmarks based on
    provided elemental_refs. It supports operations for processing benchmark elemental_refs,
    extracting relevant attributes, and computing thermal properties. It is
    compatible with various calculators and is designed to streamline the
    benchmarking process for materials' phonon-related properties.
    """

    def __init__(
        self,
        index_name: str = "mp_id",
        benchmark_name: str | Path = "alexandria-binary-pbe-phonon-2025.1.json.gz",
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes an instance with specified index and benchmark details.

        This constructor sets up an object with predefined properties such as heat
        capacity and additional fields such as the formula. It supports customizations
        via keyword arguments for further configurations.

        :param index_name: The name of the index to be used for identification in
            the dataset.
        :param benchmark_name: The benchmark file name or path containing
            the dataset information in JSON or compressed format.
        :param kwargs: Additional optional parameters for configuration.
        """
        kwargs.setdefault("properties", ("heat_capacity",))
        kwargs.setdefault("property_rename_map", {"heat_capacity": "CV"})
        kwargs.setdefault("other_fields", ("formula",))
        super().__init__(
            benchmark_name,
            index_name=index_name,
            **kwargs,
        )

    def get_prop_calc(self, calculator: Calculator, **kwargs: typing.Any) -> PropCalc:
        """
        Retrieves a phonon calculation instance based on the given calculator and
        additional keyword arguments.

        This function initializes and returns a `PhononCalc` object using the provided
        calculator instance and any optional keyword arguments to configure the
        calculation further.

        :param calculator: The calculator instance used to perform the phonon
            calculation. Must be an instance of the `Calculator` class.
        :param kwargs: Additional keyword arguments for configuring the resulting
            `PhononCalc` instance.
        :return: A new `PhononCalc` object, initialized with the input calculator and
            optional parameters.
        :rtype: PropCalc
        """
        return PhononCalc(calculator, **kwargs)

    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Processes the result dictionary to extract specific thermal property information for the provided model name.

        :param result: Dictionary containing thermal properties, with keys structured to access relevant elemental_refs
            like "thermal_properties" and "heat_capacity".
        :type result: dict
        :param model_name: The model name used as a prefix in returned result keys.
        :type model_name: str
        :return: A dictionary containing the specific heat capacity at a particular index (e.g., 30),
            prefixed by the model name.
        :rtype: dict
        """
        return {
            f"heat_capacity_{model_name}": (
                result["thermal_properties"]["heat_capacity"][30] if result is not None else float("nan")
            )
        }


class SofteningBenchmark:
    """
    A benchmark for the systematic softening of a PES, as described in:
        B. Deng, et al. npj Comput. Mater. 11, 9 (2025).
        doi: 10.1038/s41524-024-01500-6
    The dataset used here can be found in figshare through:
        https://figshare.com/articles/dataset/WBM_high_energy_states/27307776?file=50005317
    This benchmark essentially performs static calculation on pre-sampled high-energy
    PES configurations, and then compare the systematic underestimation of forces
    predicted between GGA-DFT and the provided force field.
    """

    def __init__(
        self,
        benchmark_name: str | Path = "wbm-high-energy-states.json.gz",
        index_name: str = "wbm_id",
        n_samples: int | None = None,
        seed: int = 42,
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes an instance with specified index and benchmark details.

        :param index_name: The name of the index to be used for identification in
            the dataset.
        :param benchmark_name: The benchmark file name or path containing
            the dataset information in JSON or compressed format.
        :param kwargs: Additional optional parameters for configuration.
        """
        self.index_name = index_name
        self.data = get_benchmark_data(benchmark_name) if isinstance(benchmark_name, str) else loadfn(benchmark_name)
        if n_samples:
            random.seed(seed)
            self.material_ids = random.sample(list(self.data.keys()), n_samples)
        else:
            self.material_ids = list(self.data.keys())
        self.kwargs = kwargs

    @staticmethod
    def get_linear_fitted_slope(x: list, y: list) -> float:
        """
        Return the linearly fitted slope of x and y using a simple linear model (y = ax).
        :param x: A list of the x values.
        :param y: A list of the y values.
        :return: A float of the fitted slope.
        """
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        def linear_model(x: float, a: float) -> float:
            return a * x

        popt, _ = curve_fit(linear_model, x, y)
        return popt[0]

    def run(
        self,
        calculator: Calculator,
        model_name: str,
        checkpoint_file: str | Path | None = None,
        checkpoint_freq: int = 10,
        *,
        include_full_results: bool = False,
    ) -> pd.DataFrame:
        """
        Process all the material ids by
        1. calculate the forces on all the sampled structures.
        2. perform a linear fit on the predicted forces w.r.t. provided DFT forces.
        3. returning the fitted slopes as the softening scales.

        :param calculator: The ASE-compatible calculator instance
        :type calculator: Calculator
        :param model_name: Name of the model used for properties' calculation.
            This name is updated in the results DataFrame.
        :type model_name: str
        :param checkpoint_file: File path where checkpoint is saved periodically.
            If None, no checkpoints are saved.
        :type checkpoint_file: str | Path | None
        :param checkpoint_freq: Frequency after which checkpoint is saved.
            Corresponds to the number of structures processed.
        :type checkpoint_freq: int
        :param include_full_results: Whether to include the raw force prediction in the
            returned dataframe
        :type include_full_results: bool

        :return: A dataframe containing the softening scales.
        :rtype: pd.DataFrame
        """
        checkpoint = None
        if checkpoint_file:
            checkpoint = CheckpointFile(checkpoint_file)
            results, material_ids = checkpoint.load(self.material_ids)
        else:
            results = []
            material_ids = self.material_ids

        for material_id in material_ids:
            frames = self.data[material_id]
            # Check whether the calculator supports all the elements in this structure
            support = True
            test_structure = next(iter(frames.values()))["structure"]
            for elem in test_structure.composition.elements:
                if str(elem) not in calculator.element_types:
                    support = False
            if not support:
                # skipping this structure
                continue

            force_ground_truth, force_prediction = [], []
            for frame in frames.values():
                atoms = AseAtomsAdaptor.get_atoms(frame["structure"])
                atoms.calc = calculator
                forces = atoms.get_forces().tolist()
                force_prediction.append(forces)
                force_ground_truth.append(frame["vasp_f"])

            softening_scale = self.get_linear_fitted_slope(force_ground_truth, force_prediction)

            results.append(
                {
                    "material_id": material_id,
                    "formula": test_structure.composition.reduced_formula,
                    f"softening_scale_{model_name}": softening_scale,
                    "raw_force_predictions": force_prediction,
                }
            )
            if checkpoint and (len(results) % checkpoint_freq == 0 or material_id == self.material_ids[-1]):
                checkpoint.save(results)

        if include_full_results:
            results_df = pd.DataFrame(results)
        else:
            results_df = pd.DataFrame(
                [{k: v for k, v in res.items() if k != "raw_force_predictions"} for res in results]
            )
        return results_df


class BenchmarkSuite:
    """A class to run multiple benchmarks in a single run."""

    def __init__(self, benchmarks: list) -> None:
        """
        Represents a configuration for handling a list of benchmarks. This class is designed
        to initialize with a specified list of benchmarks.

        Attributes:
            benchmarks (list): A list containing benchmark configurations or elemental_refs for
            evaluation.

        :param benchmarks: A list of benchmarks for configuration or evaluation.
        :type benchmarks: list
        """
        self.benchmarks = benchmarks

    def run(
        self,
        calculators: dict[str, Calculator],
        n_jobs: int | None = -1,
        checkpoint_freq: int = 1000,
    ) -> list[pd.DataFrame]:
        """
        Executes benchmarks using the provided calculators and combines the results into a
        list of dataframes. Each benchmark runs for all models provided by calculators, collecting
        individual results and performing validations during elemental_refs combination.

        :param calculators: A dictionary where the keys are the model names (str)
            and the values are the corresponding calculator instances (Calculator).
        :param n_jobs: The maximum number of concurrent jobs to run. If set to -1,
            utilizes all available processors. Defaults to -1.
        :param checkpoint_freq: The frequency at which progress is saved as checkpoints,
            in terms of calculation steps. Defaults to 1000.
        :return: A list of pandas DataFrames, each containing combined results
            for all calculators across the benchmarks.
        """
        all_results = []
        for benchmark in self.benchmarks:
            results: list[pd.DataFrame] = []
            for model_name, calculator in calculators.items():
                chkpt_file = f"{benchmark.__class__.__name__}_{model_name}.csv"
                result = benchmark.run(
                    calculator,
                    model_name,
                    n_jobs=n_jobs,
                    checkpoint_file=chkpt_file,
                    checkpoint_freq=checkpoint_freq,
                )
                if results:
                    # Remove duplicate DFT columns.
                    todrop = [c for c in result.columns if c in results[0].columns]
                    result = result.drop(todrop, axis=1)
                results.append(result)
            combined = results[0].join(results[1:], validate="one_to_one")
            all_results.append(combined)
        return all_results
