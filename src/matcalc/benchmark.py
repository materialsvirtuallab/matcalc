"""This module contains functions for running benchmarks on materials properties."""

from __future__ import annotations

import abc
import json
import random
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from monty.serialization import loadfn
from scipy import constants

if typing.TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

from .elasticity import ElasticityCalc
from .phonon import PhononCalc

eVA3ToGPa = constants.e / (constants.angstrom) ** 3 / constants.giga  # noqa: N816

BENCHMARK_DATA_URL = "https://api.github.com/repos/materialsvirtuallab/matcalc/contents/benchmark_data"
BENCHMARK_DATA_DOWNLOAD_URL = "https://raw.githubusercontent.com/materialsvirtuallab/matcalc/main/benchmark_data"
BENCHMARK_DATA_DIR = Path.home() / ".local" / "matcalc" / "benchmark_data"
BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_available_benchmarks() -> list[str]:
    """Checks Github for available benchmarks for download.

    Returns:
        List of available benchmarks.
    """
    r = requests.get(BENCHMARK_DATA_URL)  # noqa: S113
    return [d["name"] for d in json.loads(r.content.decode("utf-8")) if d["name"].endswith(".json.gz")]


def get_benchmark_data(name: str) -> pd.DataFrame:
    """
    Retrieve benchmark data as a Pandas DataFrame by downloading it if not already
    available locally.

    The function checks if the specified benchmark data file exists in the
    `BENCHMARK_DATA_DIR` directory. If the file does not exist, it attempts to
    download the data from a predefined URL using the benchmark name. In the case
    of a successful download, the file is saved locally. If the download fails,
    a RequestException is raised. Upon successful retrieval or download of the
    benchmark file, the data is read and returned as a Pandas DataFrame.

    :param name: Name of the benchmark data file to be retrieved
    :type name: str
    :return: Benchmark data loaded as a Pandas DataFrame
    :rtype: pd.DataFrame
    :raises requests.RequestException: If the benchmark data file cannot be
        downloaded from the specified URL
    """
    if not (BENCHMARK_DATA_DIR / name).exists():
        uri = f"{BENCHMARK_DATA_DOWNLOAD_URL}/{name}"
        r = requests.get(uri)  # noqa: S113
        if r.status_code == 200:  # noqa: PLR2004
            with open(BENCHMARK_DATA_DIR / name, "wb") as f:
                f.write(r.content)
        else:
            raise requests.RequestException(f"Bad uri: {uri}")
    return loadfn(BENCHMARK_DATA_DIR / name)


class Benchmark(metaclass=abc.ABCMeta):
    """
    Defines an abstract base class for benchmarking implementations.

    This class serves as a blueprint for creating benchmarking tools
    that operate with a predictive engagement system calculator
    and model names. Subclasses must implement the abstract
    `run` method, which is responsible for executing the benchmarking
    logic.
    """

    @abc.abstractmethod
    def run(
        self,
        calculator: Calculator,
        model_name: str,
        n_jobs: None | int = -1,
        **kwargs,  # noqa:ANN003
    ) -> pd.DataFrame:
        """
        Runs the primary execution logic for the Calculator instance, allowing
        for computation using a specific model and optional parallelization.

        :param calculator: The Calculator instance that performs the computations.
        :param model_name: The name of the model to be used during computation.
        :param n_jobs: The number of jobs for parallel computation. If None or omitted,
            defaults to -1, which signifies using all available processors.
        :return: A pandas DataFrame containing the results of the computations.
        :param kwargs: Keyword arguments passthrough to the ElasticityCalculator.
        """


class ElasticityBenchmark:
    """
    A benchmarking class to process elasticity data and evaluate potential energy
    surface models. The class initializes with benchmark data, handles sub-sampling
    for analytical purposes, and computes elasticity properties for evaluation.

    :ivar kwargs: Additional parameters passed for customization.
    :type kwargs: dict
    :ivar _ground_truth: DataFrame holding the ground truth elastic properties
        extracted from the benchmark dataset.
    :type _ground_truth: pandas.DataFrame
    """

    def __init__(
        self,
        index_name: str = "mp_id",
        benchmark_name: str | Path = "mp-binary-elasticity-2025.1.json.gz",
        n_samples: int | None = None,
        seed: int = 42,
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes the object by processing benchmark data and creating a DataFrame
        containing extracted data for further analysis. This includes creating an
        entry list, extracting required fields from benchmark data, and organizing
        associated structures. The initialization also supports sampling a subset
        of entries with an optional random seed.

        :param benchmark_name: Name or path of the benchmark file. It is either a string
            or a ``Path`` object depending on the data storage directory. Defaults to
            "mp-binary-elasticity-2025.1.json.gz".
        :param n_samples: Number of samples to extract randomly from entries. This is useful when you just want to
            run a small number of structures for code testing. If `None`, all entries from the file are used.
            Defaults to `None`.
        :param seed: Random seed used for reproducible sub-sampling of the entry dataset.
            Defaults to 42.
        :param kwargs: Keyword arguments passthrough to the ElasticityCalculator.
        """
        rows = []
        structures = []
        entries = get_benchmark_data(benchmark_name) if isinstance(benchmark_name, str) else loadfn(benchmark_name)
        if n_samples:
            random.seed(seed)
            entries = random.sample(entries, n_samples)

        # We will first create a DataFrame from the required components from the raw data.
        # We also create the list of structures in the order of the entries.
        for entry in entries:
            rows.append(
                {
                    "mp_id": entry["mp_id"],
                    "formula": entry["formula"],
                    "K_DFT": entry["bulk_modulus_vrh"],
                    "G_DFT": entry["shear_modulus_vrh"],
                }
            )
            structures.append(entry["structure"])

        self.structures = structures
        self.kwargs = kwargs
        self.ground_truth = pd.DataFrame(rows).set_index(index_name)

    def run(
        self,
        calculator: Calculator,
        model_name: str,
        n_jobs: None | int = -1,
        **kwargs,  # noqa:ANN003
    ) -> pd.DataFrame:
        """
        Runs the elasticity benchmark for a given potential energy surface (PES)
        calculator and model name. The benchmark computes bulk and shear moduli,
        and evaluates absolute error (AE) with respect to the ground truth data
        for each modulus.

        :param calculator: Instance of Calculator used for calculation.
        :type calculator: Calculator
        :param model_name: The name of the model being benchmarked.
        :type model_name: str
        :param n_jobs: Number of parallel jobs to execute for elasticity calculation. Since benchmarking is typically
            done on a large number of structures, the default is set to -1, which uses all available processors.
        :type n_jobs: None | int
        :return: DataFrame containing calculated properties, including bulk modulus
            and shear modulus for the given model, as well as their absolute
            errors compared to ground truth data.
        :rtype: pandas.DataFrame
        :param kwargs: Keyword arguments passthrough to the calc_many.
        """
        results = self.ground_truth.copy()

        elastic_calc = ElasticityCalc(calculator, **self.kwargs)

        # We use trivial parallel processing in joblib to speed up the computations.
        properties = list(elastic_calc.calc_many(self.structures, n_jobs=n_jobs, **kwargs))

        results[f"K_{model_name}"] = [d["bulk_modulus_vrh"] * eVA3ToGPa for d in properties]
        results[f"G_{model_name}"] = [d["shear_modulus_vrh"] * eVA3ToGPa for d in properties]
        results[f"AE K_{model_name}"] = np.abs(results[f"K_{model_name}"] - results["K_DFT"])
        results[f"AE G_{model_name}"] = np.abs(results[f"G_{model_name}"] - results["G_DFT"])

        return results


class PhononBenchmark:
    """
    A benchmarking class to process constant-volume heat capacity (CV) data from phonon calculations and evaluate
    potential energy surface models. The class initializes with benchmark data, handles sub-sampling for analytical
    purposes, and computes phonon properties for evaluation.

    :ivar kwargs: Additional parameters passed for customization.
    :type kwargs: dict
    :ivar _ground_truth: DataFrame holding the ground truth phonon properties extracted from the benchmark dataset.
    :type _ground_truth: pandas.DataFrame
    """

    def __init__(
        self,
        index_name: str = "mp_id",
        benchmark_name: str | Path = "alexandria-binary-phonon-2025.1.json.gz",
        n_samples: int | None = None,
        seed: int = 42,
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes the object by processing benchmark data and creating a DataFrame containing extracted data
        for further analysis. This includes creating an entry list, extracting required fields from benchmark data,
        and organizing associated structures. The initialization also supports sampling a subset of entries with an
        optional random seed.

        :param benchmark_name: Name or path of the benchmark file. Defaults to
            "alexandria-binary-phonon-2025.1.json.gz".
        :param n_samples: Number of samples to extract randomly from entries. If `None`, all entries from the file
            are used.
        :param seed: Random seed used for reproducible sub-sampling of the entry dataset.
        :param kwargs: Additional keyword arguments passed through to the PhononCalc.
        """
        rows = []
        structures = []
        # Load the benchmark data from a file or a given path object.
        entries = get_benchmark_data(benchmark_name) if isinstance(benchmark_name, str) else loadfn(benchmark_name)
        if n_samples:
            random.seed(seed)
            entries = random.sample(entries, n_samples)

        # Build the DataFrame rows and store the corresponding structures.
        for entry in entries:
            rows.append(
                {
                    "mp_id": entry["mp_id"],
                    "formula": entry["formula"],
                    "CV_DFT": entry["heat_capacity"],
                }
            )
            structures.append(entry["structure"])

        self.structures = structures
        self.kwargs = kwargs
        self.ground_truth = pd.DataFrame(rows).set_index(index_name)

    def run(
        self,
        calculator: Calculator,
        model_name: str,
        n_jobs: None | int = -1,
        **kwargs,  # noqa:ANN003
    ) -> pd.DataFrame:
        """
        Runs the phonon benchmark for a given potential energy surface (PES) calculator and model name.
        The benchmarks compute constant-volume heat capacity (CV) for each structure, and evaluates the
        absolute error (AE) with respect to the ground truth data.

        :param calculator: Instance of Calculator used for calculation.
        :type calculator: Calculator
        :param model_name: The name of the model being benchmarked.
        :type model_name: str
        :param n_jobs: Number of parallel jobs to execute for phonon calculations. Defaults to -1, which uses
            all available processors.
        :type n_jobs: None | int
        :return: DataFrame containing the computed phonon properties for the given model, along with the absolute
            errors compared to the ground truth data.
        :rtype: pandas.DataFrame
        :param kwargs: Keyword arguments passthrough to the ElasticityCalculator.
        """
        results = self.ground_truth.copy()

        # Initialize the phonon calculator with fixed parameters.
        phonon_calc = PhononCalc(calculator, **self.kwargs)

        # Compute the phonon property for all structures using parallel processing.
        properties = list(phonon_calc.calc_many(self.structures, n_jobs=n_jobs, **kwargs))

        results[f"CV_{model_name}"] = [d["thermal_properties"]["heat_capacity"][30] for d in properties]
        results[f"AE CV_{model_name}"] = np.abs(results[f"CV_{model_name}"] - results["CV_DFT"])

        return results


class BenchmarkSuite:
    """A class to run multiple benchmarks in a single run."""

    def __init__(self, benchmarks: list) -> None:
        """
        Represents a configuration for handling a list of benchmarks. This class is designed
        to initialize with a specified list of benchmarks.

        Attributes:
            benchmarks (list): A list containing benchmark configurations or data for
            evaluation.

        :param benchmarks: A list of benchmarks for configuration or evaluation.
        :type benchmarks: list
        """
        self.benchmarks = benchmarks

    def run(self, calculators: dict[str, Calculator], n_jobs: int | None = -1) -> list[pd.DataFrame]:
        """
        Executes the `run` method for each benchmark using the provided PES calculators and the number
        of jobs for parallel processing. The method manages multiple calculations for each benchmark and
        consolidates their results.

        :param calculators: A dictionary where keys are model names as strings and values
            are instances of `Calculator`.
        :param n_jobs: Number of parallel jobs. Defaults to -1 which typically means using
            all available processors.
        :return: A list of pandas DataFrame objects, each DataFrame representing the consolidated
            results of all models for a particular benchmark.
        """
        all_results = []
        for benchmark in self.benchmarks:
            results: list[pd.DataFrame] = []
            for model_name, calculator in calculators.items():
                result = benchmark.run(calculator, model_name, n_jobs=n_jobs)
                if results:
                    # Remove duplicate DFT columns.
                    todrop = [c for c in result.columns if c in results[0].columns]
                    result = result.drop(todrop, axis=1)
                results.append(result)
            combined = results[0].join(results[1:], validate="one_to_one")
            all_results.append(combined)
        return all_results
