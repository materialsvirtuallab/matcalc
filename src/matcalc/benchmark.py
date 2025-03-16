"""This module implements classes for running benchmarks on materials properties."""

from __future__ import annotations

import abc
import json
import logging
import random
import typing
from pathlib import Path

import pandas as pd
import requests
from monty.serialization import loadfn
from scipy import constants

if typing.TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

    from .base import PropCalc

from .elasticity import ElasticityCalc
from .phonon import PhononCalc

eVA3ToGPa = constants.e / (constants.angstrom) ** 3 / constants.giga  # noqa: N816

BENCHMARK_DATA_URL = "https://api.github.com/repos/materialsvirtuallab/matcalc/contents/benchmark_data"
BENCHMARK_DATA_DOWNLOAD_URL = "https://raw.githubusercontent.com/materialsvirtuallab/matcalc/main/benchmark_data"
BENCHMARK_DATA_DIR = Path.home() / ".local" / "matcalc" / "benchmark_data"
BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def _load_checkpoint(
    checkpoint_file: str | Path | None, all_data: pd.DataFrame, all_structures: list[Structure], index_name: str
) -> tuple[list, list, list]:
    """
    Loads a checkpoint file if it exists and filters the remaining data and structures
    based on the entries already processed. If a checkpoint file is not provided or
    doesn't exist, the function returns all the provided data and structures.

    :param checkpoint_file: Path to the checkpoint file to load processed entries.
    :param all_data: DataFrame containing all input data records.
    :param all_structures: List of structures corresponding to the data records.
    :param index_name: Name of the index field used to identify already processed entries.
    :return: A tuple containing three lists:
             - already processed records from the checkpoint file,
             - remaining data not processed yet,
             - remaining structures corresponding to unprocessed data.
    """
    if checkpoint_file and Path(checkpoint_file).exists():
        already_done = pd.read_csv(checkpoint_file)
        logger.info("Loaded %d entries from %s...", len(already_done), checkpoint_file)
        results = already_done.to_dict("records")
        done_ids = [d[index_name] for d in results]
        data = []
        structures = []
        for i, d in enumerate(all_data.to_dict("records")):
            if d[index_name] not in done_ids:
                data.append(d)
                structures.append(all_structures[i])
        return results, data, structures
    return [], all_data.to_dict("records"), all_structures


def _save_checkpoint(checkpoint_file: str | Path | None, results: list, index_name: str) -> None:
    """
    Saves a list of results as a CSV file at the specified checkpoint location. The function
    takes a list of results, converts it into a pandas DataFrame, sets the specified index
    column, and writes it to the provided file path.

    :param checkpoint_file: The file path where the checkpoint data will be saved. Can be
        a string, a Path object, or None. If None, no file is written.
    :param results: A list of dictionaries or objects to be saved into a CSV file.
    :param index_name: The name of the column to be set as the index in the resulting DataFrame.
    :return: None
    """
    logger.info("Saving %d entries to %s...", len(results), checkpoint_file)
    pd.DataFrame(results).set_index(index_name).to_csv(checkpoint_file)


class Benchmark(metaclass=abc.ABCMeta):
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
        benchmark_name: str | Path,
        properties: list[str],
        index_name: str,
        other_fields: tuple[str] = (),
        suffix_ground_truth: str = "DFT",
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
            "mp-binary-pbe-elasticity-2025.1.json.gz".
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
            row = {k: entry[k] for k in [index_name] + list(other_fields)}
            for prop in properties:
                row[f"{prop}_{suffix_ground_truth}"] = entry[prop]
            rows.append(row)

            structures.append(entry["structure"])

        self.properties = properties
        self.other_fields = other_fields
        self.index_name = index_name
        self.structures = structures
        self.kwargs = kwargs
        self.ground_truth = pd.DataFrame(rows)

    @abc.abstractmethod
    def get_prop_calc(self, calculator: Calculator) -> PropCalc:
        pass

    def process_result(self, result, model_name):
        return {f"{k}_{model_name}": result[k] for k in self.properties}

    def run(
        self,
        calculator: Calculator,
        model_name: str,
        n_jobs: None | int = -1,
        checkpoint_file: str | Path | None = None,
        checkpoint_freq: int = 1000,
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
        results, ground_truth, structures = _load_checkpoint(
            checkpoint_file, self.ground_truth, self.structures, self.index_name
        )

        prop_calc = self.get_prop_calc(calculator, **self.kwargs)
        for i, d in enumerate(prop_calc.calc_many(structures, n_jobs=n_jobs, allow_errors=True, **kwargs)):
            r = ground_truth[i]
            r.update(self.process_result(d, model_name))

            results.append(r)

            if checkpoint_file and (i + 1) % checkpoint_freq == 0:
                _save_checkpoint(checkpoint_file, results, self.index_name)

        return pd.DataFrame(results)


class ElasticityBenchmark(Benchmark):
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
        benchmark_name: str | Path = "mp-binary-pbe-elasticity-2025.1.json.gz",
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
            "mp-binary-pbe-elasticity-2025.1.json.gz".
        :param n_samples: Number of samples to extract randomly from entries. This is useful when you just want to
            run a small number of structures for code testing. If `None`, all entries from the file are used.
            Defaults to `None`.
        :param seed: Random seed used for reproducible sub-sampling of the entry dataset.
            Defaults to 42.
        :param kwargs: Keyword arguments passthrough to the ElasticityCalculator.
        """
        super().__init__(
            benchmark_name,
            properties=("bulk_modulus_vrh", "shear_modulus_vrh"),
            index_name=index_name,
            other_fields=("formula",),
            **kwargs,
        )

    def get_prop_calc(self, calculator: Calculator, **kwargs) -> PropCalc:
        return ElasticityCalc(calculator, **kwargs)

    def process_result(self, result, model_name) -> dict:
        d = {}
        d[f"bulk_modulus_vrh_{model_name}"] = (
            result["bulk_modulus_vrh"] * eVA3ToGPa if result is not None else float("nan")
        )
        d[f"shear_modulus_vrh_{model_name}"] = (
            result["shear_modulus_vrh"] * eVA3ToGPa if result is not None else float("nan")
        )
        return d


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
        benchmark_name: str | Path = "alexandria-binary-pbe-phonon-2025.1.json.gz",
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
            "alexandria-binary-pbe-phonon-2025.1.json.gz".
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

        self.index_name = index_name
        self.structures = structures
        self.kwargs = kwargs
        self.ground_truth = pd.DataFrame(rows)

    def run(
        self,
        calculator: Calculator,
        model_name: str,
        n_jobs: None | int = -1,
        checkpoint_file: str | Path | None = None,
        checkpoint_freq: int = 1000,
        **kwargs,  # noqa:ANN003
    ) -> pd.DataFrame:
        """
        Executes the phonon calculation workflow using the given phonon calculator, model name,
        and various optional parameters. The function supports checkpointing to save intermediate
        results during long-running computations. It calculates heat capacity for a list of
        structures and stores the results in a DataFrame.

        :param calculator: The phonon calculator instance used to perform the calculations.
        :param model_name: The name of the model being used for the calculations. This will be
                           used as a key in the resulting DataFrame.
        :param n_jobs: Number of parallel jobs to run. If None or -1, it uses all available cores.
        :param checkpoint_file: Path for saving checkpoint files. If provided, intermediate
                                results will be saved to this file during computation.
        :param checkpoint_freq: Frequency at which intermediate results are saved to the
                                checkpoint file. After every 'checkpoint_freq' structures, the
                                results are saved.
        :param kwargs: Additional keyword arguments passed to the phonon calculation methods.
        :return: A DataFrame containing the calculated results, including heat capacities for
                 the structures processed.
        """
        results, data, structures = _load_checkpoint(
            checkpoint_file, self.ground_truth, self.structures, self.index_name
        )

        # Initialize the phonon calculator with fixed parameters.
        phonon_calc = PhononCalc(calculator, **self.kwargs)
        for i, d in enumerate(phonon_calc.calc_many(structures, n_jobs=n_jobs, allow_errors=True, **kwargs)):
            r = data[i]
            r[f"CV_{model_name}"] = d["thermal_properties"]["heat_capacity"][30]

            results.append(r)

            if checkpoint_file and (i + 1) % checkpoint_freq == 0:
                _save_checkpoint(checkpoint_file, results, self.index_name)

        return pd.DataFrame(results)


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

    def run(
        self,
        calculators: dict[str, Calculator],
        n_jobs: int | None = -1,
        checkpoint_freq: int = 1000,
    ) -> list[pd.DataFrame]:
        """
        Executes benchmarks using the provided calculators and combines the results into a
        list of dataframes. Each benchmark runs for all models provided by calculators, collecting
        individual results and performing validations during data combination.

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
                    calculator, model_name, n_jobs=n_jobs, checkpoint_file=chkpt_file, checkpoint_freq=checkpoint_freq
                )
                if results:
                    # Remove duplicate DFT columns.
                    todrop = [c for c in result.columns if c in results[0].columns]
                    result = result.drop(todrop, axis=1)
                results.append(result)
            combined = results[0].join(results[1:], validate="one_to_one")
            all_results.append(combined)
        return all_results
