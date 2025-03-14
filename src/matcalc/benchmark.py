"""This module contains functions for running benchmarks on materials properties."""

from __future__ import annotations

import random
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from monty.serialization import loadfn
from scipy import constants

if typing.TYPE_CHECKING:
    from matcalc.utils import PESCalculator

from .elasticity import ElasticityCalc

eVA3ToGPa = constants.e / (constants.angstrom) ** 3 / constants.giga  # noqa: N816

BENCHMARK_DATA_DIR = Path(__file__).parent / ".." / ".." / "benchmark_data"


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
        benchmark_name: str | Path = "mp-elasticity-2025.1.json.gz",
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
            "mp-elasticity-2025.1.json.gz".
        :param n_samples: Number of samples to extract randomly from entries. If `None`,
            all entries from the file are used. Defaults to `None`.
        :param seed: Random seed used for reproducible sub-sampling of the entry dataset.
            Defaults to 42.
        :param kwargs: Keyword arguments passthrough to the ElasticityCalculator.
        """
        rows = []
        structures = []
        if isinstance(benchmark_name, str):
            benchmark_name = Path(BENCHMARK_DATA_DIR / benchmark_name)

        entries = loadfn(benchmark_name)

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
        self.ground_truth = pd.DataFrame(rows)

    def run(
        self,
        calculator: PESCalculator,
        model_name: str,
        n_jobs: None | int = -1,
    ) -> pd.DataFrame:
        """
        Runs the elasticity benchmark for a given potential energy surface (PES)
        calculator and model name. The benchmark computes bulk and shear moduli,
        and evaluates absolute error (AE) with respect to the ground truth data
        for each modulus.

        :param calculator: Instance of PESCalculator used for calculation.
        :type calculator: PESCalculator
        :param model_name: The name of the model being benchmarked.
        :type model_name: str
        :param n_jobs: Number of parallel jobs to execute for elasticity calculation. Since benchmarking is typically
            done on a large number of structures, the default is set to -1, which uses all available processors.
        :type n_jobs: None | int
        :return: DataFrame containing calculated properties, including bulk modulus
            and shear modulus for the given model, as well as their absolute
            errors compared to ground truth data.
        :rtype: pandas.DataFrame
        """
        results = self.ground_truth.copy()

        elastic_calc = ElasticityCalc(calculator, **self.kwargs)

        # We use trivial parallel processing in joblib to speed up the computations.
        properties = list(elastic_calc.calc_many(self.structures, n_jobs=n_jobs))

        results[f"K_{model_name}"] = [d["bulk_modulus_vrh"] * eVA3ToGPa for d in properties]
        results[f"G_{model_name}"] = [d["shear_modulus_vrh"] * eVA3ToGPa for d in properties]
        results[f"AE K {model_name}"] = np.abs(results[f"K_{model_name}"] - results["K_DFT"])
        results[f"AE G {model_name}"] = np.abs(results[f"G_{model_name}"] - results["G_DFT"])

        return results
