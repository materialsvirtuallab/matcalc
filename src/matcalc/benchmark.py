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


def run_elasticity_benchmark(
    calculator: PESCalculator,
    model_name: str,
    benchmark_name: str | Path = "mp-elasticity-2025.1.json.gz",
    n_samples: int | None = None,
    seed: int = 42,
    n_jobs: None | int = -1,
) -> pd.DataFrame:
    """
    Runs an elasticity benchmark by calculating the bulk and shear moduli for a given set
    of structures using the provided calculator. The benchmark results include comparisons
    with reference DFT values.

    :param calculator: An instance of a calculator object used for performing calculations.
    :param model_name: The name of the model used for calculations, included in results.
    :param benchmark_name: str | Path
        Path to the benchmark data file. By default, it points to a JSON.gz file
        containing elasticity data for materials. Can also be a string representing
        the file name located in the benchmark data directory.
    :param n_samples: int | None
        Number of random samples to select from the dataset. If None, all entries
        from the dataset are included in the benchmark. Default is None.
    :param seed: int
        Random seed used for selecting samples. Default value is 42.
    :param n_jobs: int
        Number of parallel jobs to use for calculations. Default is -1, which
        uses all available CPUs for parallel processing.

    :return: pandas.DataFrame
        A dataframe containing the benchmark results. Includes reference DFT bulk
        and shear modulus values, calculated properties using the specified model,
        and absolute errors for each property. Additional information such as
        material IDs and formulae is also included.
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

    results = pd.DataFrame(rows)

    elastic_calc = ElasticityCalc(calculator, fmax=0.05, relax_structure=True)

    # We use trivial parallel processing in joblib to speed up the computations.
    properties = list(elastic_calc.calc_many(structures, n_jobs=n_jobs))

    results[f"K_{model_name}"] = [d["bulk_modulus_vrh"] * eVA3ToGPa for d in properties]
    results[f"G_{model_name}"] = [d["shear_modulus_vrh"] * eVA3ToGPa for d in properties]
    results[f"AE K {model_name}"] = np.abs(results[f"K_{model_name}"] - results["K_DFT"])
    results[f"AE G {model_name}"] = np.abs(results[f"G_{model_name}"] - results["G_DFT"])

    return results
