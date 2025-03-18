from __future__ import annotations

import itertools
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
import requests
from matcalc.benchmark import (
    BenchmarkSuite,
    CheckpointFile,
    ElasticityBenchmark,
    PhononBenchmark,
    get_available_benchmarks,
    get_benchmark_data,
)

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator


def test_get_benchmark_data() -> None:
    d = get_benchmark_data("mp-pbe-elasticity-2025.3.json.gz")
    assert len(d) > 10000
    with pytest.raises(requests.RequestException) as _:
        get_benchmark_data("bad_url")


def test_elasticity_benchmark(pes_calculator: PESCalculator) -> None:
    benchmark = ElasticityBenchmark(n_samples=10)
    results = benchmark.run(pes_calculator, "toy")
    assert len(results) == 10
    # Compute MAE
    assert np.abs(results["K_vrh_toy"] - results["K_vrh_DFT"]).mean() == pytest.approx(65.20042336543436, abs=1e-1)

    benchmark = ElasticityBenchmark(benchmark_name="mp-pbe-elasticity-2025.3.json.gz", n_samples=10)

    chkpt_file = "checkpoint.json"

    results = benchmark.run(
        pes_calculator,
        "toy",
        checkpoint_file=chkpt_file,
        checkpoint_freq=3,
        include_full_results=True,
    )

    assert len(results.columns) == 10
    assert "structure" in results.columns

    assert os.path.exists(chkpt_file)
    results, *_ = CheckpointFile(chkpt_file).load()
    assert len(results) % 3 == 0

    os.remove(chkpt_file)


def test_phonon_benchmark(pes_calculator: PESCalculator) -> None:
    benchmark = PhononBenchmark(n_samples=10, write_phonon=False)
    results = benchmark.run(pes_calculator, "toy")
    assert len(results) == 10
    assert np.abs(results["CV_toy"] - results["CV_DFT"]).mean() == pytest.approx(27.636954450580486, abs=1e-1)


def test_benchmark_suite(pes_calculator: PESCalculator) -> None:
    elasticity_benchmark = ElasticityBenchmark(n_samples=2, benchmark_name="mp-pbe-elasticity-2025.3.json.gz")
    elasticity_benchmark.run(
        pes_calculator,
        "toy1",
        checkpoint_file="checkpoint.json",
        checkpoint_freq=1,
    )
    phonon_benchmark = PhononBenchmark(n_samples=2, write_phonon=False)
    suite = BenchmarkSuite(benchmarks=[elasticity_benchmark, phonon_benchmark])
    results = suite.run({"toy1": pes_calculator, "toy2": pes_calculator}, checkpoint_freq=1)
    for bench, name in itertools.product(["ElasticityBenchmark", "PhononBenchmark"], ["toy1", "toy2"]):
        assert os.path.exists(f"{bench}_{name}.csv")
        os.remove(f"{bench}_{name}.csv")
    assert len(results) == 2
    assert "K_vrh_toy1" in results[0].columns
    assert "K_vrh_toy2" in results[0].columns
    assert "G_vrh_toy1" in results[0].columns
    assert "G_vrh_toy2" in results[0].columns
    assert "CV_toy1" in results[1].columns
    assert "CV_toy2" in results[1].columns
    os.remove("checkpoint.json")


def test_available_benchmarks() -> None:
    assert "mp-binary-pbe-elasticity-2025.1.json.gz" in get_available_benchmarks()
    assert "alexandria-binary-pbe-phonon-2025.1.json.gz" in get_available_benchmarks()
