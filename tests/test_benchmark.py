from __future__ import annotations

import itertools
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
import requests
from matcalc.benchmark import (
    BenchmarkSuite,
    ElasticityBenchmark,
    PhononBenchmark,
    _load_checkpoint,
    get_available_benchmarks,
    get_benchmark_data,
)

if TYPE_CHECKING:
    from matgl.ext.ase import M3GNetCalculator


def test_get_benchmark_data() -> None:
    d = get_benchmark_data("mp-pbe-elasticity-2025.3.json.gz")
    assert len(d) > 10000
    with pytest.raises(requests.RequestException) as _:
        get_benchmark_data("bad_url")


def test_elasticity_benchmark(M3GNetCalc: M3GNetCalculator) -> None:
    benchmark = ElasticityBenchmark(n_samples=10)
    results = benchmark.run(M3GNetCalc, "toy")
    assert len(results) == 10
    # Compute MAE
    assert np.abs(results["bulk_modulus_vrh_toy"] - results["bulk_modulus_vrh_DFT"]).mean() == pytest.approx(
        65.20042336543436, abs=1e-1
    )

    benchmark = ElasticityBenchmark(benchmark_name="mp-pbe-elasticity-2025.3.json.gz", n_samples=10)
    benchmark.run(M3GNetCalc, "toy", checkpoint_file="checkpoint.csv", checkpoint_freq=3)
    assert os.path.exists("checkpoint.csv")
    results, data, structures = _load_checkpoint(
        "checkpoint.csv", benchmark.ground_truth, benchmark.structures, "mp_id"
    )
    assert len(results) % 3 == 0
    os.remove("checkpoint.csv")


def test_phonon_benchmark(M3GNetCalc: M3GNetCalculator) -> None:
    benchmark = PhononBenchmark(n_samples=10, write_phonon=False)
    results = benchmark.run(M3GNetCalc, "toy")
    assert len(results) == 10
    assert np.abs(results["CV_toy"] - results["CV_DFT"]).mean() == pytest.approx(27.636954450580486, abs=1e-1)


def test_benchmark_suite(M3GNetCalc: M3GNetCalculator) -> None:
    elasticity_benchmark = ElasticityBenchmark(n_samples=2)
    phonon_benchmark = PhononBenchmark(n_samples=2, write_phonon=False)
    suite = BenchmarkSuite(benchmarks=[elasticity_benchmark, phonon_benchmark])
    results = suite.run({"toy1": M3GNetCalc, "toy2": M3GNetCalc}, checkpoint_freq=1)
    for bench, name in itertools.product(["ElasticityBenchmark", "PhononBenchmark"], ["toy1", "toy2"]):
        assert os.path.exists(f"{bench}_{name}.csv")
        os.remove(f"{bench}_{name}.csv")
    assert len(results) == 2
    assert "bulk_modulus_vrh_toy1" in results[0].columns
    assert "bulk_modulus_vrh_toy2" in results[0].columns
    assert "shear_modulus_vrh_toy1" in results[0].columns
    assert "shear_modulus_vrh_toy2" in results[0].columns
    assert "CV_toy1" in results[1].columns
    assert "CV_toy2" in results[1].columns


def test_available_benchmarks() -> None:
    assert "mp-binary-pbe-elasticity-2025.1.json.gz" in get_available_benchmarks()
    assert "alexandria-binary-pbe-phonon-2025.1.json.gz" in get_available_benchmarks()
