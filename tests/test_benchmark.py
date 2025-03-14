from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from matcalc.benchmark import BenchmarkSuite, ElasticityBenchmark, get_available_benchmarks

if TYPE_CHECKING:
    from matgl.ext.ase import M3GNetCalculator


def test_elasticity_benchmark(M3GNetCalc: M3GNetCalculator) -> None:
    benchmark = ElasticityBenchmark(n_samples=10)
    results = benchmark.run(M3GNetCalc, "toy")
    assert len(results) == 10
    assert results["AE K_toy"].mean() == pytest.approx(65.20042336543436, abs=1e-1)


def test_benchmark_suite(M3GNetCalc: M3GNetCalculator) -> None:
    # TODO: This needs to be done with different benchmarks.
    benchmark = ElasticityBenchmark(n_samples=3)
    suite = BenchmarkSuite(benchmarks=[benchmark, benchmark])
    results = suite.run({"toy1": M3GNetCalc, "toy2": M3GNetCalc})

    assert len(results) == 2
    assert "K_toy1" in results[0].columns
    assert "K_toy2" in results[0].columns
    assert "G_toy1" in results[0].columns
    assert "G_toy2" in results[0].columns


def test_available_benchmarks() -> None:
    assert "mp-binary-elasticity-2025.1.json.gz" in get_available_benchmarks()
