from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from matcalc.benchmark import ElasticityBenchmark

if TYPE_CHECKING:
    from matgl.ext.ase import M3GNetCalculator


def test_elasticity_benchmark(M3GNetCalc: M3GNetCalculator) -> None:
    benchmark = ElasticityBenchmark(n_samples=10)
    results = benchmark.run(M3GNetCalc, "toy")
    assert len(results) == 10
    assert results["AE K_toy"].mean() == pytest.approx(65.20042336543436, abs=1e-1)
