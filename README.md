<h1 align="center">
  <img src="https://github.com/materialsvirtuallab/matcalc/assets/30958850/89486f2f-73fb-40fb-803a-dfafe510eb6d" width="100" alt="MatCalc logo" style="vertical-align: middle;" /><br>
  MatCalc
</h1>

[![Testing](https://github.com/materialsvirtuallab/matcalc/workflows/Testing/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Testing/badge.svg)
[![Linting](https://github.com/materialsvirtuallab/matcalc/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Linting/badge.svg)
[![codecov](https://codecov.io/gh/materialsvirtuallab/matcalc/branch/main/graph/badge.svg?token=OR7Z9WWRRC)](https://codecov.io/gh/materialsvirtuallab/matcalc)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/matcalc?logo=pypi&logoColor=white)](https://pypi.org/project/matcalc?logo=pypi&logoColor=white)
[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matcalc)](https://github.com/materialsvirtuallab/matcalc/blob/main/LICENSE)

## Introduction

MatCalc is a Python library for calculating and benchmarking material properties from the potential energy surface
(PES). The PES can come from DFT or, more commonly, from machine learning interatomic potentials (MLIPs).

Calculating material properties often requires involved setups of various simulation codes. The
goal of MatCalc is to provide a simplified, consistent interface to access these properties with any
parameterization of the PES.

MatCalc is part of the MatML ecosystem, which includes the [MatGL] (Materials Graph Library) and [maml] (MAterials
Machine Learning) packages, the [MatPES] (Materials Potential Energy Surface) dataset, and the [MatCalc] (Materials
Calculator).

## Documentation

The API documentation and tutorials are available at https://matcalc.ai.

## Outline

The main base class in MatCalc is `PropCalc` (property calculator). [All `PropCalc` subclasses](https://github.com/search?q=repo%3Amaterialsvirtuallab%2Fmatcalc%20%22(PropCalc)%22) should implement a
`calc(pymatgen.Structure) -> dict` method that returns a dictionary of properties.

In general, `PropCalc` should be initialized with an ML model or [ASE] calculator, which is then used by either ASE,
LAMMPS or some other simulation code to perform calculations of properties. The `matcalc.PESCalculator` class
provides easy access to many universal MLIPs as well as an interface to

# Basic Usage

MatCalc provides convenient methods to quickly compute properties, using a minimal amount of code. The following is
an example of a computation of the elastic constants of Si using the `TensorNet-MatPES-PBE-v2025.1-PES` universal MLIP.

```python
from matcalc import PESCalculator, ElasticityCalc
from pymatgen.ext.matproj import MPRester

mpr = MPRester()
si = mpr.get_structure_by_material_id("mp-149")
c = ElasticityCalc(PESCalculator.load_universal("TensorNet-MatPES-PBE-v2025.1-PES"), relax_structure=True)
props = c.calc(si)
print(f"K_VRH = {props['bulk_modulus_vrh'] * 160.2176621} GPa")
```

The output is `K_VRH = 102.08363100102596 GPa`.

MatCalc also supports trivial parallelization using joblib via the `calc_many` method.

```python
structures = [si] * 20

def serial_calc():
    return [c.calc(s) for s in structures]

def parallel_calc():
    # n_jobs = -1 uses all processors available.
    return list(c.calc_many(structures, n_jobs=-1))

%timeit -n 5 -r 1 serial_calc()
# Output is 8.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)

%timeit -n 5 -r 1 parallel_calc()
# Output is 2.08 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)
# This was run on 10 CPUs on a Mac.
```

### CLI tool

A CLI tool provides a means to use universal MLIPs to obtain properties for any structure. Example usage:

```shell
matcalc calc -p ElasticityCalc -s Li2O.cif
```

## Benchmarking

MatCalc makes it easy to perform a large number of calculations rapidly. With the release of MatPES, we have released
the `MatCalc-Benchmark`.

For example, the following code can be used to run the ElasticityBenchmark on `TensorNet-MatPES-PBE-v2025.1-PES` UMLIP.

```python
from matcalc import PESCalculator
calculator = PESCalculator.load_universal("TensorNet-MatPES-PBE-v2025.1-PES")
from matcalc.benchmark import ElasticityBenchmark
benchmark = ElasticityBenchmark(fmax=0.05, relax_structure=True)
results = benchmark.run(calculator, "TensorNet-MatPES")
```

The entire run takes ~ 16mins when parallelized over 10 CPUs on a Mac.

You can even run entire suites of benchmarks on multiple models, as follows:

```python
from matcalc import PESCalculator
tensornet = PESCalculator.load_universal("TensorNet-MatPES-PBE-v2025.1-PES")
m3gnet = PESCalculator.load_universal("M3GNet-MatPES-PBE-v2025.1-PES")
from matcalc.benchmark import BenchmarkSuite, ElasticityBenchmark, PhononBenchmark

elasticity_benchmark = ElasticityBenchmark(fmax=0.5, relax_structure=True)
phonon_benchmark = PhononBenchmark(write_phonon=False)
suite = BenchmarkSuite(benchmarks=[elasticity_benchmark, phonon_benchmark])
results = suite.run({"M3GNet": m3gnet, "TensorNet": tensornet})
results.to_csv("benchmark_results.csv")
```

These will usually take a long time to run. Running on HPC resources is recommended. Please set `n_samples` when
initializing the benchmark to limit the number of calculations to do some testing before running the full benchmark.

## Citing

A manuscript on `matcalc` is currently in the works. In the meantime, please see [`citation.cff`](citation.cff) or the GitHub
sidebar for a BibTeX and APA citation.

[maml]: https://materialsvirtuallab.github.io/maml/
[MatGL]: https://matgl.ai
[MatPES]: https://matpes.ai
[MatCalc]: https://matcalc.ai
[ASE]: https://wiki.fysik.dtu.dk/ase/
