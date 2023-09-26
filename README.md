<h1 align="center">
  <img src="https://github.com/materialsvirtuallab/matcalc/assets/30958850/89486f2f-73fb-40fb-803a-dfafe510eb6d" width="150" height="150" alt="MatCalc logo" style="vertical-align: middle;" />
  MatCalc
</h1>

<h4 align="center">

[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matcalc)](https://github.com/materialsvirtuallab/matcalc/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matcalc/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/matcalc/workflows/Testing/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Testing/badge.svg)
[![codecov](https://codecov.io/gh/materialsvirtuallab/matcalc/branch/main/graph/badge.svg?token=OR7Z9WWRRC)](https://codecov.io/gh/materialsvirtuallab/matcalc)
[![Requires Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/matcalc?logo=pypi&logoColor=white)](https://pypi.org/project/matcalc?logo=pypi&logoColor=white)
</h4>

## Introduction

MatCalc is a Python library for calculating materials properties from the potential energy surface (PES). The
PES can be from DFT or, more commonly, from machine learning interatomic potentials (MLIPs).

Calculating various materials properties can require relatively involved setup of various simulation codes. The
goal of MatCalc is to provide a simplified, consistent interface to access these properties with any
parameterization of the PES.

## Outline

The main base class in MatCalc is `PropCalc` (property calculator). [All `PropCalc` subclasses](https://github.com/search?q=repo%3Amaterialsvirtuallab%2Fmatcalc%20%22(PropCalc)%22) should implement a
`calc(pymatgen.Structure) -> dict` method that returns a dict of properties.

In general, `PropCalc` should be initialized with an ML model or ASE calculator, which is then used by either ASE,
LAMMPS or some other simulation code to perform calculations of properties.
