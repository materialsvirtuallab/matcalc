# Introduction

MatCalc is a python library for calculating materials properties using machine learning interatomic potentials (MLIPs).
Calculating various materials properties can require relatively involved setup of various simulation codes. The
goal of MatCalc is to provide a simplified interface to access these properties with any MLIP.

# Outline

The main base class in MatCalc is PropCalc (property calculator). All PropCalc subclasses should implement a
`calc(structure) -> dict` method that takes in a Pymatgen Structure and returns a dict of properties.

In general, PropCalc should be initialized with an ML model or ASE calculator, which is then used by either ASE,
LAMMPS or some other simulation code to perform calculations of properties.
