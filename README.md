# Introduction

MatCalc is a python library for calculating materials properties using machine learning interatomic potentials.

# Outline

The main base class in MatCalc is PropCalc (property calculator). All PropCalc subclasses should implement a
`calc(structure) -> dict` method that takes in a Pymatgen Structure and returns a dict of properties.

In general, PropCalc should be initialized with a model, which is then used by either ASE or LAMMPS or some
other simulation code to perform calculations of properties.
