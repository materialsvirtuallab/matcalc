from __future__ import annotations

import os

from setuptools import find_packages, setup

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="matcalc",
    version="0.0.1",
    author="Eliott Liu, Ji Qi, Tsz Wai Ko, Shyue Ping Ong",
    author_email="ongsp@ucsd.edu",
    maintainer="Shyue Ping Ong",
    maintainer_email="ongsp@ucsd.edu",
    description="Calculators for materials properties.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "materials",
        "interatomic potential",
        "force field",
        "science",
        "property prediction",
        "AI",
        "machine learning",
        "graph",
        "deep learning",
    ],
    packages=find_packages(),
    package_data={},
    install_requires=("ase", "pymatgen", "joblib", "phonopy"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
