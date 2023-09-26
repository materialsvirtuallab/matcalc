"""Setup script for matcalc."""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="matcalc",
    version="0.0.3",
    author="Eliott Liu, Ji Qi, Tsz Wai Ko, Janosh Riebesell, Shyue Ping Ong",
    author_email="ongsp@ucsd.edu",
    maintainer="Shyue Ping Ong",
    maintainer_email="ongsp@ucsd.edu",
    description="Calculators for materials properties from the potential energy surface.",
    long_description=open("README.md").read(),  # noqa: SIM115
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
    python_requires=">=3.8",
    package_data={},
    install_requires=("ase", "pymatgen", "joblib", "phonopy"),
    extras_require={"models": ["matgl", "chgnet"]},
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
