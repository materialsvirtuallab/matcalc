---
layout: default
title: API Documentation
nav_order: 5
---

# matcalc package

Calculators for materials properties.


## matcalc.base module

Define basic API.

### *class* PropCalc

Bases: `object`

API for a property calculator.

#### \_abc_impl *= <_abc._abc_data object>*

#### *abstract* calc(structure: Structure)

All PropCalc subclasses should implement a calc method that takes in a pymatgen structure
and returns a dict. The method can return more than one property.

* **Parameters:**
  **structure** – Pymatgen structure.
* **Returns:**
  In the form {“prop_name”: value}.
* **Return type:**
  dict[str, Any]

#### calc_many(structures: Sequence[Structure], n_jobs: None | int = None, \*\*kwargs)

Performs calc on many structures. The return type is a generator given that the calc method can potentially be
expensive. It is trivial to convert the generator to a list/tuple.

* **Parameters:**
  * **structures** – List or generator of Structures.
  * **n_jobs** – The maximum number of concurrently running jobs. If -1 all CPUs are used. For n_jobs below -1,
    (n_cpus + 1 + n_jobs) are used. None is a marker for unset that will be interpreted as n_jobs=1
    unless the call is performed under a parallel_config() context manager that sets another value for
    n_jobs.
  * **\*\*kwargs** – Passthrough to joblib.Parallel.
* **Returns:**
  Generator of dicts.

## matcalc.elasticity module

Calculator for phonon properties.

### *class* ElasticityCalc(calculator: Calculator, norm_strains: float = 0.01, shear_strains: float = 0.01, fmax: float = 0.1, relax_structure: bool = True)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Calculator for elastic properties.

* **Parameters:**
  * **calculator** – ASE Calculator to use.
  * **fmax** – maximum force in the relaxed structure (if relax_structure).
  * **norm_strains** – strain value to apply to each normal mode.
  * **shear_strains** – strain value to apply to each shear mode.
  * **relax_structure** – whether to relax the provided structure with the given calculator.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure)

Calculates elastic properties of Pymatgen structure with units determined by the calculator.

* **Parameters:**
  **structure** – Pymatgen structure.

Returns: {
: elastic_tensor: Elastic tensor as a pymatgen ElasticTensor object,
  shear_modulus_vrh: Voigt-Reuss-Hill shear modulus based on elastic tensor,
  bulk_modulus_vrh: Voigt-Reuss-Hill bulk modulus based on elastic tensor,
  youngs_modulus: Young’s modulus based on elastic tensor,

}

## matcalc.eos module

Calculators for EOS and associated properties.

### *class* EOSCalc(calculator: Calculator, optimizer: Optimizer | str = 'FIRE', max_steps: int = 500, max_abs_strain: float = 0.1, n_points: int = 11, fmax: float = 0.1, relax_structure: bool = True)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Equation of state calculator.

* **Parameters:**
  * **calculator** – ASE Calculator to use.
  * **optimizer** (*str* *|* *ase Optimizer*) – The optimization algorithm. Defaults to “FIRE”.
  * **max_steps** (*int*) – Max number of steps for relaxation. Defaults to 500.
  * **max_abs_strain** (*float*) – The maximum absolute strain applied to the structure. Defaults to 0.1 (10% strain).
  * **n_points** (*int*) – Number of points in which to compute the EOS. Defaults to 11.
  * **fmax** (*float*) – Max force for relaxation (of structure as well as atoms).
  * **relax_structure** – Whether to first relax the structure. Set to False if structures provided are pre-relaxed
    with the same calculator. Defaults to True.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure)

Fit the Birch-Murnaghan equation of state.

* **Parameters:**
  **structure** – pymatgen Structure object.

Returns: {
: eos: {
  : volumes: list[float] in Angstrom^3,
    energies: list[float] in eV,
  <br/>
  },
  bulk_modulus_bm: Birch-Murnaghan bulk modulus in GPa.

}

## matcalc.phonon module

Calculator for phonon properties.

### *class* PhononCalc(calculator: Calculator, atom_disp: float = 0.015, supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)), t_step: float = 10, t_max: float = 1000, t_min: float = 0, fmax: float = 0.1, relax_structure: bool = True)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Calculator for phonon properties.

* **Parameters:**
  * **calculator** – ASE Calculator to use.
  * **fmax** – Max forces. This criterion is more stringent than for simple relaxation.
  * **atom_disp** – Atomic displacement
  * **supercell_matrix** – Supercell matrix to use. Defaults to 2x2x2 supercell.
  * **t_step** – Temperature step.
  * **t_max** – Max temperature.
  * **t_min** – Min temperature.
  * **relax_structure** – Whether to first relax the structure. Set to False if structures
    provided are pre-relaxed with the same calculator.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure)

Calculates thermal properties of Pymatgen structure with phonopy.

* **Parameters:**
  **structure** – Pymatgen structure.

Returns:
{

> phonon: Phonopy object with force constants produced
> thermal_properties:

> > {
> > : temperatures: list of temperatures in Kelvin,
> >   free_energy: list of Helmholtz free energies at corresponding temperatures in eV,
> >   entropy: list of entropies at corresponding temperatures in eV/K,
> >   heat_capacity: list of heat capacities at constant volume at corresponding temperatures in eV/K^2,

> > }

}

### \_calc_forces(calculator: Calculator, supercell: PhonopyAtoms)

Helper to compute forces on a structure.

* **Parameters:**
  * **calculator** – ASE Calculator
  * **supercell** – Supercell from phonopy.
* **Returns:**
  forces

## matcalc.relaxation module

Relaxation properties.

### *class* RelaxCalc(calculator: Calculator, optimizer: Optimizer | str = 'FIRE', max_steps: int = 500, traj_file: str | None = None, interval: int = 1, fmax: float = 0.1, relax_cell: bool = True)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Relaxes and computes the relaxed parameters of a structure.

* **Parameters:**
  * **calculator** – ASE Calculator to use.
  * **optimizer** (*str* *|* *ase Optimizer*) – The optimization algorithm. Defaults to “FIRE”.
  * **max_steps** (*int*) – Max number of steps for relaxation. Defaults to 500.
  * **traj_file** (*str* *|* *None*) – File to save the trajectory to. Defaults to None.
  * **interval** (*int*) – The step interval for saving the trajectories. Defaults to 1.
  * **fmax** (*float*) – Total force tolerance for relaxation convergence.
    fmax is a sum of force and stress forces. Defaults to 0.1 (eV/A).
  * **relax_cell** (*bool*) – Whether to relax the cell (or just atoms).
* **Raises:**
  **ValueError** – If the optimizer is not a valid ASE optimizer.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure)

Perform relaxation to obtain properties.

* **Parameters:**
  **structure** – Pymatgen structure.

Returns: {
: final_structure: final_structure,
  energy: trajectory observer final energy in eV,
  volume: lattice.volume in A^3,
  a: lattice.a in A,
  b: lattice.b in A,
  c: lattice.c in A,
  alpha: lattice.alpha in degrees,
  beta: lattice.beta in degrees,
  gamma: lattice.gamma in degrees,

}

### *class* TrajectoryObserver(atoms: Atoms)

Bases: `object`

Trajectory observer is a hook in the relaxation process that saves the
intermediate structures.

Init the Trajectory Observer from a Atoms.

* **Parameters:**
  **atoms** (*Atoms*) – Structure to observe.

#### save(filename: str)

Save the trajectory to file.

* **Parameters:**
  **filename** (*str*) – filename to save the trajectory.

## matcalc.util module

Some utility methods, e.g., for getting calculators from well-known sources.

### get_universal_calculator(name: str | Calculator, \*\*kwargs)

Helper method to get some well-known **universal** calculators.
Imports should be inside if statements to ensure that all models are optional dependencies.
All calculators must be universal, i.e. encompass a wide swath of the periodic table.
Though matcalc can be used with any MLIP, even custom ones, this function is not meant as

> a list of all MLIPs.
* **Parameters:**
  * **name** (*str*) – Name of calculator.
  * **\*\*kwargs** – Passthrough to calculator init.
* **Raises:**
  **ValueError** – on unrecognized model name.
* **Returns:**
  Calculator
