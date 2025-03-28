---
layout: default
title: API Documentation
nav_order: 5
---

# matcalc package

Calculators for materials properties.


## matcalc._base module

Define basic API.

### *class* ChainedCalc(prop_calcs: Sequence[[PropCalc](#matcalc._base.PropCalc)])

Bases: [`PropCalc`](#matcalc._base.PropCalc)

A chained calculator that runs a series of PropCalcs on a structure or set of structures.

Often, you may want to obtain multiple properties at once, e.g., perform a relaxation with a formation energy
computation and a elasticity calculation. This can be done using this class by supplying a list of calculators.
Note that it is likely

Initialize a chained calculator.

* **Parameters:**
  **prop_calcs** – Sequence of prop calcs.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure | dict[str, Any]) → dict[str, Any]

Runs the series of PropCalcs on a structure.

* **Parameters:**
  **structure** – Pymatgen structure or a dict containing a pymatgen Structure under a “final_structure” or
  “structure” key. Allowing dicts provide the means to chain calculators, e.g., do a relaxation followed
  by an elasticity calculation.
* **Returns:**
  In the form {“prop_name”: value}.
* **Return type:**
  dict[str, Any]

#### calc_many(structures: Sequence[Structure | dict[str, Any]], n_jobs: None | int = None, allow_errors: bool = False, \*\*kwargs: Any) → Generator[dict | None]

Runs the sequence of PropCalc on many structures.

* **Parameters:**
  * **structures** – List or generator of Structures.
  * **n_jobs** – The maximum number of concurrently running jobs. If -1 all CPUs are used. For n_jobs below -1,
    (n_cpus + 1 + n_jobs) are used. None is a marker for unset that will be interpreted as n_jobs=1
    unless the call is performed under a parallel_config() context manager that sets another value for
    n_jobs.
  * **allow_errors** – Whether to skip failed calculations. For these calculations, None will be returned. For
    large scale calculations, you may want this to be True to avoid the entire calculation failing.
    Defaults to False.
  * **\*\*kwargs** – Passthrough to calc_many method of all PropCalcs.
* **Returns:**
  Generator of dicts.

### *class* PropCalc

Bases: `ABC`

Abstract base class for property calculations.

This class defines the interface for performing property calculations on
structures (using pymatgen’s Structure objects or a dictionary containing a
pymatgen structure). Subclasses are expected to implement the calc method
to define specific property calculation logic. Additionally, this class provides
an implementation of the calc_many method, which enables concurrent calculations
on multiple structures using joblib.

#### \_abc_impl *= <_abc._abc_data object>*

#### *abstract* calc(structure: Structure | dict[str, Any]) → dict[str, Any]

Abstract method to calculate and return a standardized format of structural data.

This method processes input structural data, which could either be a dictionary
or a pymatgen Structure object, and returns a dictionary representation. If a
dictionary is provided, it must include either the key `final_structure` or
`structure`. For a pymatgen Structure input, it will be converted to a dictionary
with the key `final_structure`. To support chaining, a super() call should be made
by subclasses to ensure that the input dictionary is standardized.

* **Parameters:**
  **structure** (*Structure* *|* *dict* *[**str* *,* *Any* *]*) – A pymatgen Structure object or a dictionary containing structural
  data with keys such as `final_structure` or `structure`.
* **Returns:**
  A dictionary with the key `final_structure` mapping to the corresponding
  structural data.
* **Return type:**
  dict[str, Any]
* **Raises:**
  **ValueError** – If the input dictionary does not include the required keys
  `final_structure` or `structure`.

#### calc_many(structures: Sequence[Structure | dict[str, Any]], n_jobs: None | int = None, allow_errors: bool = False, \*\*kwargs: Any) → Generator[dict | None]

Calculate properties for multiple structures concurrently.

This method leverages parallel processing to compute properties for a
given sequence of structures. It uses the joblib.Parallel library to
support multi-job execution and manage error handling behavior based
on user configuration.

* **Parameters:**
  * **structures** – A sequence of Structure objects or dictionaries
    representing the input structures to be processed. Each entry in
    the sequence is processed independently.
  * **n_jobs** – The number of jobs to run in parallel. If set to None,
    joblib will determine the optimal number of jobs based on the
    system’s CPU configuration.
  * **allow_errors** – A boolean flag indicating whether to tolerate
    exceptions during processing. When set to True, any failed
    calculation will result in a None value for that structure
    instead of raising an exception.
  * **kwargs** – Additional keyword arguments passed directly to
    joblib.Parallel, which allows customization of parallel
    processing behavior.
* **Returns:**
  A generator yielding dictionaries with computed properties
  for each structure or None if an error occurred (depending on
  the allow_errors flag).

## matcalc._cli module

Command line interface to matcalc.

### calculate_property(args: Any) → None

Implements calculate property.

* **Parameters:**
  **args**
* **Returns:**

### clear_cache(args: Any) → None

Clear the benchmark cache.

* **Parameters:**
  **args**
* **Returns:**

### main() → None

Handle main.

## matcalc._elasticity module

Calculator for elastic properties.

### *class* ElasticityCalc(calculator: Calculator, \*, norm_strains: Sequence[float] | float = (-0.01, -0.005, 0.005, 0.01), shear_strains: Sequence[float] | float = (-0.06, -0.03, 0.03, 0.06), fmax: float = 0.1, symmetry: bool = False, relax_structure: bool = True, relax_deformed_structures: bool = False, use_equilibrium: bool = True, relax_calc_kwargs: dict | None = None)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

Class for calculating elastic properties of a material. This includes creating
an elastic tensor, shear modulus, bulk modulus, and other related properties with
the help of strain and stress analyses. It leverages the provided ASE Calculator
for computations and supports relaxation of structures when necessary.

* **Variables:**
  * **calculator** – The ASE Calculator used for performing computations.
  * **norm_strains** – Sequence of normal strain values to be applied.
  * **shear_strains** – Sequence of shear strain values to be applied.
  * **fmax** – Maximum force tolerated for structure relaxation.
  * **symmetry** – Whether to apply symmetry reduction techniques during calculations.
  * **relax_structure** – Whether the initial structure should be relaxed before applying strains.
  * **relax_deformed_structures** – Whether to relax atomic positions in deformed/strained structures.
  * **use_equilibrium** – Whether to use equilibrium stress and strain in calculations.
  * **relax_calc_kwargs** – Additional arguments for relaxation calculations.

Initializes the class with parameters to construct normalized and shear strain values
and control relaxation behavior for structures. Validates input parameters to ensure
appropriate constraints are maintained.

* **Parameters:**
  * **calculator** – Calculator object used for performing calculations.
  * **norm_strains** – Sequence of normalized strain values applied during deformation.
    Can also be a single float. Must not be empty or contain zero.
  * **shear_strains** – Sequence of shear strain values applied during deformation.
    Can also be a single float. Must not be empty or contain zero.
  * **fmax** – Maximum force magnitude tolerance for relaxation. Default is 0.1.
  * **symmetry** – Boolean flag to enforce symmetry in deformation. Default is False.
  * **relax_structure** – Boolean flag indicating if the structure should be relaxed before
    applying strains. Default is True.
  * **relax_deformed_structures** – Boolean flag indicating if the deformed structures
    should be relaxed. Default is False.
  * **use_equilibrium** – Boolean flag indicating if equilibrium conditions should be used for
    calculations. Automatically enabled if multiple normal and shear strains are provided.
  * **relax_calc_kwargs** – Optional dictionary containing keyword arguments for structure
    relaxation calculations.

#### \_abc_impl *= <_abc._abc_data object>*

#### \_elastic_tensor_from_strains(strains: ArrayLike, stresses: ArrayLike, eq_stress: ArrayLike = None, tol: float = 1e-07) → tuple[ElasticTensor, float]

Compute the elastic tensor from given strain and stress data using least-squares
fitting.

This function calculates the elastic constants from strain-stress relations,
using a least-squares fitting procedure for each independent component of stress
and strain tensor pairs. An optional equivalent stress array can be supplied.
Residuals from the fitting process are accumulated and returned alongside the
elastic tensor. The elastic tensor is zeroed according to the given tolerance.

* **Parameters:**
  * **strains** – Strain data array-like, representing different strain states.
  * **stresses** – Stress data array-like corresponding to the given strain states.
  * **eq_stress** – Optional array-like, equivalent stress values for equilibrium stress states.
    Defaults to None.
  * **tol** – A float representing the tolerance threshold used for zeroing the elastic
    tensor. Defaults to 1e-7.
* **Returns:**
  A tuple consisting of:
  : - ElasticTensor object: The computed and zeroed elastic tensor in Voigt
      notation.
    - float: The summed residuals from least-squares fittings across all
      tensor components.

#### calc(structure: Structure | dict[str, Any]) → dict[str, Any]

Performs a calculation to determine the elastic tensor and related elastic
properties. It involves multiple steps such as optionally relaxing the input
structure, generating deformed structures, calculating stresses, and evaluating
elastic properties. The method supports equilibrium stress computation and various
relaxations depending on configuration.

* **Parameters:**
  **structure** – The input structure which can either be an instance of Structure or
  a dictionary containing structural data.
* **Returns:**
  A dictionary containing the calculation results that include:
  - elastic_tensor: The computed elastic tensor of the material.
  - shear_modulus_vrh: Shear modulus obtained from the elastic tensor
  > using the Voigt-Reuss-Hill approximation.
  - bulk_modulus_vrh: Bulk modulus calculated using the Voigt-Reuss-Hill
    approximation.
  - youngs_modulus: Young’s modulus derived from the elastic tensor.
  - residuals_sum: The residual sum from the elastic tensor fitting.
  - structure: The (potentially relaxed) final structure after calculations.

## matcalc._eos module

Calculators for EOS and associated properties.

### *class* EOSCalc(calculator: Calculator, \*, optimizer: Optimizer | str = 'FIRE', max_steps: int = 500, max_abs_strain: float = 0.1, n_points: int = 11, fmax: float = 0.1, relax_structure: bool = True, relax_calc_kwargs: dict | None = None)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

Performs equation of state (EOS) calculations using a specified ASE calculator.

This class is intended to fit the Birch-Murnaghan equation of state, determine the
bulk modulus, and provide other relevant physical properties of a given structure.
The EOS calculation includes applying volumetric strain to the structure, optional
initial relaxation of the structure, and evaluation of energies and volumes
corresponding to the applied strain.

* **Variables:**
  * **calculator** – The ASE Calculator used for the calculations.
  * **optimizer** – Optimization algorithm. Defaults to “FIRE”.
  * **relax_structure** – Indicates if the structure should be relaxed initially. Defaults to True.
  * **n_points** – Number of strain points for the EOS calculation. Defaults to 11.
  * **max_abs_strain** – Maximum absolute volumetric strain. Defaults to 0.1 (10% strain).
  * **fmax** – Maximum force tolerance for relaxation. Defaults to 0.1 eV/Å.
  * **max_steps** – Maximum number of optimization steps during relaxation. Defaults to 500.
  * **relax_calc_kwargs** – Additional keyword arguments for relaxation calculations. Defaults to None.

Constructor for initializing the data and configurations necessary for a
calculation and optimization process. This class enables the setup of
simulation parameters, structural relaxation options, and optimizations
with specified constraints and tolerances.

* **Parameters:**
  * **calculator** (*Calculator*) – The calculator object that handles the computation of
    forces, energies, and other related properties for the system being
    studied.
  * **optimizer** (*Optimizer* *|* *str* *,* *optional*) – The optimization algorithm used for structural relaxations
    or energy minimizations. Can be an optimizer object or the string name
    of the algorithm. Default is “FIRE”.
  * **max_steps** (*int* *,* *optional*) – The maximum number of steps allowed during the optimization
    or relaxation process. Default is 500.
  * **max_abs_strain** (*float* *,* *optional*) – The maximum allowable absolute strain for relaxation
    processes. Default is 0.1.
  * **n_points** (*int* *,* *optional*) – The number of points or configurations evaluated during
    the simulation or calculation process. Default is 11.
  * **fmax** (*float* *,* *optional*) – The force convergence criterion, specifying the maximum force
    threshold (per atom) for stopping relaxations. Default is 0.1.
  * **relax_structure** (*bool* *,* *optional*) – A flag indicating whether structural relaxation
    should be performed before proceeding with further steps. Default is True.
  * **relax_calc_kwargs** (*dict* *|* *None* *,* *optional*) – Additional keyword arguments to customize the
    relaxation calculation process. Default is None.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure | dict[str, Any]) → dict

Performs energy-strain calculations using Birch-Murnaghan equations of state to extract
equation of state properties such as bulk modulus and R-squared score of the fit.

This function calculates properties of a material system under strain, specifically
its volumetric energy response produced by applying incremental strain, then fits
the Birch-Murnaghan equation of state to the calculated energy and volume data.
Optionally, a relaxation is applied to the structure between calculations of its
strained configurations.

* **Parameters:**
  **structure** – Input structure for calculations. Can be a Structure object or
  a dictionary representation of its atomic configuration and parameters.
* **Returns:**
  A dictionary containing results of the calculations, including relaxed
  structures under conditions of strain, energy-volume data, Birch-Murnaghan
  bulk modulus (in GPa), and R-squared fit of the Birch-Murnaghan model to the
  data.

## matcalc._neb module

NEB calculations.

### *class* NEBCalc(calculator: Calculator, images: list[Structure], \*, optimizer: str | Optimizer = 'BFGS', traj_folder: str | None = None, interval: int = 1, climb: bool = True, \*\*kwargs: Any)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

Nudged Elastic Band calculator.

* **Parameters:**
  * **images** (*list*) – A list of pymatgen structures as NEB image structures.
  * **calculator** (*str* *|* *Calculator*) – ASE Calculator to use. Defaults to M3GNet-MP-2021.2.8-DIRECT-PES.
  * **optimizer** (*str* *|* *Optimizer*) – The optimization algorithm. Defaults to “BEGS”.
  * **traj_folder** (*str* *|* *None*) – The folder address to store NEB trajectories. Defaults to None.
  * **interval** (*int*) – The step interval for saving the trajectories. Defaults to 1.
  * **climb** (*bool*) – Whether to enable climb image NEB. Defaults to True.
  * **kwargs** – Other arguments passed to ASE NEB object.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(fmax: float = 0.1, max_steps: int = 1000) → tuple[float, float]

Perform NEB calculation.

* **Parameters:**
  * **fmax** (*float*) – Convergence criteria for NEB calculations defined by Max forces.
    Defaults to 0.1 eV/A.
  * **max_steps** (*int*) – Maximum number of steps in NEB calculations. Defaults to 1000.
* **Returns:**
  The energy barrier in eV.
* **Return type:**
  float

#### *classmethod* from_end_images(calculator: Calculator, start_struct: Structure, end_struct: Structure, \*, n_images: int = 7, interpolate_lattices: bool = False, autosort_tol: float = 0.5, \*\*kwargs: Any) → [NEBCalc](#matcalc._neb.NEBCalc)

Initialize a NEBCalc from end images.

* **Parameters:**
  * **start_struct** (*Structure*) – The starting image as a pymatgen Structure.
  * **end_struct** (*Structure*) – The ending image as a pymatgen Structure.
  * **calculator** (*str* *|* *Calculator*) – ASE Calculator to use. Defaults to M3GNet-MP-2021.2.8-DIRECT-PES.
  * **n_images** (*int*) – The number of intermediate image structures to create.
  * **interpolate_lattices** (*bool*) – Whether to interpolate the lattices when creating NEB
    path with Structure.interpolate() in pymatgen. Defaults to False.
  * **autosort_tol** (*float*) – A distance tolerance in angstrom in which to automatically
    sort end_struct to match to the closest points in start_struct. This
    argument is required for Structure.interpolate() in pymatgen.
    Defaults to 0.5.
  * **kwargs** – Other arguments passed to construct NEBCalc.

## matcalc._phonon module

Calculator for phonon properties.

### *class* PhononCalc(calculator: Calculator, atom_disp: float = 0.015, supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)), t_step: float = 10, t_max: float = 1000, t_min: float = 0, fmax: float = 0.1, optimizer: str = 'FIRE', relax_structure: bool = True, relax_calc_kwargs: dict | None = None, write_force_constants: bool | str | Path = False, write_band_structure: bool | str | Path = False, write_total_dos: bool | str | Path = False, write_phonon: bool | str | Path = True)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

PhononCalc is a specialized class for calculating thermal properties of structures
using phonopy. It extends the functionalities of base property calculation classes
and integrates phonopy for phonon-related computations.

The class is designed to work with a provided calculator and a structure, enabling
the computation of various thermal properties such as free energy, entropy,
and heat capacity as functions of temperature. It supports relaxation of the
structure, control over displacement magnitudes, and customization of output
file paths for storing intermediate and final results.

* **Variables:**
  * **calculator** – Calculator object to perform energy and force evaluations.
  * **atom_disp** – Magnitude of atomic displacement for phonon calculations.
  * **supercell_matrix** – Matrix defining the supercell size for phonon calculations.
  * **t_step** – Temperature step size in Kelvin for thermal property calculations.
  * **t_max** – Maximum temperature in Kelvin for thermal property calculations.
  * **t_min** – Minimum temperature in Kelvin for thermal property calculations.
  * **fmax** – Maximum force tolerance for structure relaxation.
  * **optimizer** – Optimizer to be used for structure relaxation.
  * **relax_structure** – Flag to indicate whether the structure should be relaxed
    before phonon calculations.
  * **relax_calc_kwargs** – Additional keyword arguments for structure relaxation calculations.
  * **write_force_constants** – Path or boolean flag indicating where to save the
    calculated force constants.
  * **write_band_structure** – Path or boolean flag indicating where to save the
    calculated phonon band structure.
  * **write_total_dos** – Path or boolean flag indicating where to save the total
    density of states (DOS) data.
  * **write_phonon** – Path or boolean flag indicating where to save the full
    phonon data.

#### \_abc_impl *= <_abc._abc_data object>*

#### atom_disp *: float* *= 0.015*

#### calc(structure: Structure | dict[str, Any]) → dict

Calculates thermal properties of Pymatgen structure with phonopy.

* **Parameters:**
  **structure** – Pymatgen structure.

Returns:
{

> phonon: Phonopy object with force constants produced
> thermal_properties:

> > {
> > : temperatures: list of temperatures in Kelvin,
> >   free_energy: list of Helmholtz free energies at corresponding temperatures in kJ/mol,
> >   entropy: list of entropies at corresponding temperatures in J/K/mol,
> >   heat_capacity: list of heat capacities at constant volume at corresponding temperatures in J/K/mol,
> >   The units are originally documented in phonopy.
> >   See phonopy.Phonopy.run_thermal_properties()
> >   ([https://github.com/phonopy/phonopy/blob/develop/phonopy/api_phonopy.py#L2591](https://github.com/phonopy/phonopy/blob/develop/phonopy/api_phonopy.py#L2591))
> >   -> phonopy.phonon.thermal_properties.ThermalProperties.run()
> >   ([https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L498](https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L498))
> >   -> phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_free_energy()
> >   ([https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L217](https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L217))
> >   phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_entropy()
> >   ([https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L233](https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L233))
> >   phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_heat_capacity()
> >   ([https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L225](https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L225))

> > }

}

#### calculator *: Calculator*

#### fmax *: float* *= 0.1*

#### optimizer *: str* *= 'FIRE'*

#### relax_calc_kwargs *: dict | None* *= None*

#### relax_structure *: bool* *= True*

#### supercell_matrix *: ArrayLike* *= ((2, 0, 0), (0, 2, 0), (0, 0, 2))*

#### t_max *: float* *= 1000*

#### t_min *: float* *= 0*

#### t_step *: float* *= 10*

#### write_band_structure *: bool | str | Path* *= False*

#### write_force_constants *: bool | str | Path* *= False*

#### write_phonon *: bool | str | Path* *= True*

#### write_total_dos *: bool | str | Path* *= False*

### \_calc_forces(calculator: Calculator, supercell: PhonopyAtoms) → ArrayLike

Helper to compute forces on a structure.

* **Parameters:**
  * **calculator** – ASE Calculator
  * **supercell** – Supercell from phonopy.
* **Returns:**
  forces

## matcalc._phonon3 module

Calculator for phonon-phonon interaction and related properties.

### *class* Phonon3Calc(calculator: Calculator, fc2_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)), fc3_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)), mesh_numbers: ArrayLike = (20, 20, 20), disp_kwargs: dict[str, Any] = <factory>, thermal_conductivity_kwargs: dict = <factory>, relax_structure: bool = True, relax_calc_kwargs: dict = <factory>, fmax: float = 0.1, optimizer: str = 'FIRE', t_min: float = 0, t_max: float = 1000, t_step: float = 10, write_phonon3: bool | str | Path = False, write_kappa: bool = False)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

Handles the calculation of phonon-phonon interactions and thermal conductivity
using the Phono3py package. Provides functionality for generating displacements,
calculating force constants, and computing lattice thermal conductivity.

Primarily designed for automating the usage of Phono3py in conjunction with
a chosen calculator for force evaluations. Supports options for structure
relaxation before calculation and customization of various computational parameters.

* **Variables:**
  * **calculator** – ASE Calculator used for force evaluations during the calculation.
  * **fc2_supercell** – Supercell matrix for the second-order force constants calculation.
  * **fc3_supercell** – Supercell matrix for the third-order force constants calculation.
  * **mesh_numbers** – Grid mesh numbers for thermal conductivity calculation.
  * **disp_kwargs** – Custom keyword arguments for generating displacements.
  * **thermal_conductivity_kwargs** – Additional keyword arguments for thermal conductivity calculations.
  * **relax_structure** – Flag indicating whether the input structure should be relaxed before calculation.
  * **relax_calc_kwargs** – Additional keyword arguments for the structure relaxation calculator.
  * **fmax** – Maximum force tolerance for the structure relaxation.
  * **optimizer** – Optimizer name to use for structure relaxation.
  * **t_min** – Minimum temperature (in Kelvin) for thermal conductivity calculation.
  * **t_max** – Maximum temperature (in Kelvin) for thermal conductivity calculation.
  * **t_step** – Temperature step size (in Kelvin) for thermal conductivity calculation.
  * **write_phonon3** – Output path for saving Phono3py results, or a boolean to toggle saving.
  * **write_kappa** – Flag indicating whether to write kappa (thermal conductivity values) to output files.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure | dict[str, Any]) → dict

Performs thermal conductivity calculations using the Phono3py library.

This method processes a given atomic structure and calculates its thermal
conductivity through third-order force constants (FC3) computations. The
process involves optional relaxation of the input structure, generation of
displacements, and force calculations corresponding to the supercell
structures. The results include computed thermal conductivity over specified
temperatures, along with intermediate Phono3py configurations.

* **Parameters:**
  **structure** – The atomic structure to compute thermal conductivity for. This can
  be provided as either a Structure object or a dictionary describing
  the structure as per specifications of the input format.
* **Returns:**
  A dictionary containing the following keys:
  - “phonon3”: The configured and processed Phono3py object containing data
  > regarding the phonon interactions and force constants.
  - ”temperatures”: A numpy array of temperatures over which thermal
    conductivity has been computed.
  - ”thermal_conductivity”: The averaged thermal conductivity values computed
    at the specified temperatures. Returns NaN if the values cannot be
    computed.

#### calculator *: Calculator*

#### disp_kwargs *: dict[str, Any]*

#### fc2_supercell *: ArrayLike* *= ((2, 0, 0), (0, 2, 0), (0, 0, 2))*

#### fc3_supercell *: ArrayLike* *= ((2, 0, 0), (0, 2, 0), (0, 0, 2))*

#### fmax *: float* *= 0.1*

#### mesh_numbers *: ArrayLike* *= (20, 20, 20)*

#### optimizer *: str* *= 'FIRE'*

#### relax_calc_kwargs *: dict*

#### relax_structure *: bool* *= True*

#### t_max *: float* *= 1000*

#### t_min *: float* *= 0*

#### t_step *: float* *= 10*

#### thermal_conductivity_kwargs *: dict*

#### write_kappa *: bool* *= False*

#### write_phonon3 *: bool | str | Path* *= False*

## matcalc._qha module

Calculator for phonon properties under quasi-harmonic approximation.

### *class* QHACalc(calculator: Calculator, t_step: float = 10, t_max: float = 1000, t_min: float = 0, fmax: float = 0.1, optimizer: str = 'FIRE', eos: str = 'vinet', relax_structure: bool = True, relax_calc_kwargs: dict | None = None, phonon_calc_kwargs: dict | None = None, scale_factors: Sequence[float] = (0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05), write_helmholtz_volume: bool | str | Path = False, write_volume_temperature: bool | str | Path = False, write_thermal_expansion: bool | str | Path = False, write_gibbs_temperature: bool | str | Path = False, write_bulk_modulus_temperature: bool | str | Path = False, write_heat_capacity_p_numerical: bool | str | Path = False, write_heat_capacity_p_polyfit: bool | str | Path = False, write_gruneisen_temperature: bool | str | Path = False)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

Class for performing quasi-harmonic approximation calculations.

This class utilizes phonopy and Pymatgen-based structure manipulation to calculate
thermal properties such as Gibbs free energy, thermal expansion, heat capacity, and
bulk modulus as a function of temperature under the quasi-harmonic approximation.
It allows for structural relaxation, handling customized scale factors for lattice constants,
and fine-tuning various calculation parameters. Calculation results can be selectively
saved to output files.

* **Variables:**
  * **calculator** – Calculator instance used for electronic structure or energy calculations.
  * **t_step** – Temperature step size in Kelvin.
  * **t_max** – Maximum temperature in Kelvin.
  * **t_min** – Minimum temperature in Kelvin.
  * **fmax** – Maximum force threshold for structure relaxation in eV/Å.
  * **optimizer** – Type of optimizer used for structural relaxation.
  * **eos** – Equation of state used for fitting energy vs. volume data.
  * **relax_structure** – Whether to perform structure relaxation before phonon calculations.
  * **relax_calc_kwargs** – Additional keyword arguments for structure relaxation calculations.
  * **phonon_calc_kwargs** – Additional keyword arguments for phonon calculations.
  * **scale_factors** – List of scale factors for lattice scaling.
  * **write_helmholtz_volume** – Path or boolean to control saving Helmholtz free energy vs. volume data.
  * **write_volume_temperature** – Path or boolean to control saving volume vs. temperature data.
  * **write_thermal_expansion** – Path or boolean to control saving thermal expansion coefficient data.
  * **write_gibbs_temperature** – Path or boolean to control saving Gibbs free energy as a function of temperature.
  * **write_bulk_modulus_temperature** – Path or boolean to control saving bulk modulus vs. temperature data.
  * **write_heat_capacity_p_numerical** – Path or boolean to control saving numerically calculated heat capacity vs.
    temperature data.
  * **write_heat_capacity_p_polyfit** – Path or boolean to control saving polynomial-fitted heat capacity vs.
    temperature data.
  * **write_gruneisen_temperature** – Path or boolean to control saving Grüneisen parameter vs. temperature data.

#### \_abc_impl *= <_abc._abc_data object>*

#### \_calculate_energy(structure: Structure) → float

Helper to calculate the electronic energy of a structure.

* **Parameters:**
  **structure** – Pymatgen structure for which the energy is calculated.
* **Returns:**
  Electronic energy of the structure.

#### \_calculate_thermal_properties(structure: Structure) → dict

Helper to calculate the thermal properties of a structure.

* **Parameters:**
  **structure** – Pymatgen structure for which the thermal properties are calculated.
* **Returns:**
  Dictionary of thermal properties containing free energies, entropies and heat capacities.

#### \_collect_properties(structure: Structure) → tuple[list, list, list, list, list]

Helper to collect properties like volumes, electronic energies, and thermal properties.

* **Parameters:**
  **structure** – Pymatgen structure for which the properties need to be calculated.
* **Returns:**
  Tuple containing lists of volumes, electronic energies, free energies, entropies,
  : and heat capacities for different scale factors.

#### \_create_qha(volumes: list, electronic_energies: list, temperatures: list, free_energies: list, entropies: list, heat_capacities: list) → PhonopyQHA

Helper to create a PhonopyQHA object for quasi-harmonic approximation.

* **Parameters:**
  * **volumes** – List of volumes corresponding to different scale factors.
  * **electronic_energies** – List of electronic energies corresponding to different volumes.
  * **temperatures** – List of temperatures in ascending order (in Kelvin).
  * **free_energies** – List of free energies corresponding to different volumes and temperatures.
  * **entropies** – List of entropies corresponding to different volumes and temperatures.
  * **heat_capacities** – List of heat capacities corresponding to different volumes and temperatures.
* **Returns:**
  Phonopy.qha object.

#### \_generate_output_dict(qha: PhonopyQHA, volumes: list, electronic_energies: list, temperatures: list) → dict

Helper to generate the output dictionary after QHA calculation.

* **Parameters:**
  * **qha** – Phonopy.qha object.
  * **volumes** – List of volumes corresponding to different scale factors.
  * **electronic_energies** – List of electronic energies corresponding to different volumes.
  * **temperatures** – List of temperatures in ascending order (in Kelvin).
* **Returns:**
  Dictionary containing the results of QHA calculation.

#### \_scale_structure(structure: Structure, scale_factor: float) → Structure

Helper to scale the lattice of a structure.

* **Parameters:**
  * **structure** – Pymatgen structure to be scaled.
  * **scale_factor** – Factor by which the lattice constants are scaled.
* **Returns:**
  Pymatgen structure with scaled lattice constants.

#### \_write_output_files(qha: PhonopyQHA) → None

Helper to write various output files based on the QHA calculation.

* **Parameters:**
  **qha** – Phonopy.qha object

#### calc(structure: Structure | dict[str, Any]) → dict

Calculates thermal properties of Pymatgen structure with phonopy under quasi-harmonic approximation.

* **Parameters:**
  **structure** – Pymatgen structure.

Returns:
{

> “qha”: Phonopy.qha object,
> “scale_factors”: List of scale factors of lattice constants,
> “volumes”: List of unit cell volumes at corresponding scale factors (in Angstrom^3),
> “electronic_energies”: List of electronic energies at corresponding volumes (in eV),
> “temperatures”: List of temperatures in ascending order (in Kelvin),
> “thermal_expansion_coefficients”: List of volumetric thermal expansion coefficients at corresponding

> > temperatures (in Kelvin^-1),

> “gibbs_free_energies”: List of Gibbs free energies at corresponding temperatures (in eV),
> “bulk_modulus_P”: List of bulk modulus at constant pressure at corresponding temperatures (in GPa),
> “heat_capacity_P”: List of heat capacities at constant pressure at corresponding temperatures (in J/K/mol),
> “gruneisen_parameters”: List of Gruneisen parameters at corresponding temperatures,

}

#### calculator *: Calculator*

#### eos *: str* *= 'vinet'*

#### fmax *: float* *= 0.1*

#### optimizer *: str* *= 'FIRE'*

#### phonon_calc_kwargs *: dict | None* *= None*

#### relax_calc_kwargs *: dict | None* *= None*

#### relax_structure *: bool* *= True*

#### scale_factors *: Sequence[float]* *= (0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05)*

#### t_max *: float* *= 1000*

#### t_min *: float* *= 0*

#### t_step *: float* *= 10*

#### write_bulk_modulus_temperature *: bool | str | Path* *= False*

#### write_gibbs_temperature *: bool | str | Path* *= False*

#### write_gruneisen_temperature *: bool | str | Path* *= False*

#### write_heat_capacity_p_numerical *: bool | str | Path* *= False*

#### write_heat_capacity_p_polyfit *: bool | str | Path* *= False*

#### write_helmholtz_volume *: bool | str | Path* *= False*

#### write_thermal_expansion *: bool | str | Path* *= False*

#### write_volume_temperature *: bool | str | Path* *= False*

## matcalc._relaxation module

Relaxation properties.

### *class* RelaxCalc(calculator: Calculator, \*, optimizer: Optimizer | str = 'FIRE', max_steps: int = 500, traj_file: str | None = None, interval: int = 1, fmax: float = 0.1, relax_atoms: bool = True, relax_cell: bool = True, cell_filter: Filter = <class 'ase.filters.FrechetCellFilter'>, perturb_distance: float | None = None)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

A class to perform structural relaxation calculations using ASE (Atomic Simulation Environment).

This class facilitates structural relaxation by integrating with ASE tools for
optimized geometry and/or cell parameters. It enables convergence on forces, stress,
and total energy, offering customization for relaxation parameters and further
enabling properties to be extracted from the relaxed structure.

* **Variables:**
  * **calculator** – ASE Calculator used for force and energy evaluations.
  * **optimizer** – Algorithm for performing the optimization.
  * **max_steps** – Maximum number of optimization steps allowed.
  * **traj_file** – Path to save relaxation trajectory (optional).
  * **interval** – Interval for saving trajectory frames during relaxation.
  * **fmax** – Force tolerance for convergence (eV/Å).
  * **relax_atoms** – Whether atomic positions are relaxed.
  * **relax_cell** – Whether the cell parameters are relaxed.
  * **cell_filter** – ASE filter used for modifying the cell during relaxation.
  * **perturb_distance** – Distance (Å) for random perturbation to break symmetry.

Initializes the relaxation procedure for an atomic configuration system.

This constructor sets up the relaxation pipeline, configuring the required
calculator, optimizer, relaxation parameters, and logging options. The
relaxation process aims to find the minimum energy configuration, optionally
relaxing atoms and/or the simulation cell within the specified constraints.

* **Parameters:**
  * **calculator** – A calculator object used to perform energy and force
    calculations during the relaxation process.
  * **optimizer** – The optimization algorithm to use for relaxation. It can
    either be an instance of an Optimizer class or a string identifier for
    a recognized ASE optimizer. Defaults to “FIRE”.
  * **max_steps** – The maximum number of optimization steps to perform
    during the relaxation process. Defaults to 500.
  * **traj_file** – Path to a file for periodic trajectory output (if specified).
    This file logs the atomic positions and cell configurations after a given
    interval. Defaults to None.
  * **interval** – The interval (in steps) at which the trajectory file is
    updated. Defaults to 1.
  * **fmax** – The force convergence threshold. Relaxation continues until the
    maximum force on any atom falls below this value. Defaults to 0.1.
  * **relax_atoms** – A flag indicating whether the atomic positions are to
    be relaxed. Defaults to True.
  * **relax_cell** – A flag indicating whether the simulation cell is to
    be relaxed. Defaults to True.
  * **cell_filter** – The filter to apply when relaxing the simulation cell.
    This determines constraints or allowed degrees of freedom during
    cell relaxation. Defaults to FrechetCellFilter.
  * **perturb_distance** – A perturbation distance used for initializing
    the system configuration before relaxation. If None, no perturbation
    is applied. Defaults to None.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure | dict) → dict

Calculate the final relaxed structure, energy, forces, and stress for a given
structure and update the result dictionary with additional geometric properties.

This method takes an input structure and performs a relaxation process
depending on the specified parameters. If the perturb_distance attribute
is provided, the structure is perturbed before relaxation. The relaxation
process can involve optimization of both atomic positions and the unit cell
if specified. Results of the calculation including final structure geometry,
energy, forces, stresses, and lattice parameters are returned in a dictionary.

* **Parameters:**
  **structure** – Input structure for calculation. Can be provided as a
  Structure object or a dictionary convertible to Structure.
* **Returns:**
  Dictionary containing the final relaxed structure, calculated
  energy, forces, stress, and lattice parameters.
* **Return type:**
  dict

### *class* TrajectoryObserver(atoms: Atoms, energies: list[float] = <factory>, forces: list[np.ndarray] = <factory>, stresses: list[np.ndarray] = <factory>, atom_positions: list[np.ndarray] = <factory>, cells: list[np.ndarray] = <factory>)

Bases: `object`

Class for observing and recording the properties of an atomic structure during relaxation.

The TrajectoryObserver class is designed to track and store the atomic properties like
energies, forces, stresses, atom positions, and cell structure of an atomic system
represented by an Atoms object. It provides functionality to save recorded data
to a file for further analysis or usage.

* **Variables:**
  * **atoms** – The atomic structure being observed.
  * **energies** – List of potential energy values of the atoms during relaxation.
  * **forces** – List of force arrays recorded for the atoms during relaxation.
  * **stresses** – List of stress tensors recorded for the atoms during relaxation.
  * **atom_positions** – List of atomic positions recorded during relaxation.
  * **cells** – List of unit cell arrays recorded during relaxation.

#### atom_positions *: list[np.ndarray]*

#### atoms *: Atoms*

#### cells *: list[np.ndarray]*

#### energies *: list[float]*

#### forces *: list[np.ndarray]*

#### save(filename: str) → None

Save the trajectory to file.

* **Parameters:**
  **filename** (*str*) – filename to save the trajectory.

#### stresses *: list[np.ndarray]*

## matcalc._stability module

Calculator for stability related properties.

### *class* EnergeticsCalc(calculator: Calculator, \*, elemental_refs: Literal['MatPES-PBE', 'MatPES-r2SCAN'] | dict = 'MatPES-PBE', use_dft_gs_reference: bool = False, relax_structure: bool = True, relax_calc_kwargs: dict | None = None)

Bases: [`PropCalc`](#matcalc._base.PropCalc)

Handles the computation of energetic properties such as formation energy per atom,
cohesive energy per atom, and relaxed structures for input compositions. This class
enables a streamlined setup for performing computational property calculations based
on different reference data and relaxation configurations.

* **Variables:**
  * **calculator** – The computational calculator used for numerical simulations and property
    evaluations.
  * **elemental_refs** – Reference data dictionary or identifier for elemental properties.
    If a string (“MatPES-PBE” or “MatPES-r2SCAN”), loads default references;
    if a dictionary, uses custom provided data.
  * **use_dft_gs_reference** – Whether to use DFT ground state data for energy computations
    when referencing elemental properties.
  * **relax_structure** – Specifies whether to relax the input structures before property
    calculations. If True, relaxation is applied.
  * **relax_calc_kwargs** – Optional keyword arguments for fine-tuning relaxation calculation
    settings or parameters.

Initializes the class with the given calculator and optional configurations for
elemental references, density functional theory (DFT) ground state reference, and
options for structural relaxation.

This constructor allows initializing essential components of the object, tailored
for specific computational settings. The parameters include configurations for
elemental references, an optional DFT ground state reference, and structural
relaxation preferences.

* **Parameters:**
  * **calculator** (*Calculator*) – A Calculator instance for performing calculations.
  * **elemental_refs** (*Literal* *[* *"MatPES-PBE"* *,*  *"MatPES-r2SCAN"* *]*  *|* *dict*) – Specifies the elemental references to be used. It can either be
    a predefined string identifier (“MatPES-PBE”, “MatPES-r2SCAN”) or a dictionary
    mapping elements to their energy references.
  * **use_dft_gs_reference** (*bool*) – Determines whether to use DFT ground state
    energy as a reference. Defaults to False.
  * **relax_structure** (*bool*) – Specifies if the structure should be relaxed before
    proceeding with calculations. Defaults to True.
  * **relax_calc_kwargs** (*dict* *|* *None*) – Additional keyword arguments for the relaxation
    calculation. Can be a dictionary of settings or None. Defaults to None.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure | dict[str, Any]) → dict[str, Any]

Calculates the formation energy per atom, cohesive energy per atom, and final
relaxed structure for a given input structure using a relaxation calculation
and reference elemental data. This function also optionally utilizes DFT
ground-state references for formation energy calculations. The cohesive energy is always referenced to the
DFT atomic energies.

* **Parameters:**
  **structure** (*Structure*) – The input structure to be relaxed and analyzed.
* **Returns:**
  A dictionary containing the formation energy per atom, cohesive
  energy per atom, and the final relaxed structure.
* **Return type:**
  dict[str, Any]

## matcalc.benchmark module

This module implements classes for running benchmarks on materials properties.

### *class* Benchmark(benchmark_name: str | Path, properties: Sequence[str], index_name: str, other_fields: tuple = (), property_rename_map: dict[str, str] | None = None, suffix_ground_truth: str = 'DFT', n_samples: int | None = None, seed: int = 42, \*\*kwargs)

Bases: `object`

Represents an abstract base class for benchmarking elasticity properties of materials.

This class provides functionality to process benchmark elemental_refs, create a DataFrame for analysis,
and run calculations using a specified potential energy surface (PES) calculator. It is designed
to facilitate benchmarking of bulk and shear moduli against pre-defined ground truth elemental_refs.

* **Variables:**
  * **properties** – List of properties to extract and benchmark. These properties
    are key inputs for analysis tasks.
  * **other_fields** – Tuple of additional fields in the benchmark entries to
    include in the processed elemental_refs. Useful for metadata or optional attributes.
  * **index_name** – Name of the index field in the benchmark dataset. This is used
    as the primary key for identifying entries.
  * **structures** – List of structures extracted from the benchmark entries. Structures
    are objects describing material geometries stored in the dataset.
  * **kwargs** – Additional keywords passed through to the ElasticityCalculator or associated
    processes for extended configuration.
  * **ground_truth** – DataFrame containing the processed benchmark elemental_refs, including ground truth
    reference values for materials properties.

Initializes an instance for processing benchmark elemental_refs and constructing a DataFrame
representing the ground truth properties of input structures. Additionally, stores
information about input structures and other auxiliary elemental_refs for further usage.

* **Parameters:**
  * **benchmark_name** (*str* *|* *Path*) – The name of the benchmark dataset or a path to a file containing
    the benchmark entries.
  * **properties** (*list* *[**str* *]*) – A list of property names to extract.
  * **index_name** (*str*) – The name of the field used as the index for the resulting DataFrame (typically a unique id
    like mp_id).
  * **other_fields** (*tuple* *[**str* *]*) – Additional fields to include in the DataFrame, default is an empty tuple. Useful ones
    are for example formula or metadata.
  * **property_rename_map** (*dict* *|* *None*) – A dict used to rename the properties for easier reading.
  * **suffix_ground_truth** (*str*) – The suffix added to the property names in the DataFrame for
    distinguishing ground truth values, default is “DFT”.
  * **n_samples** (*int* *|* *None*) – Number of samples to randomly select from the benchmark dataset, or
    `None` to include all samples, default is `None`.
  * **seed** (*int*) – Seed value for random sampling of entries (if n_samples is specified), default is `42`.
  * **kwargs** (*dict*) – Additional keyword arguments for configuring the PropCalc..
* **Raises:**
  * **FileNotFoundError** – If the provided benchmark_name is a path that does not exist.
  * **ValueError** – If invalid or incomplete elemental_refs is encountered in the benchmark entries.

#### \_abc_impl *= <_abc._abc_data object>*

#### *abstract* get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc._base.PropCalc)

Abstract method to retrieve a property calculation object using the provided calculator and additional
parameters.
This method must be implemented by subclasses and will utilize the provided calculator to create a
PropCalc instance, possibly influenced by additional keyword arguments.

* **Parameters:**
  * **calculator** (*Calculator*) – The calculator instance to be used for generating the property calculation.
  * **kwargs** (*dict*) – Additional keyword arguments that can influence the property calculation process.
* **Returns:**
  An instance of PropCalc representing the property calculation result.
* **Return type:**
  [PropCalc](#matcalc._base.PropCalc)

#### *abstract* process_result(result: dict | None, model_name: str) → dict

Implements post-processing of results. A default implementation is provided that simply appends the model name
as a suffix to the key of the input dictionary for all properties. Subclasses can override this method to
provide more sophisticated processing.

* **Parameters:**
  * **result** (*dict*) – Input dictionary containing key-value pairs to be processed.
  * **model_name** (*str*) – The name of the model to append to each key as a suffix.
* **Returns:**
  A new dictionary with modified keys based on the model name suffix.
* **Return type:**
  dict

#### run(calculator: Calculator, model_name: str, \*, n_jobs: None | int = -1, checkpoint_file: str | Path | None = None, checkpoint_freq: int = 1000, delete_checkpoint_on_finish: bool = True, include_full_results: bool = False, \*\*kwargs) → pd.DataFrame

Processes a collection of structures using a calculator, saves intermittent
checkpoints, and returns the results in a DataFrame. This function supports
parallel computation and allows for error tolerance during processing.

The function also retrieves a property calculator and utilizes it to calculate
desired results for the given set of structures. Checkpoints are saved
periodically based on the specified frequency, ensuring that progress is not
lost in case of interruptions.

* **Parameters:**
  * **calculator** (*Calculator*) – ASE-compatible calculator instance used to provide PES information for PropCalc.
  * **model_name** (*str*) – Name of the model used for properties’ calculation.
    This name is updated in the results DataFrame.
  * **n_jobs** (*int* *|* *None*) – Number of parallel jobs to be used in the computation. Use -1
    to allocate all cores available on the system. Defaults to -1.
  * **checkpoint_file** (*str* *|* *Path* *|* *None*) – File path where checkpoint elemental_refs is saved periodically.
    If None, no checkpoints are saved.
  * **checkpoint_freq** (*int*) – Frequency after which checkpoint elemental_refs is saved.
    Corresponds to the number of structures processed.
  * **delete_checkpoint_on_finish** (*bool*) – Whether to delete checkpoint files when the benchmark finishes. Defaults to
    True.
  * **include_full_results** (*bool*) – Whether to save full results from PropCalc.calc for analysis afterwards. For
    instance, the ElasticityProp does not just compute the bulk and shear moduli, but also the full elastic
    tensors, which can be used for other kinds of analysis. Defaults to False.
  * **kwargs** (*dict*) – Additional keyword arguments passed to the property calculator,
    for instance, to customize its behavior or computation options.
* **Returns:**
  A pandas DataFrame containing the processed results for the given
  input structures. The DataFrame includes updated results and relevant
  metrics.
* **Return type:**
  pd.DataFrame

### *class* BenchmarkSuite(benchmarks: list)

Bases: `object`

Represents a suite for handling and executing a list of benchmarks. This class is designed
for the comprehensive execution and management of benchmarks with support for configurable
parallel computation and checkpointing.

The purpose of this class is to facilitate the execution of multiple benchmarks using
various computational models (calculators) while enabling efficient resource utilization
and result aggregation. It supports checkpointing to handle long computations reliably.

* **Variables:**
  **benchmarks** – A list of benchmarks to be configured or evaluated.

Represents a collection of benchmarks.

This class is designed to store and manage a list of benchmarks. It provides
an initialization method to set up the benchmark list during object creation.
It does not include any specialized methods or functionality beyond holding
a list of benchmarks.

#### benchmarks

A list of benchmarks provided during initialization.

* **Type:**
  list

#### run(calculators: dict[str, Calculator], \*, n_jobs: int | None = -1, checkpoint_freq: int = 1000, delete_checkpoint_on_finish: bool = True) → list[pd.DataFrame]

Executes benchmarks using the provided calculators and combines the results into a
list of dataframes. Each benchmark runs for all models provided by calculators, collecting
individual results and performing validations during elemental_refs combination.

* **Parameters:**
  * **calculators** – A dictionary where the keys are the model names (str)
    and the values are the corresponding calculator instances (Calculator).
  * **n_jobs** – The maximum number of concurrent jobs to run. If set to -1,
    utilizes all available processors. Defaults to -1.
  * **checkpoint_freq** – The frequency at which progress is saved as checkpoints,
    in terms of calculation steps. Defaults to 1000.
  * **delete_checkpoint_on_finish** (*bool*) – Whether to delete checkpoint files when the benchmark finishes. Defaults to
    True.
* **Returns:**
  A list of pandas DataFrames, each containing combined results
  for all calculators across the benchmarks.

### *class* CheckpointFile(path: str | Path)

Bases: `object`

Represents a checkpoint file system management utility.

This class provides mechanisms to manage and process a file path and its
associated actions such as loading and saving data. It ensures standardized
path handling through the use of Path objects, enables loading checkpoint
data from a file, and facilitates the saving of resulting data.

* **Variables:**
  **path** – Standardized file system path, managed as a Path object.

Represents an initialization process for handling a filesystem path. The
provided path is converted into a Path object for standardized path
management in the application.

* **Parameters:**
  **path** – The filesystem path to be managed. Can be provided as a
  string or as a Path object.

#### load(\*args: list) → tuple

Loads elemental_refs from a specified path if it exists, returning the loaded elemental_refs along with
remaining portions of the given input arguments.

The method checks if the file path exists, and if so, it loads elemental_refs from the specified
file using a predefined loadfn function. It logs the number of loaded entries and
returns the successfully loaded elemental_refs alongside sliced input arguments based on the
number of loaded entries. If the file path does not exist, it returns empty results
and the original input arguments unchanged.

* **Parameters:**
  **args** – List of lists where each list corresponds to additional elemental_refs to
  process in conjunction with the loaded file content.
* **Returns:**
  A tuple where the first element is the loaded elemental_refs (list) from the specified
  file path (or an empty list if the path does not exist), and subsequent elements
  are the remaining unsliced portions of each input list from args or the entire
  original lists if nothing was loaded.

#### save(results: list[dict[str, Any]]) → None

Saves a list of results at the specified checkpoint location.

* **Parameters:**
  **results** – A list of dictionaries or objects to be saved.
* **Returns:**
  None

### *class* ElasticityBenchmark(index_name: str = 'mp_id', benchmark_name: str | Path = 'mp-binary-pbe-elasticity-2025.1.json.gz', \*\*kwargs)

Bases: [`Benchmark`](#matcalc.benchmark.Benchmark)

Represents a benchmark for evaluating and analyzing mechanical properties such as
bulk modulus and shear modulus for various materials. The benchmark primarily utilizes
a dataset and provides functionality for property calculation and result processing.

The class is designed to work with a predefined framework for benchmarking mechanical
properties. The benchmark dataset contains values such as bulk modulus and shear
modulus along with additional metadata. This class supports configurability through
metadata files, index names, and additional benchmark properties. It relies on
external calculators and utility classes for property computations and result handling.

Initializes the ElasticityBenchmark instance by taking benchmark metadata and
additional configuration parameters. Sets up the benchmark framework with
specified mechanical properties and metadata.

* **Parameters:**
  * **index_name** (*str*) – The name of the index used to uniquely identify records
    in the benchmark dataset.
  * **benchmark_name** (*str* *|* *Path*) – The path or name of the benchmark file that contains
    the dataset. Can either be a string or a Path object.
  * **kwargs** (*dict*) – Additional keyword arguments that may be passed to parent
    class methods or used for customization.

#### \_abc_impl *= <_abc._abc_data object>*

#### get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc._base.PropCalc)

Calculates and returns a property calculation object based on the provided
calculator and optional parameters. This is useful for initializing and
configuring a property calculation.

* **Parameters:**
  * **calculator** – A Calculator object responsible for performing numerical
    operations required for property calculations.
  * **kwargs** – Additional keyword arguments used for configuring the property
    calculation.
* **Returns:**
  An initialized PropCalc object configured based on the specified
  calculator and keyword arguments.
* **Return type:**
  [PropCalc](#matcalc._base.PropCalc)

#### process_result(result: dict | None, model_name: str) → dict

Processes the result dictionary containing bulk and shear modulus values, adjusts
them by multiplying with a predefined conversion factor, and formats the keys
according to the provided model name. If the result is None, default values of
NaN are returned for both bulk and shear modulus.

* **Parameters:**
  * **result** (*dict* *or* *None*) – A dictionary containing the bulk and shear modulus values under the keys
    ‘bulk_modulus_vrh’ and ‘shear_modulus_vrh’ respectively. It can also be
    None to indicate missing elemental_refs.
  * **model_name** (*str*) – A string representing the identifier or name of the model. It will be used
    to format the returned dictionary’s keys.
* **Returns:**
  A dictionary containing two entries. The keys will be dynamically created
  by appending the model name to the terms ‘

  ```
  bulk_modulus_vrh_
  ```

  ’ and
  ‘

  ```
  shear_modulus_vrh_
  ```

  ’. The values will either be scaled modulus values or
  NaN if the input result is None.
* **Return type:**
  dict

### *class* EquilibriumBenchmark(index_name: str = 'material_id', benchmark_name: str | Path = 'wbm-random-pbe54-equilibrium-2025.1.json.gz', folder_name: str = 'default_folder', \*\*kwargs)

Bases: [`Benchmark`](#matcalc.benchmark.Benchmark)

Represents a benchmark for evaluating and analyzing equilibrium properties of materials.
This benchmark utilizes a dataset and provides functionality for property calculation
and result processing. The class is designed to work with a predefined framework for
benchmarking equilibrium properties. The benchmark dataset contains data such as relaxed
structures, un-/corrected formation energy along with additional metadata. This class
supports configurability through metadata files, index names, and additional benchmark
properties. It relies on external calculators and utility classes for property computations
and result handling.

Initializes the EquilibriumBenchmark instance with specified benchmark metadata and
configuration parameters. It sets up the benchmark with the necessary properties
required for equilibrium benchmark analysis.

* **Parameters:**
  * **index_name** (*str*) – The name of the index used to uniquely identify records in the dataset.
  * **benchmark_name** (*str* *|* *Path*) – The path or name of the benchmark file that contains the dataset.
  * **folder_name** (*str*) – The folder name used for file operations related to structure files.
  * **kwargs** (*dict*) – Additional keyword arguments for customization.

#### \_abc_impl *= <_abc._abc_data object>*

#### get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc._base.PropCalc)

Returns a property calculation object for performing relaxation and formation energy
calculations. This method initializes the stability calculator using the provided
Calculator object and any additional configuration parameters.

* **Parameters:**
  * **calculator** (*Calculator*) – A Calculator object responsible for performing the relaxation and
    formation energy calculation.
  * **kwargs** (*dict*) – Additional keyword arguments used for configuration.
* **Returns:**
  An initialized PropCalc object configured for relaxation and formation energy
  calculations.
* **Return type:**
  [PropCalc](#matcalc._base.PropCalc)

#### process_result(result: dict | None, model_name: str) → dict

Processes the result dictionary containing final structures and formation energy per atom,
formats the keys according to the provided model name. If the result is None, default values
of NaN are returned for final structures or formation energy per atom.

* **Parameters:**
  * **result** (*dict* *or* *None*) – A dictionary containing the final structures and formation energy per atom under the keys
    ‘final_structure’ and ‘formation energy per atom’. It can also be None to indicate missing
    elemental_refs.
  * **model_name** (*str*) – A string representing the identifier or name of the model. It will be used
    to format the returned dictionary’s keys.
* **Returns:**
  A dictionary containing the specific final structure and formation energy per atomprefixed
  by the model name. The values will be NaN if the input result is None.
* **Return type:**
  dict

#### run(calculator: Calculator, model_name: str, \*, n_jobs: None | int = -1, checkpoint_file: str | Path | None = None, checkpoint_freq: int = 1000, delete_checkpoint_on_finish: bool = True, include_full_results: bool = False, \*\*kwargs) → pd.DataFrame

Processes a collection of structures using a calculator, saves intermittent checkpoints,
and returns the results in a DataFrame. In addition to the base processing performed
by the parent class, this method computes the Euclidean distance between the relaxed
structure (obtained from the property calculation) and the reference DFT structure,
using SiteStatsFingerprint. The computed distance is added as a new column in the
results DataFrame with the key “distance_{model_name}”.

This function supports parallel computation and allows for error tolerance during processing.
It retrieves a property calculator and utilizes it to calculate desired results for the given
set of structures. Checkpoints are saved periodically based on the specified frequency,
ensuring that progress is not lost in case of interruptions.

* **Parameters:**
  * **calculator** (*Calculator*) – ASE-compatible calculator instance used to provide PES information for PropCalc.
  * **model_name** (*str*) – Name of the model used for properties’ calculation.
    This name is updated in the results DataFrame.
  * **n_jobs** (*int* *|* *None*) – Number of parallel jobs to be used in the computation. Use -1
    to allocate all cores available on the system. Defaults to -1.
  * **checkpoint_file** (*str* *|* *Path* *|* *None*) – File path where checkpoint elemental_refs is saved periodically.
    If None, no checkpoints are saved.
  * **checkpoint_freq** (*int*) – Frequency after which checkpoint elemental_refs is saved.
    Corresponds to the number of structures processed.
  * **delete_checkpoint_on_finish** (*bool*) – Whether to delete checkpoint files when the benchmark finishes. Defaults to
    True.
  * **include_full_results** (*bool*) – Whether to save full results from PropCalc.calc for analysis afterwards. For
    instance, the ElasticityProp does not just compute the bulk and shear moduli, but also the full elastic
    tensors, which can be used for other kinds of analysis. Defaults to False.
  * **kwargs** (*dict*) – Additional keyword arguments passed to the property calculator,
    for instance, to customize its behavior or computation options.
* **Returns:**
  A pandas DataFrame containing the processed results for the given
  input structures. The DataFrame includes updated results and relevant
  metrics.
* **Return type:**
  pd.DataFrame

### *class* PhononBenchmark(index_name: str = 'mp_id', benchmark_name: str | Path = 'alexandria-binary-pbe-phonon-2025.1.json.gz', \*\*kwargs)

Bases: [`Benchmark`](#matcalc.benchmark.Benchmark)

Manages phonon benchmarking tasks, such as initializing benchmark elemental_refs,
performing calculations, and processing results.

This class facilitates constructing and managing phonon benchmarks based on
provided elemental_refs. It supports operations for processing benchmark elemental_refs,
extracting relevant attributes, and computing thermal properties. It is
compatible with various calculators and is designed to streamline the
benchmarking process for materials’ phonon-related properties.

Initializes an instance with specified index and benchmark details.

This constructor sets up an object with predefined properties such as heat
capacity and additional fields such as the formula. It supports customizations
via keyword arguments for further configurations.

* **Parameters:**
  * **index_name** – The name of the index to be used for identification in
    the dataset.
  * **benchmark_name** – The benchmark file name or path containing
    the dataset information in JSON or compressed format.
  * **kwargs** – Additional optional parameters for configuration.

#### \_abc_impl *= <_abc._abc_data object>*

#### get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc._base.PropCalc)

Retrieves a phonon calculation instance based on the given calculator and
additional keyword arguments.

This function initializes and returns a PhononCalc object using the provided
calculator instance and any optional keyword arguments to configure the
calculation further.

* **Parameters:**
  * **calculator** – The calculator instance used to perform the phonon
    calculation. Must be an instance of the Calculator class.
  * **kwargs** – Additional keyword arguments for configuring the resulting
    PhononCalc instance.
* **Returns:**
  A new PhononCalc object, initialized with the input calculator and
  optional parameters.
* **Return type:**
  [PropCalc](#matcalc._base.PropCalc)

#### process_result(result: dict | None, model_name: str) → dict

Processes the result dictionary to extract specific thermal property information for the provided model name.

* **Parameters:**
  * **result** (*dict*) – Dictionary containing thermal properties, with keys structured to access relevant elemental_refs
    like “thermal_properties” and “heat_capacity”.
  * **model_name** (*str*) – The model name used as a prefix in returned result keys.
* **Returns:**
  A dictionary containing the specific heat capacity at a particular index (e.g., 30),
  prefixed by the model name.
* **Return type:**
  dict

### *class* SofteningBenchmark(benchmark_name: str | Path = 'wbm-high-energy-states.json.gz', index_name: str = 'wbm_id', n_samples: int | None = None, seed: int = 42, \*\*kwargs)

Bases: `object`

A benchmark for the systematic softening of a PES, as described in:
: B. Deng, et al. npj Comput. Mater. 11, 9 (2025).
  doi: 10.1038/s41524-024-01500-6

The dataset used here can be found in figshare through:
: [https://figshare.com/articles/dataset/WBM_high_energy_states/27307776?file=50005317](https://figshare.com/articles/dataset/WBM_high_energy_states/27307776?file=50005317)

This benchmark essentially performs static calculation on pre-sampled high-energy
PES configurations, and then compare the systematic underestimation of forces
predicted between GGA-DFT and the provided force field.

Initializes an instance with specified index and benchmark details.

* **Parameters:**
  * **index_name** – The name of the index to be used for identification in
    the dataset.
  * **benchmark_name** – The benchmark file name or path containing
    the dataset information in JSON or compressed format.
  * **kwargs** – Additional optional parameters for configuration.

#### *static* get_linear_fitted_slope(x: list | ndarray, y: list | ndarray) → float

Return the linearly fitted slope of x and y using a simple linear model (y = ax).
:param x: A list of the x values.
:param y: A list of the y values.
:return: A float of the fitted slope.

#### run(calculator: Calculator, model_name: str, checkpoint_file: str | Path | None = None, checkpoint_freq: int = 10, \*, include_full_results: bool = False) → pd.DataFrame

Process all the material ids by
1. calculate the forces on all the sampled structures.
2. perform a linear fit on the predicted forces w.r.t. provided DFT forces.
3. returning the fitted slopes as the softening scales.

* **Parameters:**
  * **calculator** (*Calculator*) – The ASE-compatible calculator instance
  * **model_name** (*str*) – Name of the model used for properties’ calculation.
    This name is updated in the results DataFrame.
  * **checkpoint_file** (*str* *|* *Path* *|* *None*) – File path where checkpoint is saved periodically.
    If None, no checkpoints are saved.
  * **checkpoint_freq** (*int*) – Frequency after which checkpoint is saved.
    Corresponds to the number of structures processed.
  * **include_full_results** (*bool*) – Whether to include the raw force prediction in the
    returned dataframe
* **Returns:**
  A dataframe containing the softening scales.
* **Return type:**
  pd.DataFrame

### get_available_benchmarks() → list[str]

Fetches and returns a list of available benchmarks.

This function makes a request to a predefined URL to retrieve benchmark
data. It then filters and extracts the names of benchmarks that end with
the ‘.json.gz’ extension.

* **Returns:**
  A list of benchmark names available in the retrieved data.
* **Return type:**
  list[str]

### get_benchmark_data(name: str) → DataFrame

Retrieve benchmark data as a Pandas DataFrame. Uses fsspec to cache files locally if possible.

* **Parameters:**
  **name** (*str*) – Name of the benchmark elemental_refs file to be retrieved
* **Returns:**
  Benchmark elemental_refs loaded as a Pandas DataFrame
* **Return type:**
  pd.DataFrame
* **Raises:**
  **requests.RequestException** – If the benchmark elemental_refs file cannot be
  downloaded from the specified URL

## matcalc.config module

Sets some configuration global variables and locations for matcalc.

### clear_cache(\*, confirm: bool = True) → None

Deletes all files and subdirectories within the benchmark data directory,
effectively clearing the cache. The user is prompted for confirmation
before proceeding with the deletion to prevent accidental data loss.

* **Parameters:**
  **confirm** – A flag to bypass the confirmation prompt. If set to True,
  the function will prompt the user for confirmation. If set to False,
  the deletion will proceed without additional confirmation. Defaults to
  True.
* **Returns:**
  Returns None.

## matcalc.units module

Useful constants for unit conversions.

## matcalc.utils module

Some utility methods, e.g., for getting calculators from well-known sources.

### *class* PESCalculator(potential: LMPStaticCalculator, stress_unit: Literal['eV/A3', 'GPa'] = 'GPa', stress_weight: float = 1.0, \*\*kwargs: Any)

Bases: `Calculator`

Class for simulating and calculating potential energy surfaces (PES) using various
machine learning and classical potentials. It extends the ASE Calculator API,
allowing integration with the ASE framework for molecular dynamics and structure
optimization.

PESCalculator provides methods to perform energy, force, and stress calculations
using potentials such as MTP, GAP, NNP, SNAP, ACE, NequIP, DeepMD and MatGL (M3GNet, TensorNet, CHGNet). The class
includes utilities to load compatible models for each potential type, making it
a versatile tool for materials modeling and molecular simulations.

* **Variables:**
  * **potential** – The potential model used for PES calculations.
  * **stress_weight** – The stress weight factor to convert between units.

Initialize PESCalculator with a potential from maml.

* **Parameters:**
  * **potential** (*LMPStaticCalculator*) – maml.apps.pes._lammps.LMPStaticCalculator
  * **stress_unit** (*str*) – The unit of stress. Default to “GPa”
  * **stress_weight** (*float*) – The conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa.
    Default to 1.0.
  * **\*\*kwargs** – Additional keyword arguments passed to super()._\_init_\_().

#### \_abc_impl *= <_abc._abc_data object>*

#### calculate(atoms: Atoms | None = None, properties: list | None = None, system_changes: list | None = None) → None

Perform calculation for an input Atoms.

* **Parameters:**
  * **atoms** (*ase.Atoms*) – ase Atoms object
  * **properties** (*list*) – The list of properties to calculate
  * **system_changes** (*list*) – monitor which properties of atoms were
    changed for new calculation. If not, the previous calculation
    results will be loaded.

#### implemented_properties *: List[str]* *= ['energy', 'forces', 'stress']*

Properties calculator can handle (energy, forces, …)

#### *static* load_ace(basis_set: str | Path | ACEBBasisSet | ACECTildeBasisSet | BBasisConfiguration, \*\*kwargs: Any) → Calculator

Load an ACE (Atomic Cluster Expansion) calculator using the specified basis set.

This method utilizes the PyACE library to create and initialize a PyACECalculator
instance with a given basis set. The provided basis set can take various forms including
file paths, basis set objects, or configurations. Additional customization options
can be passed through keyword arguments.

* **Parameters:**
  * **basis_set** – The basis set used for initializing the ACE calculator. This can
    be provided as a string, Path object, ACEBBasisSet, ACECTildeBasisSet, or
    BBasisConfiguration.
  * **kwargs** – Additional configuration parameters to customize the ACE
    calculator. These keyword arguments are passed directly to the PyACECalculator
    instance during initialization.
* **Returns:**
  An instance of the Calculator class representing the initialized ACE
  calculator.

#### *static* load_deepmd(model_path: str | Path, \*\*kwargs: Any) → Calculator

Loads a Deep Potential Molecular Dynamics (DeepMD) model and returns a Calculator
object for molecular dynamics simulations.

This method imports the deepmd.calculator.DP class and initializes it with the
given model path and optional keyword arguments. The resulting Calculator object
is used to perform molecular simulations based on the specified DeepMD model.

The function requires the DeepMD-kit library to be installed to properly import
and utilize the DP class.

* **Parameters:**
  * **model_path** – Path to the trained DeepMD model file, provided as a string
    or a Path object.
  * **kwargs** – Additional options and configurations to pass into the DeepMD
    Calculator during initialization.
* **Returns:**
  An instance of the Calculator object initialized with the specified
  DeepMD model and optional configurations.
* **Return type:**
  Calculator

#### *static* load_gap(filename: str | Path, \*\*kwargs: Any) → Calculator

Loads a Gaussian Approximation Potential (GAP) model from the given file and
returns a corresponding Calculator instance. GAP is a machine learning-based
potential used for atomistic simulations and requires a specific config file as
input. Any additional arguments for the calculator can be passed via kwargs,
allowing customization.

* **Parameters:**
  * **filename** (*str* *|* *Path*) – Path to the configuration file for the GAP model.
  * **kwargs** (*Any*) – Additional keyword arguments for configuring the calculator.
* **Returns:**
  An instance of PESCalculator initialized with the GAPotential model.
* **Return type:**
  Calculator

#### *static* load_matgl(path: str | Path, \*\*kwargs: Any) → Calculator

Loads a MATGL model from the specified path and initializes a PESCalculator
with the loaded model and additional optional parameters.

This method uses the MATGL library to load a model from the given file path
or directory. It then configures a calculator using the loaded model and
the provided keyword arguments.

* **Parameters:**
  * **path** (*str* *|* *Path*) – The path to the MATGL model file or directory.
  * **kwargs** – Additional keyword arguments used to configure the calculator.
* **Returns:**
  An instance of the PESCalculator initialized with the loaded MATGL
  model and configured with the given parameters.
* **Return type:**
  Calculator

#### *static* load_mtp(filename: str | Path, elements: list, \*\*kwargs: Any) → Calculator

Load a machine-learned potential (MTPotential) from a configuration file and
create a calculator object to interface with it.

This method initializes an instance of MTPotential using a provided
configuration file and elements. It returns a PESCalculator instance,
which wraps the initialized potential model.

* **Parameters:**
  * **filename** (*str* *|* *Path*) – Path to the configuration file for the MTPotential.
  * **elements** (*list*) – List of element symbols used in the model. Each element
    should be a string representing a chemical element (e.g., “H”, “O”).
  * **kwargs** (*Any*) – Additional keyword arguments to configure the PESCalculator.
* **Returns:**
  A calculator object wrapping the MTPotential.
* **Return type:**
  Calculator

#### *static* load_nequip(model_path: str | Path, \*\*kwargs: Any) → Calculator

Loads and returns a NequIP Calculator instance from the specified model path.
This method facilitates the integration of machine learning models into ASE
by loading a model for atomic-scale simulations.

* **Parameters:**
  * **model_path** (*str* *|* *Path*) – The file path to the serialized NequIP model.
  * **kwargs** (*Any*) – Additional keyword arguments to be passed to the
    NequIPCalculator.from_deployed_model method.
* **Returns:**
  A Calculator instance initialized with the given model and parameters,
  suitable for ASE simulations.
* **Return type:**
  Calculator

#### *static* load_nnp(input_filename: str | Path, scaling_filename: str | Path, weights_filenames: list, \*\*kwargs: Any) → Calculator

Loads a neural network potential (NNP) from specified configuration files and
creates a Calculator object configured with the potential. This function allows
for customizable keyword arguments to modify the behavior of the resulting
Calculator.

* **Parameters:**
  * **input_filename** (*str* *|* *Path*) – Path to the primary input file containing NNP configuration.
  * **scaling_filename** (*str* *|* *Path*) – Path to the scaling parameters file required for the NNP.
  * **weights_filenames** (*list*) – List of paths to weight files for the NNP.
  * **kwargs** (*Any*) – Additional keyword arguments passed to the Calculator constructor.
* **Returns:**
  A Calculator object initialized with the loaded NNP settings.
* **Return type:**
  Calculator

#### *static* load_snap(param_file: str | Path, coeff_file: str | Path, \*\*kwargs: Any) → Calculator

Load a SNAP (Spectral Neighbor Analysis Potential) configuration and create a
corresponding Calculator instance.

This static method initializes a SNAPotential instance using the provided
configuration files and subsequently generates a PESCalculator based on the
created potential model and additional keyword arguments.

* **Parameters:**
  * **param_file** – Path to the parameter file required for SNAPotential configuration.
  * **coeff_file** – Path to the coefficient file required for SNAPotential configuration.
  * **kwargs** – Additional keyword arguments passed to the PESCalculator.
* **Returns:**
  A PESCalculator instance configured with the SNAPotential model.
* **Return type:**
  Calculator

#### *static* load_universal(name: str | Calculator, \*\*kwargs: Any) → Calculator

Loads a calculator instance based on the provided name or an existing calculator object. The
method supports multiple pre-built universal models and aliases for ease of use. If an existing calculator
object is passed instead of a name, it will directly return that calculator instance. Supported UMLIPs
include SOTA potentials such as M3GNet, CHGNet, TensorNet, MACE, GRACE, SevenNet, ORB, etc.

This method is designed to provide a universal interface to load various calculator types, which
may belong to different domains and packages. It auto-resolves aliases, provides default options
for certain calculators, and raises errors for unsupported inputs.

* **Parameters:**
  * **name** – The name of the calculator to load or an instance of a Calculator.
  * **kwargs** – Keyword arguments that are passed to the internal calculator initialization routines
    for models matching the specified name. These options are calculator dependent.
* **Returns:**
  An instance of the loaded calculator.
* **Raises:**
  **ValueError** – If the name provided does not match any recognized calculator type.

### get_ase_optimizer(optimizer: str | Optimizer) → Optimizer

Retrieve an ASE optimizer instance based on the provided input. This function accepts either a
string representing the name of a valid ASE optimizer or an instance/subclass of the Optimizer
class. If a string is provided, it checks the validity of the optimizer name, and if valid, retrieves
the corresponding optimizer from ASE. An error is raised if the optimizer name is invalid.

If an Optimizer subclass or instance is provided as input, it is returned directly.

* **Parameters:**
  **optimizer** (*str* *|* *Optimizer*) – The optimizer to be retrieved. Can be a string representing a valid ASE
  optimizer name or an instance/subclass of the Optimizer class.
* **Returns:**
  The corresponding ASE optimizer instance or the input Optimizer instance/subclass.
* **Return type:**
  Optimizer
* **Raises:**
  **ValueError** – If the optimizer name provided as a string is not among the valid ASE
  optimizer names defined by VALID_OPTIMIZERS.

### is_ase_optimizer(key: str | Optimizer) → bool

Determines whether the given key is an ASE optimizer. A key can
either be a string representing the name of an optimizer class
within ase.optimize or directly be an optimizer class that
subclasses Optimizer.

If the key is a string, the function checks whether it corresponds
to a class in ase.optimize that is a subclass of Optimizer.

* **Parameters:**
  **key** – The key to check, either a string name of an ASE
  optimizer class or a class object that potentially subclasses
  Optimizer.
* **Returns:**
  True if the key is either a string corresponding to an
  ASE optimizer subclass name in ase.optimize or a class that
  is a subclass of Optimizer. Otherwise, returns False.
