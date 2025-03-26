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

Bases: `ABC`

API for a property calculator.

#### \_abc_impl *= <_abc._abc_data object>*

#### *abstractmethod* calc(structure: Structure | dict[str, Any]) → dict[str, Any]

All PropCalc subclasses should implement a calc method that takes in a pymatgen structure
and returns a dict. The method can return more than one property. Generally, subclasses should have a super()
call to the abstract base method to obtain an initial result dict.

* **Parameters:**
  **structure** – Pymatgen structure or a dict containing a pymatgen Structure under a “final_structure” or
  “structure” key. Allowing dicts provide the means to chain calculators, e.g., do a relaxation followed
  by an elasticity calculation.
* **Returns:**
  In the form {“prop_name”: value}.
* **Return type:**
  dict[str, Any]

#### calc_many(structures: Sequence[Structure | dict[str, Any]], n_jobs: None | int = None, allow_errors: bool = False, \*\*kwargs: Any) → Generator[dict | None]

Performs calc on many structures. The return type is a generator given that the calc method can
potentially be expensive. It is trivial to convert the generator to a list/tuple.

* **Parameters:**
  * **structures** – List or generator of Structures.
  * **n_jobs** – The maximum number of concurrently running jobs. If -1 all CPUs are used. For n_jobs below -1,
    (n_cpus + 1 + n_jobs) are used. None is a marker for unset that will be interpreted as n_jobs=1
    unless the call is performed under a parallel_config() context manager that sets another value for
    n_jobs.
  * **allow_errors** – Whether to skip failed calculations. For these calculations, None will be returned. For
    large scale calculations, you may want this to be True to avoid the entire calculation failing.
    Defaults to False.
  * **\*\*kwargs** – Passthrough to joblib.Parallel.
* **Returns:**
  Generator of dicts.

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

#### *abstractmethod* get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc.base.PropCalc)

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
  [PropCalc](#matcalc.base.PropCalc)

#### *abstractmethod* process_result(result: dict | None, model_name: str) → dict

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

#### run(calculator: Calculator, model_name: str, , n_jobs: None | int = -1, checkpoint_file: str | Path | None = None, checkpoint_freq: int = 1000, delete_checkpoint_on_finish: bool = True, include_full_results: bool = False, \*\*kwargs) → pd.DataFrame

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

A class to run multiple benchmarks in a single run.

Represents a configuration for handling a list of benchmarks. This class is designed
to initialize with a specified list of benchmarks.

#### benchmarks

A list containing benchmark configurations or elemental_refs for

* **Type:**
  list

### evaluation.

* **Parameters:**
  **benchmarks** (*list*) – A list of benchmarks for configuration or evaluation.

#### run(calculators: dict[str, Calculator], , n_jobs: int | None = -1, checkpoint_freq: int = 1000, delete_checkpoint_on_finish: bool = True) → list[pd.DataFrame]

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

CheckpointFile class encapsulates functionality to handle checkpoint files for processing elemental_refs.

The class constructor initializes the CheckpointFile object with the provided path, all elemental_refs to be
processed, list of structures, and index name for elemental_refs identification.

load() method loads a checkpoint file if it exists, filtering the remaining elemental_refs and structures based on
entries already processed. It returns a tuple containing three lists: already processed records, remaining
elemental_refs, and remaining structures.

save() method saves a list of results at the specified checkpoint location.

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

#### get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc.base.PropCalc)

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
  [PropCalc](#matcalc.base.PropCalc)

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

#### get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc.base.PropCalc)

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
  [PropCalc](#matcalc.base.PropCalc)

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

### *class* RelaxationBenchmark(index_name: str = 'material_id', benchmark_name: str | Path = 'wbm-random-pbe54-equilibrium-2025.1.json.gz', folder_name: str = 'default_folder', \*\*kwargs)

Bases: [`Benchmark`](#matcalc.benchmark.Benchmark)

Represents a benchmark for evaluating and analyzing relaxation properties of materials.
This benchmark utilizes a dataset and provides functionality for property calculation
and result processing. The class is designed to work with a predefined framework for
benchmarking relaxation properties. The benchmark dataset contains data such as relaxed
structures along with additional metadata. This class supports configurability through
metadata files, index names, and additional benchmark properties. It relies on external
calculators and utility classes for property computations and result handling.

Initializes the RelaxationBenchmark instance with specified benchmark metadata and
configuration parameters. It sets up the benchmark with the necessary properties
required for relaxation analysis.

* **Parameters:**
  * **index_name** (*str*) – The name of the index used to uniquely identify records in the dataset.
  * **benchmark_name** (*str* *|* *Path*) – The path or name of the benchmark file that contains the dataset.
  * **folder_name** (*str*) – The folder name used for file operations related to structure files.
  * **kwargs** (*dict*) – Additional keyword arguments for customization.

#### \_abc_impl *= <_abc._abc_data object>*

#### get_prop_calc(calculator: Calculator, \*\*kwargs: Any) → [PropCalc](#matcalc.base.PropCalc)

Returns a property calculation object for performing relaxation calculations.
This method initializes the relaxation calculator using the provided Calculator
object and any additional configuration parameters.

* **Parameters:**
  * **calculator** (*Calculator*) – A Calculator object responsible for performing the relaxation calculation.
  * **kwargs** (*dict*) – Additional keyword arguments used for configuration.
* **Returns:**
  An initialized PropCalc object configured for relaxation calculations.
* **Return type:**
  [PropCalc](#matcalc.base.PropCalc)

#### process_result(result: dict | None, model_name: str) → dict

Processes the result dictionary containing final relaxed structures, formats the keys
according to the provided model name. If the result is None, default values of
NaN are returned for final structures.

* **Parameters:**
  * **result** (*dict* *or* *None*) – A dictionary containing the final relaxed structures under the keys
    ‘final_structure’. It can also be None to indicate missing elemental_refs.
  * **model_name** (*str*) – A string representing the identifier or name of the model. It will be used
    to format the returned dictionary’s keys.
* **Returns:**
  A dictionary containing the specific final relaxed structure prefixed by the model name.
  The values will be NaN if the input result is None.
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

#### run(calculator: Calculator, model_name: str, checkpoint_file: str | Path | None = None, checkpoint_freq: int = 10, , include_full_results: bool = False) → pd.DataFrame

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

Checks Github for available benchmarks for download.

* **Returns:**
  List of available benchmarks.

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

## matcalc.cli module

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

## matcalc.config module

Sets some configuration global variables and locations for matcalc.

### clear_cache(, confirm: bool = True) → None

Deletes all files in the matgl.cache. This is used to clean out downloaded models.

* **Parameters:**
  **confirm** – Whether to ask for confirmation. Default is True.

## matcalc.elasticity module

Calculator for elastic properties.

### *class* ElasticityCalc(calculator: Calculator, , norm_strains: Sequence[float] | float = (-0.01, -0.005, 0.005, 0.01), shear_strains: Sequence[float] | float = (-0.06, -0.03, 0.03, 0.06), fmax: float = 0.1, relax_structure: bool = True, relax_deformed_structures: bool = False, use_equilibrium: bool = True, relax_calc_kwargs: dict | None = None)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Calculator for elastic properties.

* **Parameters:**
  * **calculator** – ASE Calculator to use.
  * **norm_strains** – single or multiple strain values to apply to each normal mode.
    Defaults to (-0.01, -0.005, 0.005, 0.01).
  * **shear_strains** – single or multiple strain values to apply to each shear mode.
    Defaults to (-0.06, -0.03, 0.03, 0.06).
  * **fmax** – maximum force in the relaxed structure (if relax_structure). Defaults to 0.1.
  * **relax_structure** – whether to relax the provided structure with the given calculator.
    Defaults to True.
  * **relax_deformed_structures** – whether to relax the atomic positions of the deformed/strained structures
    with the given calculator. Defaults to True.
  * **use_equilibrium** – whether to use the equilibrium stress and strain. Ignored and set
    to True if either norm_strains or shear_strains has length 1 or is a float.
    Defaults to True.
  * **relax_calc_kwargs** – Arguments to be passed to the RelaxCalc, if relax_structure is True.

#### \_abc_impl *= <_abc._abc_data object>*

#### \_elastic_tensor_from_strains(strains: ArrayLike, stresses: ArrayLike, eq_stress: ArrayLike = None, tol: float = 1e-07) → tuple[ElasticTensor, float]

Slightly modified version of Pymatgen function
pymatgen.analysis.elasticity.elastic.ElasticTensor.from_independent_strains;
this is to give option to discard eq_stress,
which (if the structure is relaxed) tends to sometimes be
much lower than neighboring points.
Also has option to return the sum of the squares of the residuals
for all of the linear fits done to compute the entries of the tensor.

#### calc(structure: Structure | dict[str, Any]) → dict[str, Any]

Calculates elastic properties of Pymatgen structure with units determined by the calculator,
(often the stress_weight).

* **Parameters:**
  **structure** – Pymatgen structure.

Returns: {
: elastic_tensor: Elastic tensor as a pymatgen ElasticTensor object (in eV/A^3),
  shear_modulus_vrh: Voigt-Reuss-Hill shear modulus based on elastic tensor (in eV/A^3),
  bulk_modulus_vrh: Voigt-Reuss-Hill bulk modulus based on elastic tensor (in eV/A^3),
  youngs_modulus: Young’s modulus based on elastic tensor (in eV/A^3),
  residuals_sum: Sum of squares of all residuals in the linear fits of the
  calculation of the elastic tensor,
  structure: The equilibrium structure used for the computation.

}

## matcalc.eos module

Calculators for EOS and associated properties.

### *class* EOSCalc(calculator: Calculator, , optimizer: Optimizer | str = 'FIRE', max_steps: int = 500, max_abs_strain: float = 0.1, n_points: int = 11, fmax: float = 0.1, relax_structure: bool = True, relax_calc_kwargs: dict | None = None)

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
  * **relax_calc_kwargs** – Arguments to be passed to the RelaxCalc, if relax_structure is True.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure | dict[str, Any]) → dict

Fit the Birch-Murnaghan equation of state.

* **Parameters:**
  **structure** – pymatgen Structure object.

Returns: {
: eos: {
  : volumes: tuple[float] in Angstrom^3,
    energies: tuple[float] in eV,
  <br/>
  },
  bulk_modulus_bm: Birch-Murnaghan bulk modulus in GPa.
  r2_score_bm: R squared of Birch-Murnaghan fit of energies predicted by model to help detect erroneous
  calculations. This value should be at least around 1 - 1e-4 to 1 - 1e-5.

}

## matcalc.neb module

NEB calculations.

### *class* NEBCalc(images: list[Structure], , calculator: str | Calculator = 'M3GNet-MP-2021.2.8-DIRECT-PES', optimizer: str | Optimizer = 'BFGS', traj_folder: str | None = None, interval: int = 1, climb: bool = True, \*\*kwargs: Any)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

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

#### *classmethod* from_end_images(start_struct: Structure, end_struct: Structure, calculator: str | Calculator = 'M3GNet-MP-2021.2.8-DIRECT-PES', , n_images: int = 7, interpolate_lattices: bool = False, autosort_tol: float = 0.5, \*\*kwargs: Any) → [NEBCalc](#matcalc.neb.NEBCalc)

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

## matcalc.phonon module

Calculator for phonon properties.

### *class* PhononCalc(calculator: Calculator, atom_disp: float = 0.015, supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)), t_step: float = 10, t_max: float = 1000, t_min: float = 0, fmax: float = 0.1, optimizer: str = 'FIRE', relax_structure: bool = True, relax_calc_kwargs: dict | None = None, write_force_constants: bool | str | Path = False, write_band_structure: bool | str | Path = False, write_total_dos: bool | str | Path = False, write_phonon: bool | str | Path = True)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Calculator for phonon properties.

* **Parameters:**
  * **calculator** (*Calculator*) – ASE Calculator to use.
  * **fmax** (*float*) – Max forces. This criterion is more stringent than for simple relaxation.
    Defaults to 0.1 (in eV/Angstrom)
  * **optimizer** (*str*) – Optimizer used for RelaxCalc.
  * **atom_disp** (*float*) – Atomic displacement (in Angstrom).
  * **supercell_matrix** (*ArrayLike*) – Supercell matrix to use. Defaults to 2x2x2 supercell.
  * **t_step** (*float*) – Temperature step (in Kelvin).
  * **t_max** (*float*) – Max temperature (in Kelvin).
  * **t_min** (*float*) – Min temperature (in Kelvin).
  * **relax_structure** (*bool*) – Whether to first relax the structure. Set to False if structures
    provided are pre-relaxed with the same calculator.
  * **relax_calc_kwargs** (*dict*) – Arguments to be passed to the RelaxCalc, if relax_structure is True.
  * **write_force_constants** (*bool* *|* *str* *|* *Path*) – Whether to save force constants. Pass string or Path
    for custom filename. Set to False for storage conservation. This file can be very large, be
    careful when doing high-throughput. Defaults to False.
  * **calculations.**
  * **write_band_structure** (*bool* *|* *str* *|* *Path*) – Whether to calculate and save band structure
    (in yaml format). Defaults to False. Pass string or Path for custom filename.
  * **write_total_dos** (*bool* *|* *str* *|* *Path*) – Whether to calculate and save density of states
    (in dat format). Defaults to False. Pass string or Path for custom filename.
  * **write_phonon** (*bool* *|* *str* *|* *Path*) – Whether to save phonon object. Set to True to save
    necessary phonon calculation results. Band structure, density of states, thermal properties,
    etc. can be rebuilt from this file using the phonopy API via phonopy.load(“phonon.yaml”).
    Defaults to True. Pass string or Path for custom filename.

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

## matcalc.qha module

Calculator for phonon properties under quasi-harmonic approximation.

### *class* QHACalc(calculator: Calculator, t_step: float = 10, t_max: float = 1000, t_min: float = 0, fmax: float = 0.1, optimizer: str = 'FIRE', eos: str = 'vinet', relax_structure: bool = True, relax_calc_kwargs: dict | None = None, phonon_calc_kwargs: dict | None = None, scale_factors: Sequence[float] = (0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05), write_helmholtz_volume: bool | str | Path = False, write_volume_temperature: bool | str | Path = False, write_thermal_expansion: bool | str | Path = False, write_gibbs_temperature: bool | str | Path = False, write_bulk_modulus_temperature: bool | str | Path = False, write_heat_capacity_p_numerical: bool | str | Path = False, write_heat_capacity_p_polyfit: bool | str | Path = False, write_gruneisen_temperature: bool | str | Path = False)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Calculator for phonon properties under quasi-harmonic approximation.

* **Parameters:**
  * **calculator** (*Calculator*) – ASE Calculator to use.
  * **t_step** (*float*) – Temperature step. Defaults to 10 (in Kelvin).
  * **t_max** (*float*) – Max temperature (in Kelvin). Defaults to 1000 (in Kelvin).
  * **t_min** (*float*) – Min temperature (in Kelvin). Defaults to 0 (in Kelvin).
  * **fmax** (*float*) – Max forces. This criterion is more stringent than for simple relaxation.
    Defaults to 0.1 (in eV/Angstrom).
  * **optimizer** (*str*) – Optimizer used for RelaxCalc. Default to “FIRE”.
  * **eos** (*str*) – Equation of state used to fit F vs V, including “vinet”, “murnaghan” or
    “birch_murnaghan”. Default to “vinet”.
  * **relax_structure** (*bool*) – Whether to first relax the structure. Set to False if structures
    provided are pre-relaxed with the same calculator.
  * **relax_calc_kwargs** (*dict*) – Arguments to be passed to the RelaxCalc, if relax_structure is True.
  * **phonon_calc_kwargs** (*dict*) – Arguments to be passed to the PhononCalc.
  * **scale_factors** (*Sequence* *[**float* *]*) – Factors to scale the lattice constants of the structure.
  * **write_helmholtz_volume** (*bool* *|* *str* *|* *Path*) – Whether to save Helmholtz free energy vs volume in file.
    Pass string or Path for custom filename. Defaults to False.
  * **write_volume_temperature** (*bool* *|* *str* *|* *Path*) – Whether to save equilibrium volume vs temperature in file.
    Pass string or Path for custom filename. Defaults to False.
  * **write_thermal_expansion** (*bool* *|* *str* *|* *Path*) – Whether to save thermal expansion vs temperature in file.
    Pass string or Path for custom filename. Defaults to False.
  * **write_gibbs_temperature** (*bool* *|* *str* *|* *Path*) – Whether to save Gibbs free energy vs temperature in file.
    Pass string or Path for custom filename. Defaults to False.
  * **write_bulk_modulus_temperature** (*bool* *|* *str* *|* *Path*) – Whether to save bulk modulus vs temperature in file.
    Pass string or Path for custom filename. Defaults to False.
  * **write_heat_capacity_p_numerical** (*bool* *|* *str* *|* *Path*) – Whether to save heat capacity at constant pressure
    by numerical difference vs temperature in file. Pass string or Path for custom filename.
    Defaults to False.
  * **write_heat_capacity_p_polyfit** (*bool* *|* *str* *|* *Path*) – Whether to save heat capacity at constant pressure
    by fitting vs temperature in file. Pass string or Path for custom filename. Defaults to False.
  * **write_gruneisen_temperature** (*bool* *|* *str* *|* *Path*) – Whether to save Grueneisen parameter vs temperature in
    file. Pass string or Path for custom filename. Defaults to False.

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

## matcalc.relaxation module

Relaxation properties.

### *class* RelaxCalc(calculator: Calculator, \*, optimizer: Optimizer | str = 'FIRE', max_steps: int = 500, traj_file: str | None = None, interval: int = 1, fmax: float = 0.1, relax_atoms: bool = True, relax_cell: bool = True, cell_filter: Filter = <class 'ase.filters.FrechetCellFilter'>, perturb_distance: float | None = None)

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
  * **relax_atoms** (*bool*) – Whether to relax the atoms (or just static calculation).
  * **relax_cell** (*bool*) – Whether to relax the cell (or just atoms).
  * **cell_filter** (*Filter*) – The ASE Filter used to relax the cell. Default is FrechetCellFilter.
  * **perturb_distance** (*float* *|* *None*) – Distance in angstrom to randomly perturb each site to break symmetry.
    Defaults to None.
* **Raises:**
  **ValueError** – If the optimizer is not a valid ASE optimizer.

#### \_abc_impl *= <_abc._abc_data object>*

#### calc(structure: Structure | dict) → dict

Perform relaxation to obtain properties.

* **Parameters:**
  **structure** – Pymatgen structure.

Returns: {
: final_structure: final_structure,
  energy: static energy or trajectory observer final energy in eV,
  forces: forces in eV/A,
  stress: stress in eV/A^3,
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

#### save(filename: str) → None

Save the trajectory to file.

* **Parameters:**
  **filename** (*str*) – filename to save the trajectory.

## matcalc.stability module

Calculator for stability related properties.

### *class* EnergeticsCalc(calculator: Calculator, , elemental_refs: Literal['MatPES-PBE', 'MatPES-r2SCAN'] | dict = 'MatPES-PBE', use_dft_gs_reference: bool = False, relax_structure: bool = True, relax_calc_kwargs: dict | None = None)

Bases: [`PropCalc`](#matcalc.base.PropCalc)

Calculator for energetic properties.

Initialize the class with the required computational parameters to set up properties
and configurations. This class is used to perform calculations and provides an interface
to manage computational settings such as calculator setup, elemental references, ground
state relaxation, and additional calculation parameters.

* **Parameters:**
  * **calculator** (*Calculator*) – The computational calculator object implementing specific calculation
    protocols or methods for performing numerical simulations.
  * **elemental_refs** (*Literal* *[* *"MatPES-PBE"* *,*  *"MatPES-r2SCAN"* *]*  *|* *dict*) – Specifies the elemental reference data source. It can either be a
    predefined identifier (“MatPES-PBE” or “MatPES-r2SCAN”) to load default references or,
    alternatively, it can be a dictionary directly providing custom reference data. The dict should be of the
    format {element_symbol: {“structure”: structure_object, “energy_per_atom”: energy_per_atom,
    “energy_atomic”: energy_atomic}}
  * **use_dft_gs_reference** (*bool*) – Whether to use the ground state reference from DFT
    calculations for energetics or other property computations.
  * **relax_calc_kwargs** (*dict* *|* *None*) – Optional dictionary containing additional keyword arguments
    for customizing the configurations and execution of the relaxation calculations.

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

## matcalc.units module

Useful constants for unit conversions.

## matcalc.utils module

Some utility methods, e.g., for getting calculators from well-known sources.

### *class* PESCalculator(potential: LMPStaticCalculator, stress_unit: Literal['eV/A3', 'GPa'] = 'GPa', stress_weight: float = 1.0, \*\*kwargs: Any)

Bases: `Calculator`

Potential calculator for ASE, supporting both **universal** and **customized** potentials, including:
: Customized potentials: MatGL(M3GNet, CHGNet, TensorNet and SO3Net), MAML(MTP, GAP, NNP, SNAP and qSNAP) and ACE.
  Universal potentials: M3GNet, CHGNet, MACE and SevenNet.

Though MatCalc can be used with any MLIP, this method does not yet cover all MLIPs.
Imports should be inside if statements to ensure that all models are optional dependencies.

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

Load the ACE model for use in ASE as a calculator.

* **Parameters:**
  * **basis_set** – The specification of ACE potential, could be in following forms:
    “.ace” potential filename
    “.yaml” potential filename
    ACEBBasisSet object
    ACECTildeBasisSet object
    BBasisConfiguration object
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for the PyACECalculator.
* **Returns:**
  ASE calculator compatible with the ACE model.
* **Return type:**
  Calculator

#### *static* load_gap(filename: str | Path, \*\*kwargs: Any) → Calculator

Load the GAP model for use in ASE as a calculator.

* **Parameters:**
  * **filename** (*str* *|* *Path*) – The file storing parameters of potentials, filename should ends with “.xml”.
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for the PESCalculator.
* **Returns:**
  ASE calculator compatible with the GAP model.
* **Return type:**
  Calculator

#### *static* load_matgl(path: str | Path, \*\*kwargs: Any) → Calculator

Load the MatGL model for use in ASE as a calculator.

* **Parameters:**
  * **path** (*str* *|* *Path*) – The path to the folder storing model.
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for the M3GNetCalculator.
* **Returns:**
  ASE calculator compatible with the MatGL model.
* **Return type:**
  Calculator

#### *static* load_mtp(filename: str | Path, elements: list, \*\*kwargs: Any) → Calculator

Load the MTP model for use in ASE as a calculator.

* **Parameters:**
  * **filename** (*str* *|* *Path*) – The file storing parameters of potentials, filename should ends with “.mtp”.
  * **elements** (*list*) – The list of elements.
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for the PESCalculator.
* **Returns:**
  ASE calculator compatible with the MTP model.
* **Return type:**
  Calculator

#### *static* load_nequip(model_path: str | Path, \*\*kwargs: Any) → Calculator

Load the NequIP model for use in ASE as a calculator.

* **Parameters:**
  * **model_path** (*str* *|* *Path*) – The file storing the configuration of potentials, filename should ends with “.pth”.
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for the PESCalculator.
* **Returns:**
  ASE calculator compatible with the NequIP model.
* **Return type:**
  Calculator

#### *static* load_nnp(input_filename: str | Path, scaling_filename: str | Path, weights_filenames: list, \*\*kwargs: Any) → Calculator

Load the NNP model for use in ASE as a calculator.

* **Parameters:**
  * **input_filename** (*str* *|* *Path*) – The file storing the input configuration of
    Neural Network Potential.
  * **scaling_filename** (*str* *|* *Path*) – The file storing scaling info of
    Neural Network Potential.
  * **weights_filenames** (*list* *|* *Path*) – List of files storing weights of each specie in
    Neural Network Potential.
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for the PESCalculator.
* **Returns:**
  ASE calculator compatible with the NNP model.
* **Return type:**
  Calculator

#### *static* load_snap(param_file: str | Path, coeff_file: str | Path, \*\*kwargs: Any) → Calculator

Load the SNAP or qSNAP model for use in ASE as a calculator.

* **Parameters:**
  * **param_file** (*str* *|* *Path*) – The file storing the configuration of potentials.
  * **coeff_file** (*str* *|* *Path*) – The file storing the coefficients of potentials.
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for the PESCalculator.
* **Returns:**
  ASE calculator compatible with the SNAP or qSNAP model.
* **Return type:**
  Calculator

#### *static* load_universal(name: str | Calculator, \*\*kwargs: Any) → Calculator

Load the universal model for use in ASE as a calculator.

* **Parameters:**
  * **name** (*str* *|* *Calculator*) – The name of universal calculator.
  * **\*\*kwargs** (*Any*) – Additional keyword arguments for universal calculator.
* **Returns:**
  ASE calculator compatible with the universal model.
* **Return type:**
  Calculator

### get_ase_optimizer(optimizer: str | Optimizer) → Optimizer

Validate optimizer is a valid ASE Optimizer.

* **Parameters:**
  **optimizer** (*str* *|* *Optimizer*) – The optimization algorithm.
* **Raises:**
  **ValueError** – on unrecognized optimizer name.
* **Returns:**
  ASE Optimizer class.
* **Return type:**
  Optimizer

### get_universal_calculator(name: str | Calculator, \*\*kwargs: Any) → Calculator

Helper method to get some well-known **universal** calculators.
Imports should be inside if statements to ensure that all models are optional dependencies.
All calculators must be universal, i.e. encompass a wide swath of the periodic table.
Though MatCalc can be used with any MLIP, even custom ones, this function is not meant as

> a list of all MLIPs.
* **Parameters:**
  * **name** (*str*) – Name of calculator.
  * **\*\*kwargs** – Passthrough to calculator init.
* **Raises:**
  **ValueError** – on unrecognized model name.
* **Returns:**
  Calculator

### is_ase_optimizer(key: str | Optimizer) → bool

Check if key is the name of an ASE optimizer class.
