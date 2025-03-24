### [wbm-random-pbe54-equilibrium-2025.1.json.gz](wbm-random-pbe54-equilibrium-2025.1.json.gz)
- Relaxed structure, un- and corrected energy of random sampled structures in [WBM] downloaded in Jan 2025
- Corrects energy using MaterialsProject2020Compatibility.
- Uses PBE PAW datasets version 54.
- 1000 materials.

### [mp-binary-pbe-elasticity-2025.1.json.gz](mp-binary-pbe-elasticity-2025.1.json.gz)
- Elastic moduli of binaries in [Materials Project] downloaded in Jan 2025
- Excludes K > 500 and G > 500 structures as well as a few bad structures.
- 3953 materials.

### [mp-pbe-elasticity-2025.3.json.gz](mp-pbe-elasticity-2025.3.json.gz)
- All available elastic moduli in [Materials Project] downloaded in Mar 2025
- Excludes K <= 0, K > 500 and G <= 0, G > 500 structures.
- Excludes H2, N2, O2, F2, Cl2, He, Xe, Ne, Kr, Ar
- Excludes materials with density < 0.5 (less dense than Li, the least density solid element)
- 12122 materials.

### [alexandria-binary-pbe-phonon-2025.1.json.gz](alexandria-binary-pbe-phonon-2025.1.json.gz)
- Heat capacity at 300 K of binaries in [Alexandria Materials Database] downloaded in Jan 2025.
- Excludes deprecated structures.
- 1170 materials.
  
### [alexandria-pbe-phonon-2025.3.json.gz](alexandria-pbe-phonon-2025.3.json.gz)
- All available heat capacity at 300 K in [Alexandria Materials Database] downloaded in Mar 2025.
- Excludes deprecated structures.
- 9865 materials.

### [wbm-high-energy-states.json.gz](wbm-high-energy-states.json.gz)
- All available forces in [WBM high energy states] downloaded in Jan 2025
- Uses PBE PAW datasets version 54.
- 972 materials, 9308 structures.

[WBM]: https://figshare.com/articles/dataset/Matbench_Discovery_v1_0_0/22715158
[Materials Project]: http://materialsproject.org
[Alexandria Materials Database]: https://alexandria.icams.rub.de
[WBM high energy states]: https://figshare.com/articles/dataset/WBM_high_energy_states/27307776?file=50005317
