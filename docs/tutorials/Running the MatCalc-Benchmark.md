---
layout: default
title: Running the MatCalc-Benchmark.md
nav_exclude: true
---

# Introduction

This notebook demonstrates how to run the MatCalc-Benchmark. We will use the recently released TensorNet-MatPES-PBE-v2025.1-PES and M3GNet-MatPES-PBE-v2025.1-PES universal machine learning interatomic potentials for demonstration purposes. All that is needed to run the benchmark on a separate model is to provide a compatible ASE Calculator for your UMLIP.


```python
from __future__ import annotations

import warnings
import pandas as pd

from matcalc.utils import PESCalculator
from matcalc.benchmark import ElasticityBenchmark
```

# Elasticity Benchmark

For demonstration purposes only, we will sample 100 structures from the entire test dataset.


```python
benchmark = ElasticityBenchmark(n_samples=100, seed=2025, fmax=0.05, relax_structure=True)
results = {}
for model_name in [
    "M3GNet-MatPES-PBE-v2025.1-PES",
    "TensorNet-MatPES-PBE-v2025.1-PES",
]:
    calculator = PESCalculator.load_universal(model_name)
    short_name = model_name.split("-")[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results[short_name] = benchmark.run(calculator, short_name)
```

    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.7742090336621994e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.709268514944375e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.742873746880114e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.7430909002042245e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.768891709089071e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.7753201574315675e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.7350620551762567e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.7219600926842573e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.743591475484403e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.83445041647702e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.779318470973653e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.7864886990816315e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.754141866837261e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.6984287195309347e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.761159498999532e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/ase/filters.py:600: RuntimeWarning: logm result may be inaccurate, approximate err = 2.7603384558172673e-13
      pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor



```python
df = pd.merge(results["M3GNet"], results["TensorNet"], on='mp_id', how='inner')
```


```python
# To dump the results to a csv file, uncomment the code below.
# results.to_csv("MatCalc-Benchmark-elasticity.csv")
```


```python
mae_k_tensornet = df["AE K_TensorNet"].mean()
mae_k_m3gnet = df["AE K_M3GNet"].mean()
mae_g_tensornet = df["AE G_TensorNet"].mean()
mae_g_m3gnet = df["AE G_M3GNet"].mean()

print(f"MAE K_TensorNet = {mae_k_tensornet:.1f}")
print(f"MAE K_M3GNet = {mae_k_m3gnet:.1f}")
print(f"MAE G_TensorNet = {mae_g_tensornet:.1f}")
print(f"MAE G_M3GNet = {mae_g_m3gnet:.1f}")
```

    MAE K_TensorNet = 19.8
    MAE K_M3GNet = 28.8
    MAE G_TensorNet = 12.0
    MAE G_M3GNet = 15.9


# Statistical significance test

When comparing the performance of models, it is important to not just look at the final MAE but also to perform a rigorous statistical test of whether there is a significant difference between the MAEs. Since the model predictions are for the same set of compounds, this can be done using the paired t-test. See: https://www.jmp.com/en/statistics-knowledge-portal/t-test/two-sample-t-test


```python
from scipy.stats import ttest_rel
```


```python
print(ttest_rel(df["AE K_TensorNet"], df["AE K_M3GNet"]))
print(ttest_rel(df["AE G_TensorNet"], df["AE G_M3GNet"]))
```

    TtestResult(statistic=-1.9571078299108027, pvalue=0.05315113290636734, df=99)
    TtestResult(statistic=-2.394467730045528, pvalue=0.018526677811485988, df=99)


At an alpha of 5%, the p value show that we **reject the null hypothesis that the MAEs in K of the two models are the same**, i.e., the difference in MAEs in K of the two models is statistically significant. However, we **do not reject the null hypothesis that the MAEs in G of the two models are the same**, i.e., the difference in MAEs in G of the two models is not statistically significant.
