---
layout: default
title: Running Softening Benchmark.md
nav_exclude: true
---

## Introduction

This notebook demonstrates how to run the Softening Benchmark on M3GNet, CHGNet and TensorNet with the MatCalc repo.


```python
from __future__ import annotations

import warnings

import pandas as pd

from matcalc import load_up
from matcalc.benchmark import SofteningBenchmark
```


```python
benchmark = SofteningBenchmark(n_samples=10, seed=2025)
results = {}
for model_name in [
    "M3GNet-MatPES-PBE-v2025.1-PES",
    "M3GNet-MP-2021.2.8-DIRECT-PES",
    "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "CHGNet-MPtrj-2023.12.1-2.7M-PES",
    "TensorNet-MatPES-PBE-v2025.1-PES",
]:
    calculator = load_up(model_name)
    short_name = "-".join(model_name.split("-")[:2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results[short_name] = benchmark.run(calculator, short_name)
```

    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/bowendeng/miniforge3/lib/python3.10/site-packages/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))



```python
from functools import reduce

df = reduce(lambda left, right: pd.merge(left, right, on=["material_id", "formula"], how="inner"), results.values())
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>material_id</th>
      <th>formula</th>
      <th>softening_scale_M3GNet-MatPES</th>
      <th>softening_scale_M3GNet-MP</th>
      <th>softening_scale_CHGNet-MatPES</th>
      <th>softening_scale_CHGNet-MPtrj</th>
      <th>softening_scale_TensorNet-MatPES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wbm-3-42969</td>
      <td>MnTe2</td>
      <td>1.012727</td>
      <td>0.778262</td>
      <td>0.910873</td>
      <td>0.765415</td>
      <td>0.954871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wbm-1-29807</td>
      <td>MnPb3</td>
      <td>1.001859</td>
      <td>0.791068</td>
      <td>1.010574</td>
      <td>0.757637</td>
      <td>0.948157</td>
    </tr>
    <tr>
      <th>2</th>
      <td>wbm-3-62502</td>
      <td>PuInRu2</td>
      <td>0.836387</td>
      <td>0.667928</td>
      <td>0.923504</td>
      <td>0.645383</td>
      <td>0.984949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wbm-4-39958</td>
      <td>DyTlZnSe3</td>
      <td>0.978575</td>
      <td>1.194938</td>
      <td>0.882973</td>
      <td>0.817942</td>
      <td>0.872786</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wbm-3-23961</td>
      <td>Sc2PbF8</td>
      <td>1.035075</td>
      <td>0.910110</td>
      <td>1.009988</td>
      <td>0.979037</td>
      <td>0.942333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>wbm-5-7573</td>
      <td>CeFe2Rh</td>
      <td>0.823937</td>
      <td>1.236098</td>
      <td>0.876943</td>
      <td>0.891998</td>
      <td>1.049308</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wbm-1-51562</td>
      <td>ZrScCr2</td>
      <td>0.897361</td>
      <td>0.641984</td>
      <td>0.876990</td>
      <td>0.645717</td>
      <td>0.824205</td>
    </tr>
    <tr>
      <th>7</th>
      <td>wbm-3-37539</td>
      <td>Li(TmTe2)3</td>
      <td>0.764861</td>
      <td>0.714281</td>
      <td>0.778819</td>
      <td>0.819389</td>
      <td>0.819368</td>
    </tr>
    <tr>
      <th>8</th>
      <td>wbm-5-9851</td>
      <td>Mn3CoH4</td>
      <td>1.135135</td>
      <td>0.732736</td>
      <td>0.994660</td>
      <td>0.739150</td>
      <td>0.896947</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wbm-2-45812</td>
      <td>ThMnSi2</td>
      <td>1.094684</td>
      <td>0.740175</td>
      <td>0.962078</td>
      <td>0.833515</td>
      <td>0.951027</td>
    </tr>
  </tbody>
</table>
</div>

