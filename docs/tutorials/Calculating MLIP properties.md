---
layout: default
title: Calculating MLIP properties.md
nav_exclude: true
---

```python
from __future__ import annotations

import warnings
from time import perf_counter

import matplotlib.pyplot as plt
from mp_api.client import MPRester
from tqdm import tqdm

from matcalc import ElasticityCalc, EOSCalc, PhononCalc, RelaxCalc, PESCalculator

warnings.filterwarnings("ignore", category=UserWarning, module="matgl")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
```


```python
mp_data = MPRester().materials.summary.search(
    formula=["LiCl", "NaCl"], fields=["material_id", "structure"], num_chunks=1, chunk_size=10
)
```


    Retrieving SummaryDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



```python
models = [(name, PESCalculator.load_universal(name)) for name in ("M3GNet", "CHGNet")]
```


```python
fmax = 0.1
opt = "BFGSLineSearch"
```


```python
prop_preds = []

for dct in (pbar := tqdm(mp_data[:10])):  # Here we just do a sampling of 20 structures.
    mat_id, formula = dct.material_id, dct.structure.formula
    pbar.set_description(f"Running {mat_id} ({formula})")
    model_preds = {"material_id": mat_id, "formula": formula, "nsites": len(dct.structure)}

    for model_name, model in models:
        # The general principle is to do a relaxation first and just reuse the same structure.
        prop_calcs = [
            ("relax", RelaxCalc(model, fmax=fmax, optimizer=opt)),
            ("elastic", ElasticityCalc(model, fmax=fmax, relax_structure=False)),
            ("eos", EOSCalc(model, fmax=fmax, relax_structure=False, optimizer=opt)),
            ("phonon", PhononCalc(model, fmax=fmax, relax_structure=False)),
        ]
        properties = {}
        for name, prop_calc in prop_calcs:
            start_time = perf_counter()
            properties[name] = prop_calc.calc(dct.structure)
            if name == "relax":
                # Replace the structure with the one from relaxation for other property computations.
                struct = properties[name]["final_structure"]
            model_preds[f"time_{name}_{model_name}"] = perf_counter() - start_time
        model_preds[model_name] = properties
    prop_preds.append(model_preds)
```

    Running mp-1120767 (Na6 Cl6):   0%|                                                                                                                                    | 0/5 [00:00<?, ?it/s]/Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/phonopy/structure/cells.py:1482: UserWarning: Crystal structure is distorted in a tricky way so that phonopy could not handle the crystal symmetry properly. It is recommended to symmetrize crystal structure well and then re-start phonon calculation from scratch.
      perm_between = _compute_permutation_c(sorted_a, sorted_b, lattice, symprec)
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/phonopy/structure/cells.py:1482: UserWarning: Crystal structure is distorted in a tricky way so that phonopy could not handle the crystal symmetry properly. It is recommended to symmetrize crystal structure well and then re-start phonon calculation from scratch.
      perm_between = _compute_permutation_c(sorted_a, sorted_b, lattice, symprec)
    Running mp-22851 (Na1 Cl1):  20%|█████████████████████████▏                                                                                                    | 1/5 [00:22<01:29, 22.40s/it]/Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    /Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    Running mp-22862 (Na1 Cl1):  40%|██████████████████████████████████████████████████▍                                                                           | 2/5 [00:25<00:33, 11.08s/it]/Users/shyue/miniconda3/envs/mavrl/lib/python3.11/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    Running mp-22905 (Li1 Cl1): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:40<00:00,  8.09s/it]



```python
import pandas as pd

df_preds = pd.DataFrame(prop_preds)
for model_name, _ in models:
    df_preds[f"time_total_{model_name}"] = (
        df_preds[f"time_relax_{model_name}"]
        + df_preds[f"time_elastic_{model_name}"]
        + df_preds[f"time_phonon_{model_name}"]
        + df_preds[f"time_eos_{model_name}"]
    )
```


```python
df_preds
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
      <th>nsites</th>
      <th>time_relax_M3GNet</th>
      <th>time_elastic_M3GNet</th>
      <th>time_eos_M3GNet</th>
      <th>time_phonon_M3GNet</th>
      <th>M3GNet</th>
      <th>time_relax_CHGNet</th>
      <th>time_elastic_CHGNet</th>
      <th>time_eos_CHGNet</th>
      <th>time_phonon_CHGNet</th>
      <th>CHGNet</th>
      <th>time_total_M3GNet</th>
      <th>time_total_CHGNet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mp-1120767</td>
      <td>Na6 Cl6</td>
      <td>12</td>
      <td>0.135454</td>
      <td>3.053201</td>
      <td>1.598205</td>
      <td>2.911434</td>
      <td>{'relax': {'final_structure': [[-2.27515821  3...</td>
      <td>0.342480</td>
      <td>7.674351</td>
      <td>2.772426</td>
      <td>3.912949</td>
      <td>{'relax': {'final_structure': [[-2.27515821  3...</td>
      <td>7.698293</td>
      <td>14.702205</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mp-22851</td>
      <td>Na1 Cl1</td>
      <td>2</td>
      <td>0.027752</td>
      <td>0.355090</td>
      <td>0.136402</td>
      <td>0.453617</td>
      <td>{'relax': {'final_structure': [[0. 0. 0.] Na, ...</td>
      <td>0.109993</td>
      <td>0.994192</td>
      <td>0.363952</td>
      <td>0.706294</td>
      <td>{'relax': {'final_structure': [[0. 0. 0.] Na, ...</td>
      <td>0.972861</td>
      <td>2.174431</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mp-22862</td>
      <td>Na1 Cl1</td>
      <td>2</td>
      <td>0.040585</td>
      <td>0.323761</td>
      <td>0.137622</td>
      <td>0.522284</td>
      <td>{'relax': {'final_structure': [[2.23230856e-07...</td>
      <td>0.136384</td>
      <td>0.901810</td>
      <td>0.352454</td>
      <td>0.747818</td>
      <td>{'relax': {'final_structure': [[-8.15455981e-0...</td>
      <td>1.024252</td>
      <td>2.138466</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mp-1185319</td>
      <td>Li2 Cl2</td>
      <td>4</td>
      <td>0.022864</td>
      <td>0.568282</td>
      <td>0.201292</td>
      <td>2.356742</td>
      <td>{'relax': {'final_structure': [[0.         0. ...</td>
      <td>0.055026</td>
      <td>1.256125</td>
      <td>0.477806</td>
      <td>2.971281</td>
      <td>{'relax': {'final_structure': [[0.         0. ...</td>
      <td>3.149180</td>
      <td>4.760239</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mp-22905</td>
      <td>Li1 Cl1</td>
      <td>2</td>
      <td>0.039179</td>
      <td>0.377370</td>
      <td>0.150175</td>
      <td>0.655296</td>
      <td>{'relax': {'final_structure': [[ 9.82945868e-0...</td>
      <td>0.205040</td>
      <td>1.170390</td>
      <td>0.441206</td>
      <td>0.807996</td>
      <td>{'relax': {'final_structure': [[-6.67281562e-0...</td>
      <td>1.222021</td>
      <td>2.624632</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
for i, (model_name, model) in enumerate(models):
    ax = axes[i]
    df_preds.plot(x="nsites", y=f"time_total_{model_name}", kind="scatter", ax=ax)
    ax.set_xlabel("Number of sites")
    ax.set_ylabel("Time for relaxation (s)")
    ax.set_title(model_name)

plt.tight_layout()
plt.show()
```



![png](assets/Calculating%20MLIP%20properties_7_0.png)




```python
for model_name, _ in models[:2]:
    ax = df_preds[f"time_total_{model_name}"].hist(label=model_name, alpha=0.6)

ax.set_xlabel("Total Time (s)")
ax.set_ylabel("Count")
ax.legend()
plt.show()
```



![png](assets/Calculating%20MLIP%20properties_8_0.png)
