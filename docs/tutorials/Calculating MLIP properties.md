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

from matcalc.elasticity import ElasticityCalc
from matcalc.eos import EOSCalc
from matcalc.phonon import PhononCalc
from matcalc.relaxation import RelaxCalc
from matcalc.utils import get_universal_calculator

warnings.filterwarnings("ignore", category=UserWarning, module="matgl")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
```

    /Users/aoo216/miniforge3/envs/uip_dev/lib/python3.10/site-packages/paramiko/pkey.py:100: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from this module in 48.0.0.
      "cipher": algorithms.TripleDES,
    /Users/aoo216/miniforge3/envs/uip_dev/lib/python3.10/site-packages/paramiko/transport.py:259: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from this module in 48.0.0.
      "class": algorithms.TripleDES,



```python
mp_data = MPRester().materials.search(
    num_sites=(1, 8), fields=["material_id", "structure"], num_chunks=1, chunk_size=100
)
```


    Retrieving MaterialsDoc documents:   0%|          | 0/100 [00:00<?, ?it/s]



```python
models = [(name, get_universal_calculator(name)) for name in ("M3GNet", "CHGNet", "MACE", "SevenNet")]
```

    CHGNet v0.3.0 initialized with 412,525 parameters
    CHGNet will run on mps
    Using Materials Project MACE for MACECalculator with /Users/aoo216/.cache/mace/5yyxdm76
    Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization.
    Default dtype float32 does not match model dtype float64, converting models to float32.



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

    Running mp-1183126 (Ac6 Pa2):  40%|████      | 4/10 [00:28<00:42,  7.02s/it]    /Users/aoo216/miniforge3/envs/uip_dev/lib/python3.10/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    Running mp-1183091 (Ac3 Er1):  80%|████████  | 8/10 [01:31<00:26, 13.40s/it]    /Users/aoo216/miniforge3/envs/uip_dev/lib/python3.10/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    Running mp-985294 (Ac6 Er2):  90%|█████████ | 9/10 [01:37<00:10, 10.98s/it] /Users/aoo216/miniforge3/envs/uip_dev/lib/python3.10/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
    Running mp-985294 (Ac6 Er2): 100%|██████████| 10/10 [01:53<00:00, 11.37s/it]



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
      <th>...</th>
      <th>MACE</th>
      <th>time_relax_SevenNet</th>
      <th>time_elastic_SevenNet</th>
      <th>time_eos_SevenNet</th>
      <th>time_phonon_SevenNet</th>
      <th>SevenNet</th>
      <th>time_total_M3GNet</th>
      <th>time_total_CHGNet</th>
      <th>time_total_MACE</th>
      <th>time_total_SevenNet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mp-1185285</td>
      <td>Li1 Ac1 Hg2</td>
      <td>4</td>
      <td>0.053090</td>
      <td>0.407072</td>
      <td>0.128789</td>
      <td>0.551829</td>
      <td>{'relax': {'final_structure': [[0. 0. 0.] Li, ...</td>
      <td>0.481385</td>
      <td>0.711362</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[0. 0. 0.] Li, ...</td>
      <td>2.169395</td>
      <td>0.883805</td>
      <td>0.450812</td>
      <td>0.676039</td>
      <td>{'relax': {'final_structure': [[-7.42776665e-0...</td>
      <td>1.140780</td>
      <td>2.458804</td>
      <td>1.869540</td>
      <td>4.180051</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mp-1183106</td>
      <td>Ac2 Zn1 In1</td>
      <td>4</td>
      <td>0.030421</td>
      <td>0.268268</td>
      <td>0.120450</td>
      <td>0.364040</td>
      <td>{'relax': {'final_structure': [[6.04095565 6.0...</td>
      <td>0.079872</td>
      <td>0.882399</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[6.04095565 6.0...</td>
      <td>0.031452</td>
      <td>0.679544</td>
      <td>0.291791</td>
      <td>0.706820</td>
      <td>{'relax': {'final_structure': [[6.04095565 6.0...</td>
      <td>0.783179</td>
      <td>1.620735</td>
      <td>1.304092</td>
      <td>1.709606</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mp-976333</td>
      <td>Li2 Ac1 Tl1</td>
      <td>4</td>
      <td>0.024676</td>
      <td>0.289122</td>
      <td>0.131656</td>
      <td>0.395162</td>
      <td>{'relax': {'final_structure': [[3.65217186 3.6...</td>
      <td>0.117572</td>
      <td>1.021683</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[3.65217186 3.6...</td>
      <td>0.033575</td>
      <td>0.744124</td>
      <td>0.351666</td>
      <td>0.754758</td>
      <td>{'relax': {'final_structure': [[3.65217186 3.6...</td>
      <td>0.840616</td>
      <td>1.855198</td>
      <td>1.505909</td>
      <td>1.884123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mp-1006278</td>
      <td>Ac1 Eu1 Au2</td>
      <td>4</td>
      <td>0.016674</td>
      <td>0.285528</td>
      <td>0.125015</td>
      <td>0.870638</td>
      <td>{'relax': {'final_structure': [[1.90872072 1.9...</td>
      <td>0.115037</td>
      <td>0.747267</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[1.91937276 1.9...</td>
      <td>0.065882</td>
      <td>0.720139</td>
      <td>0.392468</td>
      <td>1.784170</td>
      <td>{'relax': {'final_structure': [[1.92015988 1.9...</td>
      <td>1.297855</td>
      <td>1.480986</td>
      <td>1.815516</td>
      <td>2.962660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mp-1183126</td>
      <td>Ac6 Pa2</td>
      <td>8</td>
      <td>0.138763</td>
      <td>0.763623</td>
      <td>0.476390</td>
      <td>3.300546</td>
      <td>{'relax': {'final_structure': [[1.94807284 1.1...</td>
      <td>0.514401</td>
      <td>0.787965</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[1.97716899 1.1...</td>
      <td>0.253474</td>
      <td>1.627623</td>
      <td>1.526866</td>
      <td>9.245135</td>
      <td>{'relax': {'final_structure': [[1.97259296 1.1...</td>
      <td>4.679322</td>
      <td>3.900306</td>
      <td>8.007501</td>
      <td>12.653097</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mp-862894</td>
      <td>Ac1 Sb1 Au2</td>
      <td>4</td>
      <td>0.024058</td>
      <td>0.302274</td>
      <td>0.143157</td>
      <td>0.709291</td>
      <td>{'relax': {'final_structure': [[1.85504146 1.8...</td>
      <td>0.169026</td>
      <td>0.713197</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[1.86636684 1.8...</td>
      <td>0.080085</td>
      <td>0.779797</td>
      <td>0.349200</td>
      <td>2.018222</td>
      <td>{'relax': {'final_structure': [[1.86114475 1.8...</td>
      <td>1.178781</td>
      <td>1.538933</td>
      <td>1.866018</td>
      <td>3.227304</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mp-1183483</td>
      <td>Ca1 Ac1 Rh2</td>
      <td>4</td>
      <td>0.017743</td>
      <td>0.307425</td>
      <td>0.150159</td>
      <td>0.409490</td>
      <td>{'relax': {'final_structure': [[0. 0. 0.] Ca, ...</td>
      <td>0.128595</td>
      <td>1.218095</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[-3.26710569e-0...</td>
      <td>0.072052</td>
      <td>0.799258</td>
      <td>0.381329</td>
      <td>0.833204</td>
      <td>{'relax': {'final_structure': [[-1.43153227e-0...</td>
      <td>0.884818</td>
      <td>2.183029</td>
      <td>1.706518</td>
      <td>2.085844</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mp-865927</td>
      <td>Ac1 Ti1 O3</td>
      <td>5</td>
      <td>0.057905</td>
      <td>0.590551</td>
      <td>0.220642</td>
      <td>1.857726</td>
      <td>{'relax': {'final_structure': [[ 1.98335786e-0...</td>
      <td>0.958868</td>
      <td>2.058432</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[4.89322621e-07...</td>
      <td>0.140441</td>
      <td>1.605045</td>
      <td>0.642871</td>
      <td>3.506733</td>
      <td>{'relax': {'final_structure': [[-4.54351190e-0...</td>
      <td>2.726824</td>
      <td>7.196069</td>
      <td>3.330835</td>
      <td>5.895090</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mp-1183091</td>
      <td>Ac3 Er1</td>
      <td>4</td>
      <td>0.017766</td>
      <td>0.284594</td>
      <td>0.111494</td>
      <td>0.405632</td>
      <td>{'relax': {'final_structure': [[6.87753743e-17...</td>
      <td>0.332892</td>
      <td>0.711292</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[6.87753743e-17...</td>
      <td>0.028600</td>
      <td>0.647463</td>
      <td>0.271069</td>
      <td>0.778450</td>
      <td>{'relax': {'final_structure': [[6.87753743e-17...</td>
      <td>0.819486</td>
      <td>1.981288</td>
      <td>1.152438</td>
      <td>1.725581</td>
    </tr>
    <tr>
      <th>9</th>
      <td>mp-985294</td>
      <td>Ac6 Er2</td>
      <td>8</td>
      <td>0.043345</td>
      <td>0.874058</td>
      <td>0.300116</td>
      <td>1.700058</td>
      <td>{'relax': {'final_structure': [[1.98893139 1.1...</td>
      <td>0.106140</td>
      <td>0.704069</td>
      <td>...</td>
      <td>{'relax': {'final_structure': [[1.9926     1.1...</td>
      <td>0.085181</td>
      <td>1.962314</td>
      <td>0.935084</td>
      <td>4.654795</td>
      <td>{'relax': {'final_structure': [[1.98893139 1.1...</td>
      <td>2.917576</td>
      <td>1.779912</td>
      <td>3.932626</td>
      <td>7.637374</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 27 columns</p>
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
