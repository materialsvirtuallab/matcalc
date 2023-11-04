---
layout: default
title: Calculating MLIP properties.md
nav_exclude: true
---

```python
from matcalc.relaxation import RelaxCalc
from matcalc.phonon import PhononCalc
from matcalc.eos import EOSCalc
from matcalc.elasticity import ElasticityCalc
from matcalc.util import get_universal_calculator
from datetime import datetime
from tqdm import tqdm

from pymatgen.ext.matproj import MPRester
```


```python
mpr = MPRester()
```

    /Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/mprester.py:182: UserWarning: mpcontribs-client not installed. Install the package to query MPContribs data, or construct pourbaix diagrams: 'pip install mpcontribs-client'
      warnings.warn(



```python
mp_data = mpr.materials._search(nelements=2, fields=["material_id", "structure"])
```


    Retrieving MaterialsDoc documents:   0%|          | 0/20627 [00:00<?, ?it/s]



```python
universal_calcs = [(name, get_universal_calculator(name)) for name in ("M3GNet", "CHGNet")]
```

    CHGNet initialized with 400,438 parameters
    CHGNet will run on cpu



```python
fmax = 0.1
opt = "BFGSLineSearch"
```


```python
data = []

for d in tqdm(mp_data[:20]):  # Here we just do a sampling of 20 structures.
    s = d.structure
    dd = {"mid": d.material_id, "composition": s.composition.formula, "nsites": len(s)}
    for uc_name, uc in universal_calcs:
        # The general principle is to do a relaxation first and just reuse the same structure.
        prop_calcs = [
            ("relax", RelaxCalc(uc, fmax=fmax, optimizer=opt)),
            ("elastic", ElasticityCalc(uc, fmax=fmax, relax_structure=False)),
            ("eos", EOSCalc(uc, fmax=fmax, relax_structure=False, optimizer=opt)),
            ("phonon", PhononCalc(uc, fmax=fmax, relax_structure=False)),
        ]
        properties = {}
        for name, c in prop_calcs:
            starttime = datetime.now()
            properties[name] = c.calc(s)
            endtime = datetime.now()
            if name == "relax":
                # Replace the structure with the one from relaxation for other property computations.
                s = properties[name]["final_structure"]
            dd[f"time_{name}_{uc_name}"] = (endtime - starttime).total_seconds()
        dd[uc_name] = properties
    data.append(dd)
```

     60%|████████████████████████████████████████████████████▊                                   | 12/20 [02:31<01:23, 10.47s/it]/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(
     70%|█████████████████████████████████████████████████████████████▌                          | 14/20 [02:35<00:36,  6.13s/it]/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/phonopy/structure/cells.py:1396: UserWarning: Crystal structure is distorted in a tricky way so that phonopy could not handle the crystal symmetry properly. It is recommended to symmetrize crystal structure well and then re-start phonon calculation from scratch.
      warnings.warn(msg)
    100%|████████████████████████████████████████████████████████████████████████████████████████| 20/20 [03:34<00:00, 10.75s/it]



```python
import pandas as pd

df = pd.DataFrame(data)
for uc_name, _ in universal_calcs:
    df[f"time_total_{uc_name}"] = (
        df[f"time_relax_{uc_name}"]
        + df[f"time_elastic_{uc_name}"]
        + df[f"time_phonon_{uc_name}"]
        + df[f"time_eos_{uc_name}"]
    )
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
      <th>mid</th>
      <th>composition</th>
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
      <td>mp-1106268</td>
      <td>Pr14 Pd6</td>
      <td>20</td>
      <td>5.038833</td>
      <td>0.341546</td>
      <td>2.877670</td>
      <td>18.464435</td>
      <td>{'relax': {'final_structure': [[-5.19403841  2...</td>
      <td>1.880930</td>
      <td>0.175465</td>
      <td>0.936000</td>
      <td>14.569963</td>
      <td>{'relax': {'final_structure': [[-5.17384363  2...</td>
      <td>26.722484</td>
      <td>17.562358</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mp-974315</td>
      <td>Ru2 I2</td>
      <td>4</td>
      <td>0.773536</td>
      <td>0.179219</td>
      <td>1.070539</td>
      <td>2.316252</td>
      <td>{'relax': {'final_structure': [[ 1.84542095 -1...</td>
      <td>1.160128</td>
      <td>0.089849</td>
      <td>0.437277</td>
      <td>2.086802</td>
      <td>{'relax': {'final_structure': [[ 1.65819622 -0...</td>
      <td>4.339546</td>
      <td>3.774056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mp-1206714</td>
      <td>Al2 Sn1</td>
      <td>3</td>
      <td>0.167687</td>
      <td>0.115795</td>
      <td>0.224074</td>
      <td>0.234789</td>
      <td>{'relax': {'final_structure': [[1.59915279 1.5...</td>
      <td>0.079418</td>
      <td>0.037410</td>
      <td>0.090126</td>
      <td>0.148499</td>
      <td>{'relax': {'final_structure': [[1.7730046  1.7...</td>
      <td>0.742345</td>
      <td>0.355453</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mp-30339</td>
      <td>Er1 Ag2</td>
      <td>3</td>
      <td>0.103580</td>
      <td>0.186156</td>
      <td>0.595875</td>
      <td>0.667356</td>
      <td>{'relax': {'final_structure': [[ 8.89873414e-0...</td>
      <td>0.099127</td>
      <td>0.044774</td>
      <td>0.626511</td>
      <td>0.447964</td>
      <td>{'relax': {'final_structure': [[6.55745455e-08...</td>
      <td>1.552967</td>
      <td>1.218376</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mp-696</td>
      <td>Si4 Pt4</td>
      <td>8</td>
      <td>0.884884</td>
      <td>0.321063</td>
      <td>4.699728</td>
      <td>7.904360</td>
      <td>{'relax': {'final_structure': [[0.89916847 3.8...</td>
      <td>3.056038</td>
      <td>0.255416</td>
      <td>4.077960</td>
      <td>3.434237</td>
      <td>{'relax': {'final_structure': [[1.0806827  3.8...</td>
      <td>13.810035</td>
      <td>10.823651</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mp-866222</td>
      <td>Ac2 Si6</td>
      <td>8</td>
      <td>3.100128</td>
      <td>0.253995</td>
      <td>0.823962</td>
      <td>6.351784</td>
      <td>{'relax': {'final_structure': [[-5.41563953e-0...</td>
      <td>0.266855</td>
      <td>0.148424</td>
      <td>0.597566</td>
      <td>5.063765</td>
      <td>{'relax': {'final_structure': [[-7.10695448e-0...</td>
      <td>10.529869</td>
      <td>6.076610</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mp-11422</td>
      <td>Gd1 Hg1</td>
      <td>2</td>
      <td>0.078004</td>
      <td>0.116329</td>
      <td>0.168464</td>
      <td>0.172210</td>
      <td>{'relax': {'final_structure': [[0. 0. 0.] Gd, ...</td>
      <td>0.038391</td>
      <td>0.033596</td>
      <td>0.055406</td>
      <td>0.109223</td>
      <td>{'relax': {'final_structure': [[ 1.26072246e-0...</td>
      <td>0.535007</td>
      <td>0.236616</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mp-21238</td>
      <td>Th4 Si4</td>
      <td>8</td>
      <td>0.755927</td>
      <td>0.203286</td>
      <td>1.478711</td>
      <td>2.996610</td>
      <td>{'relax': {'final_structure': [[1.08960316 0.7...</td>
      <td>0.201497</td>
      <td>0.103488</td>
      <td>1.154557</td>
      <td>1.503147</td>
      <td>{'relax': {'final_structure': [[1.0389477  0.7...</td>
      <td>5.434534</td>
      <td>2.962689</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mp-1215363</td>
      <td>Zr4 Pd1</td>
      <td>5</td>
      <td>0.272751</td>
      <td>0.206009</td>
      <td>1.201648</td>
      <td>2.377837</td>
      <td>{'relax': {'final_structure': [[1.68717563e-07...</td>
      <td>0.210600</td>
      <td>0.124840</td>
      <td>0.459993</td>
      <td>1.993330</td>
      <td>{'relax': {'final_structure': [[1.04751628e-06...</td>
      <td>4.058245</td>
      <td>2.788763</td>
    </tr>
    <tr>
      <th>9</th>
      <td>mp-1212658</td>
      <td>Ga1 C6</td>
      <td>7</td>
      <td>0.628700</td>
      <td>0.237454</td>
      <td>1.295100</td>
      <td>3.996263</td>
      <td>{'relax': {'final_structure': [[-1.88279105e-0...</td>
      <td>1.873289</td>
      <td>0.199898</td>
      <td>1.064403</td>
      <td>4.189688</td>
      <td>{'relax': {'final_structure': [[-6.66235219e-0...</td>
      <td>6.157517</td>
      <td>7.327278</td>
    </tr>
    <tr>
      <th>10</th>
      <td>mp-1101922</td>
      <td>Eu4 Fe8</td>
      <td>12</td>
      <td>1.728544</td>
      <td>0.420562</td>
      <td>1.829121</td>
      <td>6.689619</td>
      <td>{'relax': {'final_structure': [[-2.87170119e-0...</td>
      <td>1.939794</td>
      <td>0.214220</td>
      <td>1.440152</td>
      <td>6.306060</td>
      <td>{'relax': {'final_structure': [[4.98291305e-04...</td>
      <td>10.667846</td>
      <td>9.900226</td>
    </tr>
    <tr>
      <th>11</th>
      <td>mp-1183610</td>
      <td>Ca2 Sm6</td>
      <td>8</td>
      <td>0.502753</td>
      <td>0.239588</td>
      <td>0.704932</td>
      <td>0.858113</td>
      <td>{'relax': {'final_structure': [[3.63931889 2.1...</td>
      <td>0.128453</td>
      <td>0.058229</td>
      <td>0.135467</td>
      <td>0.982124</td>
      <td>{'relax': {'final_structure': [[3.78035221 2.1...</td>
      <td>2.305386</td>
      <td>1.304273</td>
    </tr>
    <tr>
      <th>12</th>
      <td>mp-1187980</td>
      <td>Yb6 Pb2</td>
      <td>8</td>
      <td>0.434660</td>
      <td>0.166910</td>
      <td>0.685716</td>
      <td>0.849555</td>
      <td>{'relax': {'final_structure': [[2.0021903  1.1...</td>
      <td>0.157995</td>
      <td>0.044524</td>
      <td>0.122118</td>
      <td>1.047720</td>
      <td>{'relax': {'final_structure': [[1.91249449 1.1...</td>
      <td>2.136841</td>
      <td>1.372357</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mp-1187953</td>
      <td>Yb3 Pb1</td>
      <td>4</td>
      <td>0.187564</td>
      <td>0.120173</td>
      <td>0.218063</td>
      <td>0.192944</td>
      <td>{'relax': {'final_structure': [[2.84492250e-09...</td>
      <td>0.048110</td>
      <td>0.046492</td>
      <td>0.073163</td>
      <td>0.093079</td>
      <td>{'relax': {'final_structure': [[1.78586414e-09...</td>
      <td>0.718744</td>
      <td>0.260844</td>
    </tr>
    <tr>
      <th>14</th>
      <td>mp-1185577</td>
      <td>Cs2 Hg6</td>
      <td>8</td>
      <td>0.207866</td>
      <td>0.190234</td>
      <td>0.435208</td>
      <td>1.504534</td>
      <td>{'relax': {'final_structure': [[1.97069239e-05...</td>
      <td>0.155879</td>
      <td>0.058824</td>
      <td>3.772319</td>
      <td>1.030184</td>
      <td>{'relax': {'final_structure': [[2.13005453e-05...</td>
      <td>2.337842</td>
      <td>5.017206</td>
    </tr>
    <tr>
      <th>15</th>
      <td>mp-570436</td>
      <td>Ca2 Ir4</td>
      <td>6</td>
      <td>0.230031</td>
      <td>0.262220</td>
      <td>0.385067</td>
      <td>0.766176</td>
      <td>{'relax': {'final_structure': [[2.33205448 1.6...</td>
      <td>0.160386</td>
      <td>0.138879</td>
      <td>0.285640</td>
      <td>1.973823</td>
      <td>{'relax': {'final_structure': [[2.32066406 1.6...</td>
      <td>1.643494</td>
      <td>2.558728</td>
    </tr>
    <tr>
      <th>16</th>
      <td>mp-1184183</td>
      <td>Cu1 Ge3</td>
      <td>4</td>
      <td>0.339547</td>
      <td>0.177124</td>
      <td>0.320189</td>
      <td>0.642880</td>
      <td>{'relax': {'final_structure': [[-9.97837602e-0...</td>
      <td>0.252591</td>
      <td>0.251099</td>
      <td>0.315171</td>
      <td>1.908043</td>
      <td>{'relax': {'final_structure': [[ 1.15179087e-0...</td>
      <td>1.479740</td>
      <td>2.726904</td>
    </tr>
    <tr>
      <th>17</th>
      <td>mp-1025440</td>
      <td>Cu2 Ge6</td>
      <td>8</td>
      <td>0.312863</td>
      <td>0.243399</td>
      <td>0.831579</td>
      <td>1.779199</td>
      <td>{'relax': {'final_structure': [[2.17845727 1.2...</td>
      <td>0.902331</td>
      <td>0.141078</td>
      <td>1.117333</td>
      <td>4.821146</td>
      <td>{'relax': {'final_structure': [[2.06660372 1.1...</td>
      <td>3.167040</td>
      <td>6.981888</td>
    </tr>
    <tr>
      <th>18</th>
      <td>mp-1184147</td>
      <td>Cu2 Ge6</td>
      <td>8</td>
      <td>1.466085</td>
      <td>0.279086</td>
      <td>1.348633</td>
      <td>7.323365</td>
      <td>{'relax': {'final_structure': [[ 3.87471272 -2...</td>
      <td>0.440801</td>
      <td>0.183226</td>
      <td>1.144264</td>
      <td>3.075049</td>
      <td>{'relax': {'final_structure': [[ 3.65967384 -2...</td>
      <td>10.417169</td>
      <td>4.843340</td>
    </tr>
    <tr>
      <th>19</th>
      <td>mp-1187368</td>
      <td>Tb2 Mn6</td>
      <td>8</td>
      <td>5.826307</td>
      <td>0.317086</td>
      <td>1.602082</td>
      <td>3.894644</td>
      <td>{'relax': {'final_structure': [[3.13273031 1.8...</td>
      <td>0.626025</td>
      <td>0.191146</td>
      <td>1.259279</td>
      <td>4.358320</td>
      <td>{'relax': {'final_structure': [[3.08463436 1.7...</td>
      <td>11.640119</td>
      <td>6.434770</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = df.plot(x="nsites", y="time_relax_M3GNet", kind="scatter")
```



![png](assets/Calculating%20MLIP%20properties_8_0.png)




```python
ax = df["time_total_M3GNet"].hist()
ax = df["time_total_CHGNet"].hist()
```



![png](assets/Calculating%20MLIP%20properties_9_0.png)
