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
import numpy as np

from matcalc import PESCalculator
from matcalc.benchmark import ElasticityBenchmark
```

# Elasticity Benchmark

For demonstration purposes only, we will sample 10 structures from the entire test dataset.


```python
benchmark = ElasticityBenchmark(n_samples=10, seed=2025, fmax=0.05, relax_structure=True)
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


```python
df = pd.merge(results["M3GNet"], results["TensorNet"], on='mp_id', how='inner', suffixes=(None, "_dupe"))
```


```python
# To dump the results to a csv file, uncomment the code below.
# results.to_csv("MatCalc-Benchmark-elasticity.csv")
```


```python
for c in df.columns:
    if c.startswith("K") or c.startswith("G"):
        df[f"AE {c}"] = np.abs(df[c] - df[f"{c.split('_')[0]}_vrh_DFT"])
        print(f"MAE {c} = {df[f'AE {c}'].mean():.1f}")
```

    MAE K_vrh_DFT = 0.0
    MAE G_vrh_DFT = 0.0
    MAE K_vrh_M3GNet = 70.4
    MAE G_vrh_M3GNet = 21.0
    MAE K_vrh_DFT_dupe = 0.0
    MAE G_vrh_DFT_dupe = 0.0
    MAE K_vrh_TensorNet = 25.4
    MAE G_vrh_TensorNet = 10.7


# Statistical significance test

When comparing the performance of models, it is important to not just look at the final MAE but also to perform a rigorous statistical test of whether there is a significant difference between the MAEs. Since the model predictions are for the same set of compounds, this can be done using the paired t-test. See: https://www.jmp.com/en/statistics-knowledge-portal/t-test/two-sample-t-test


```python
from scipy.stats import ttest_rel
```


```python
print(ttest_rel(df["AE K_vrh_TensorNet"], df["AE K_vrh_M3GNet"]))
print(ttest_rel(df["AE G_vrh_TensorNet"], df["AE G_vrh_M3GNet"]))
```

    TtestResult(statistic=-1.1585260485661484, pvalue=0.2764656551436894, df=9)
    TtestResult(statistic=-1.4169146794224114, pvalue=0.19017986319358168, df=9)


At an alpha of 5%, the p value show that we **reject the null hypothesis that the MAEs in K of the two models are the same**, i.e., the difference in MAEs in K of the two models is statistically significant. However, we **do not reject the null hypothesis that the MAEs in G of the two models are the same**, i.e., the difference in MAEs in G of the two models is not statistically significant.
