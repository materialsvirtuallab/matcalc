---
layout: default
title: LiFePO4-NEB.md
nav_exclude: true
---

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Prepare-NEB-end-structures" data-toc-modified-id="Prepare-NEB-end-structures-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Prepare NEB end structures</a></span><ul class="toc-item"><li><span><a href="#Download-from-Materials-Project-and-create-supercell" data-toc-modified-id="Download-from-Materials-Project-and-create-supercell-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Download from Materials Project and create supercell</a></span></li><li><span><a href="#Relax-supercells-with-M3GNet-DIRECT,-M3GNet-MS,-and-CHGNet" data-toc-modified-id="Relax-supercells-with-M3GNet-DIRECT,-M3GNet-MS,-and-CHGNet-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Relax supercells with M3GNet-DIRECT, M3GNet-MS, and CHGNet</a></span></li><li><span><a href="#Create-and-relax-NEB-end-structures----b-and-c-directions" data-toc-modified-id="Create-and-relax-NEB-end-structures----b-and-c-directions-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Create and relax NEB end structures -- b and c directions</a></span></li></ul></li><li><span><a href="#NEB-calculations-with-M3GNet-DIRECT,-M3GNet-MS,-and-CHGNet" data-toc-modified-id="NEB-calculations-with-M3GNet-DIRECT,-M3GNet-MS,-and-CHGNet-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>NEB calculations with M3GNet-DIRECT, M3GNet-MS, and CHGNet</a></span><ul class="toc-item"><li><span><a href="#generate-NEB-images-from-end-structures-and-conduct-NEB" data-toc-modified-id="generate-NEB-images-from-end-structures-and-conduct-NEB-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>generate NEB images from end structures and conduct NEB</a></span></li><li><span><a href="#analyze-and-plot-NEB-results" data-toc-modified-id="analyze-and-plot-NEB-results-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>analyze and plot NEB results</a></span></li><li><span><a href="#Store-NEB-images-in-one-cif-file-for-visualization" data-toc-modified-id="Store-NEB-images-in-one-cif-file-for-visualization-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Store NEB images in one cif file for visualization</a></span></li><li><span><a href="#Visualize-NEB-path-(snapshots-of-VESTA-visualization-of-path_final.cif)" data-toc-modified-id="Visualize-NEB-path-(snapshots-of-VESTA-visualization-of-path_final.cif)-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Visualize NEB path (snapshots of VESTA visualization of path_final.cif)</a></span></li></ul></li></ul></div>


DFT barrier heights: path b = 0.27 eV and path c = 2.5 eV. (see table 1 in https://doi.org/10.1039/C5TA05062F)



```python
from __future__ import annotations

from pymatgen.ext.matproj import MPRester
from ase.neb import NEB, NEBTools

from matcalc.util import get_universal_calculator

mpr = MPRester("mp-api-key")
```

# Prepare NEB end structures


## Download from Materials Project and create supercell



```python
s_LFPO = mpr.get_structure_by_material_id("mp-19017")
s_LFPO.make_supercell([1, 2, 2], in_place=True)
s_LFPO.to("NEB_data/LiFePO4_supercell.cif", "cif")
s_LFPO.lattice.abc, s_LFPO.formula
```




    ((10.23619605, 11.941510200000154, 9.309834380000202), 'Li16 Fe16 P16 O64')



## Relax supercells with M3GNet-DIRECT, M3GNet-MS, and CHGNet



```python
models = {
    "M3GNet-DIRECT": get_universal_calculator("M3GNet-MP-2021.2.8-DIRECT-PES"),
    "M3GNet-MS": get_universal_calculator("M3GNet-MP-2021.2.8-PES"),
    "CHGNet": get_universal_calculator("CHGNet"),
}
```


```python
%%time
results = {}
for model_name, model in models.items():
    relaxer = RelaxCalc(model, optimizer="BFGS", relax_cell=True, fmax=0.02)
    supercell_LFPO_relaxed = relaxer.calc(s_LFPO)["final_structure"]
    results[model_name] = {"supercell_LFPO": supercell_LFPO_relaxed}
```

    /Users/qiji/repos/matgl/src/matgl/layers/_basis.py:119: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      root = torch.tensor(roots[i])
    /Users/qiji/miniconda3/lib/python3.9/site-packages/dgl/backend/pytorch/tensor.py:445: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      assert input.numel() == input.storage().size(), (


    CPU times: user 6min 26s, sys: 4min 10s, total: 10min 36s
    Wall time: 1min 35s


## Create and relax NEB end structures -- b and c directions



```python
%%time
for model_name, model in models.items():
    relaxer = RelaxCalc(model, optimizer="BFGS", relax_cell=False, fmax=0.02)
    supercell_LFPO_relaxed = results[model_name]["supercell_LFPO"]

    # NEB path along b and c directions have the same starting image.
    s_LFPO_end_b = supercell_LFPO_relaxed.copy()
    s_LFPO_end_b.remove_sites([11])
    s_LFPO_end_b_relaxed = relaxer.calc(s_LFPO_end_b)["final_structure"]
    s_LFPO_end_c = supercell_LFPO_relaxed.copy()
    s_LFPO_end_c.remove_sites([4])
    s_LFPO_end_c_relaxed = relaxer.calc(s_LFPO_end_c)["final_structure"]
    s_LFPO_start_bc = supercell_LFPO_relaxed.copy()
    s_LFPO_start_bc.remove_sites([5])
    s_LFPO_start_bc_relaxed = relaxer.calc(s_LFPO_start_bc)["final_structure"]
    results[model_name].update(
        {
            "supercell_LFPO_end_b": s_LFPO_end_b_relaxed,
            "supercell_LFPO_end_c": s_LFPO_end_c_relaxed,
            "supercell_LFPO_start_bc": s_LFPO_start_bc_relaxed,
        }
    )
```

    CPU times: user 6min 5s, sys: 4min 2s, total: 10min 7s
    Wall time: 1min 47s


# NEB calculations with M3GNet-DIRECT, M3GNet-MS, and CHGNet

The universal potentials provide reasonable agreement with each other and with literature.

References:

1. https://pubs.rsc.org/en/content/articlelanding/2011/ee/c1ee01782a
2. https://doi.org/10.1103/PhysRevApplied.7.034007


## generate NEB images from end structures and conduct NEB



```python
%%time
for neb_path in "bc":
    for model_name, model in models.items():
        NEBcalc = NEBCalc.from_end_images(
            start_struct=results[model_name]["supercell_LFPO_start_bc"],
            end_struct=results[model_name][f"supercell_LFPO_end_{neb_path}"],
            calculator=model,
            n_images=7,
            climb=True,
            traj_folder=f"NEB_data/traj_{neb_path}_{model_name}/",
        )
        barrier = NEBcalc.calc(fmax=0.05)[0]
        results[model_name][f"NEB_{neb_path}"] = NEBcalc.neb
        print(f"Barrier along {neb_path} by {model_name}: {barrier} eV.")
```

          Step     Time          Energy         fmax
    BFGS:    0 15:19:23     -756.289551        1.7385
    BFGS:    1 15:19:33     -756.400269        1.1797
    BFGS:    2 15:19:44     -756.557556        0.7485
    BFGS:    3 15:19:55     -756.633423        0.5410
    BFGS:    4 15:20:05     -756.664551        0.4628
    BFGS:    5 15:20:16     -756.674255        0.4553
    BFGS:    6 15:20:27     -756.685913        0.4410
    BFGS:    7 15:20:37     -756.703186        0.4304
    BFGS:    8 15:20:47     -756.723999        0.3180
    BFGS:    9 15:20:58     -756.737427        0.3052
    BFGS:   10 15:21:09     -756.743713        0.3104
    BFGS:   11 15:21:19     -756.752258        0.3113
    BFGS:   12 15:21:30     -756.762878        0.2574
    BFGS:   13 15:21:40     -756.776611        0.2825
    BFGS:   14 15:21:51     -756.789490        0.2677
    BFGS:   15 15:22:01     -756.797546        0.2350
    BFGS:   16 15:22:12     -756.802307        0.2246
    BFGS:   17 15:22:22     -756.807617        0.1795
    BFGS:   18 15:22:33     -756.815979        0.1706
    BFGS:   19 15:22:43     -756.825684        0.1962
    BFGS:   20 15:22:54     -756.832275        0.2298
    BFGS:   21 15:23:04     -756.834961        0.1380
    BFGS:   22 15:23:15     -756.836792        0.0978
    BFGS:   23 15:23:25     -756.839172        0.1364
    BFGS:   24 15:23:39     -756.842102        0.1563
    BFGS:   25 15:23:50     -756.844360        0.1480
    BFGS:   26 15:24:01     -756.845764        0.0800
    BFGS:   27 15:24:13     -756.846558        0.0503
    BFGS:   28 15:24:24     -756.847595        0.0708
    BFGS:   29 15:24:35     -756.848633        0.1031
    BFGS:   30 15:24:45     -756.849548        0.1069
    BFGS:   31 15:24:55     -756.850220        0.0721
    BFGS:   32 15:25:06     -756.850708        0.0733
    BFGS:   33 15:25:16     -756.851318        0.0624
    BFGS:   34 15:25:27     -756.851807        0.0857
    BFGS:   35 15:25:38     -756.852295        0.0808
    BFGS:   36 15:25:48     -756.852661        0.0612
    BFGS:   37 15:25:59     -756.852905        0.0376
    Barrier along b by M3GNet-DIRECT: 0.15570068359375022 eV.
          Step     Time          Energy         fmax
    BFGS:    0 15:26:18     -756.778992        2.2643
    BFGS:    1 15:26:27     -756.910278        1.4575
    BFGS:    2 15:26:36     -757.041931        0.5749
    BFGS:    3 15:26:45     -757.073242        0.4459
    BFGS:    4 15:26:55     -757.100769        0.3983
    BFGS:    5 15:27:04     -757.107971        0.4363
    BFGS:    6 15:27:13     -757.114258        0.4002
    BFGS:    7 15:27:22     -757.125671        0.2933
    BFGS:    8 15:27:31     -757.141235        0.2346
    BFGS:    9 15:27:40     -757.152466        0.2432
    BFGS:   10 15:27:54     -757.158203        0.2187
    BFGS:   11 15:28:03     -757.164734        0.2389
    BFGS:   12 15:28:13     -757.176086        0.2578
    BFGS:   13 15:28:23     -757.187561        0.2303
    BFGS:   14 15:28:37     -757.196777        0.1828
    BFGS:   15 15:30:39     -757.204285        0.2127
    BFGS:   16 15:30:49     -757.211792        0.2250
    BFGS:   17 15:31:02     -757.219910        0.2329
    BFGS:   18 15:31:12     -757.227661        0.2223
    BFGS:   19 15:31:26     -757.233276        0.1275
    BFGS:   20 15:31:37     -757.236511        0.1160
    BFGS:   21 15:31:47     -757.239197        0.0977
    BFGS:   22 15:31:57     -757.242371        0.1527
    BFGS:   23 15:32:08     -757.245300        0.1527
    BFGS:   24 15:32:20     -757.247498        0.0824
    BFGS:   25 15:32:30     -757.248718        0.0508
    BFGS:   26 15:32:40     -757.249451        0.0660
    BFGS:   27 15:32:49     -757.250305        0.1019
    BFGS:   28 15:32:58     -757.251282        0.0959
    BFGS:   29 15:33:07     -757.252014        0.0438
    Barrier along b by M3GNet-MS: 0.16577148437500006 eV.
          Step     Time          Energy         fmax
    BFGS:    0 15:33:28     -837.974968        2.0705
    BFGS:    1 15:33:42     -838.128515        1.3693
    BFGS:    2 15:33:52     -838.279415        0.9841
    BFGS:    3 15:34:02     -838.378233        0.7758
    BFGS:    4 15:34:12     -838.425817        0.6562
    BFGS:    5 15:34:23     -838.450852        0.6599
    BFGS:    6 15:34:34     -838.480704        0.5974
    BFGS:    7 15:34:44     -838.510132        0.4720
    BFGS:    8 15:34:55     -838.535115        0.4638
    BFGS:    9 15:35:06     -838.549511        0.3139
    BFGS:   10 15:35:17     -838.565337        0.3192
    BFGS:   11 15:35:29     -838.583704        0.3742
    BFGS:   12 15:35:41     -838.601435        0.3055
    BFGS:   13 15:35:56     -838.614244        0.2258
    BFGS:   14 15:36:10     -838.622659        0.2390
    BFGS:   15 15:36:22     -838.631445        0.2130
    BFGS:   16 15:36:35     -838.636632        0.1455
    BFGS:   17 15:36:47     -838.640496        0.1560
    BFGS:   18 15:37:01     -838.643990        0.1220
    BFGS:   19 15:37:13     -838.648436        0.1290
    BFGS:   20 15:37:25     -838.652405        0.1219
    BFGS:   21 15:37:38     -838.654469        0.0716
    BFGS:   22 15:37:52     -838.655793        0.0587
    BFGS:   23 15:38:08     -838.657169        0.0658
    BFGS:   24 15:38:18     -838.658651        0.0636
    BFGS:   25 15:38:30     -838.660027        0.0756
    BFGS:   26 15:38:44     -838.661191        0.0606
    BFGS:   27 15:38:56     -838.662144        0.0505
    BFGS:   28 15:39:08     -838.662991        0.0559
    BFGS:   29 15:39:19     -838.663467        0.0360
    Barrier along b by CHGNet: 0.15079450607299794 eV.
          Step     Time          Energy         fmax
    BFGS:    0 15:39:42     -755.309570        4.6894
    BFGS:    1 15:39:58     -755.584900        1.7557
    BFGS:    2 15:40:13     -755.710266        1.3291
    BFGS:    3 15:40:26     -755.900452        0.8225
    BFGS:    4 15:40:44     -755.952881        0.6253
    BFGS:    5 15:40:56     -755.977600        0.6679
    BFGS:    6 15:41:07     -755.990234        0.5458
    BFGS:    7 15:41:19     -756.008728        0.4235
    BFGS:    8 15:41:36     -756.034424        0.2833
    BFGS:    9 15:41:50     -756.053284        0.2605
    BFGS:   10 15:42:03     -756.062317        0.3140
    BFGS:   11 15:42:15     -756.066711        0.2848
    BFGS:   12 15:42:27     -756.069824        0.2368
    BFGS:   13 15:42:41     -756.074585        0.1819
    BFGS:   14 15:42:54     -756.080383        0.1244
    BFGS:   15 15:43:10     -756.083862        0.1346
    BFGS:   16 15:43:23     -756.086182        0.1171
    BFGS:   17 15:43:35     -756.088257        0.1292
    BFGS:   18 15:43:53     -756.091003        0.1262
    BFGS:   19 15:44:05     -756.093872        0.0886
    BFGS:   20 15:44:18     -756.095459        0.0855
    BFGS:   21 15:44:29     -756.096252        0.0741
    BFGS:   22 15:44:42     -756.097168        0.0814
    BFGS:   23 15:44:56     -756.098633        0.0923
    BFGS:   24 15:45:12     -756.100037        0.0928
    BFGS:   25 15:45:27     -756.100769        0.0582
    BFGS:   26 15:45:39     -756.101074        0.0455
    Barrier along c by M3GNet-DIRECT: 0.9075317382812518 eV.
          Step     Time          Energy         fmax
    BFGS:    0 15:46:00     -754.155762        8.8634
    BFGS:    1 15:46:12     -755.259277        4.4173
    BFGS:    2 15:46:26     -755.776550        2.1116
    BFGS:    3 15:46:36     -756.130371        1.1281
    BFGS:    4 15:46:46     -756.257507        0.7728
    BFGS:    5 15:46:59     -756.326538        0.6759
    BFGS:    6 15:47:09     -756.395874        0.6868
    BFGS:    7 15:47:19     -756.456360        0.5239
    BFGS:    8 15:47:32     -756.500610        0.3723
    BFGS:    9 15:47:42     -756.514526        0.2442
    BFGS:   10 15:47:52     -756.517883        0.3035
    BFGS:   11 15:48:03     -756.524414        0.2710
    BFGS:   12 15:48:15     -756.538940        0.3067
    BFGS:   13 15:48:28     -756.553345        0.2676
    BFGS:   14 15:50:42     -756.557312        0.1829
    BFGS:   15 15:50:52     -756.557373        0.1839
    BFGS:   16 15:51:02     -756.559143        0.1692
    BFGS:   17 15:51:12     -756.563782        0.2088
    BFGS:   18 15:51:21     -756.568726        0.1530
    BFGS:   19 15:51:30     -756.570557        0.1195
    BFGS:   20 15:51:39     -756.570862        0.1098
    BFGS:   21 15:51:48     -756.571716        0.1347
    BFGS:   22 15:52:02     -756.573853        0.1314
    BFGS:   23 15:52:13     -756.575867        0.0860
    BFGS:   24 15:52:25     -756.576538        0.0690
    BFGS:   25 15:52:34     -756.576843        0.0518
    BFGS:   26 15:52:45     -756.577454        0.0788
    BFGS:   27 15:52:54     -756.578308        0.0760
    BFGS:   28 15:53:03     -756.579102        0.0598
    BFGS:   29 15:53:13     -756.579285        0.0419


    Barrier along c by M3GNet-MS: 0.8385009765624982 eV.
          Step     Time          Energy         fmax
    BFGS:    0 15:53:34     -835.787680        6.1364
    BFGS:    1 15:53:44     -836.417958        2.7437
    BFGS:    2 15:53:55     -836.680432        1.4703
    BFGS:    3 15:54:10     -836.901305        0.8638
    BFGS:    4 15:54:22     -836.977469        0.8676
    BFGS:    5 15:54:32     -837.087403        0.7341
    BFGS:    6 15:54:42     -837.182093        0.7841
    BFGS:    7 15:54:53     -837.234545        0.4048
    BFGS:    8 15:55:02     -837.248836        0.3205
    BFGS:    9 15:55:13     -837.258204        0.4190
    BFGS:   10 15:55:26     -837.276835        0.4181
    BFGS:   11 15:55:39     -837.293561        0.2552
    BFGS:   12 15:55:54     -837.299595        0.2638
    BFGS:   13 15:56:05     -837.302771        0.2829
    BFGS:   14 15:56:18     -837.309704        0.3134
    BFGS:   15 15:56:30     -837.317697        0.2168
    BFGS:   16 15:56:44     -837.320978        0.1450
    BFGS:   17 15:56:56     -837.321507        0.1712
    BFGS:   18 15:57:08     -837.324154        0.2301
    BFGS:   19 15:57:20     -837.329447        0.1667
    BFGS:   20 15:57:32     -837.332305        0.1272
    BFGS:   21 15:57:47     -837.332146        0.1086
    BFGS:   22 15:58:00     -837.333046        0.1558
    BFGS:   23 15:58:12     -837.335639        0.1346
    BFGS:   24 15:58:24     -837.337386        0.0796
    BFGS:   25 15:58:35     -837.337651        0.0647
    BFGS:   26 15:58:48     -837.337862        0.1204
    BFGS:   27 15:58:59     -837.338709        0.1218
    BFGS:   28 15:59:11     -837.340033        0.0717
    BFGS:   29 15:59:24     -837.340403        0.0548
    BFGS:   30 15:59:37     -837.340509        0.0713
    BFGS:   31 15:59:50     -837.340827        0.0859
    BFGS:   32 16:00:02     -837.341515        0.0615
    BFGS:   33 16:00:15     -837.341938        0.0486
    Barrier along c by CHGNet: 1.4723238945007289 eV.
    CPU times: user 2h 25min 22s, sys: 54min 31s, total: 3h 19min 54s
    Wall time: 41min 13s


## analyze and plot NEB results



```python
%%time
import matplotlib.pyplot as plt

for neb_path in "bc":
    for model_name, model in models.items():
        NEB_tool = NEBTools(results[model_name][f"NEB_{neb_path}"].images)
        print(f"Path along {neb_path}, {model_name}: ")
        fig = NEB_tool.plot_band()
        plt.show()
```

    Path along b, M3GNet-DIRECT:




![png](assets/LiFePO4-NEB_15_1.png)



    Path along b, M3GNet-MS:




![png](assets/LiFePO4-NEB_15_3.png)



    Path along b, CHGNet:




![png](assets/LiFePO4-NEB_15_5.png)



    Path along c, M3GNet-DIRECT:




![png](assets/LiFePO4-NEB_15_7.png)



    Path along c, M3GNet-MS:




![png](assets/LiFePO4-NEB_15_9.png)



    Path along c, CHGNet:




![png](assets/LiFePO4-NEB_15_11.png)



    CPU times: user 2min 33s, sys: 1min 7s, total: 3min 41s
    Wall time: 50 s


## Store NEB images in one cif file for visualization



```python
from itertools import chain

from pymatgen.core import PeriodicSite, Structure
from pymatgen.io.ase import AseAtomsAdaptor


def generate_path_cif_from_images(images: list, filename: str) -> None:
    """Generate a cif file from a list of image atoms."""
    image_structs = list(map(AseAtomsAdaptor().get_structure, images))
    sites = set()
    lattice = image_structs[0].lattice
    sites.update(
        PeriodicSite(site.species, site.frac_coords, lattice) for site in chain(*(struct for struct in image_structs))
    )
    neb_path = Structure.from_sites(sorted(sites))
    neb_path.to(filename, "cif")
```


```python
%%time
for neb_path in "bc":
    for model_name, model in models.items():
        NEB_tool = NEBTools(results[model_name][f"NEB_{neb_path}"].images)
        generate_path_cif_from_images(NEB_tool.images, f"NEB_data/traj_{neb_path}_{model_name}/path_final.cif")
```

    CPU times: user 23.7 s, sys: 193 ms, total: 23.9 s
    Wall time: 20.6 s


## Visualize NEB path (snapshots of VESTA visualization of path_final.cif)



```python
from IPython.display import Image
```


```python
print("Final path b by M3GNet-DIRECT:")
Image("NEB_data/M3GNet-DIRECT-path-b.png")
```

    Final path b by M3GNet-DIRECT:






![png](assets/LiFePO4-NEB_21_1.png)





```python
print("Final path c by M3GNet-DIRECT:")
Image("NEB_data/M3GNet-DIRECT-path-c.png")
```

    Final path c by M3GNet-DIRECT:






![png](assets/LiFePO4-NEB_22_1.png)
