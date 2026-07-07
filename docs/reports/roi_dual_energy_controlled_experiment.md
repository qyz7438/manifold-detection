# ROI Dual-Energy Controlled Experiment

## Purpose

This local controlled experiment tests the `basic`-derived claim that detector
ROI structure should be constrained in an intra/inter dual-energy plane:

```text
E_intra = class compactness + class-basin stability
E_inter = class-relation alignment + class-index anchor + class separation
```

It is not a detector AP experiment.  It is a sanity check that the energy form
separates the two expected failure modes before we attach it to real ROI
training.

## Command

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts\explore_roi_dual_energy_controlled.py --seed 42 --output output\roi_dual_energy_controlled_seed42.json
```

Five-seed sweep:

```powershell
$seeds = 1,2,3,42,999
foreach ($s in $seeds) {
  E:\anaconda\01\envs\RLimage\python.exe scripts\explore_roi_dual_energy_controlled.py --seed $s --output output\roi_dual_energy_controlled_seed$s.json
}
```

## Five-Seed Result

Mean +/- population std over seeds `1,2,3,42,999`.

| variant | e_intra | e_inter | dual_energy | inter_reference_alignment | inter_anchor_energy | reference_cls_acc | current_proto_cls_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.2578 +/- 0.1530 | 0.5338 +/- 0.0281 | 0.8958 +/- 0.0699 | 0.7137 +/- 0.0388 | 0.9727 +/- 0.0531 | 0.0892 +/- 0.0128 | 0.6071 +/- 0.0324 |
| intra_only | 0.0006 +/- 0.0001 | 0.5111 +/- 0.0618 | 0.2559 +/- 0.0309 | 0.7611 +/- 0.0632 | 1.0853 +/- 0.0748 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| inter_only | 5.0927 +/- 0.3101 | 0.0036 +/- 0.0014 | 2.5482 +/- 0.1552 | 0.9998 +/- 0.0001 | 0.0013 +/- 0.0014 | 0.0892 +/- 0.0128 | 0.0892 +/- 0.0110 |
| dual | 0.0079 +/- 0.0014 | 0.0072 +/- 0.0026 | 0.0075 +/- 0.0019 | 0.9998 +/- 0.0001 | 0.0162 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

## Interpretation

`intra_only` proves the old prototype-collapse risk: it drives ROI features into
their current class prototypes, but it does not repair the class-relation
structure or class identity.  The class-index anchor is still bad.

`inter_only` proves the opposite risk: it repairs class-relation/anchor energy,
but ROI samples remain scattered or even move further from stable class basins.

`dual` is the only variant that simultaneously reduces `E_intra`, reduces
`E_inter`, aligns relation structure, preserves class-index identity, and keeps
features classified by both current and reference prototypes.

The first version of this experiment exposed a real degeneracy: relation-matrix
alignment alone can reduce `E_inter` while allowing a global class-axis rotation.
For detection, this is insufficient because classifier output channels are
class-indexed.  The maintained `E_inter` now includes a light
`prototype_anchor_energy` term when reference prototypes or classifier weights
are provided.

## Limitation

This is a controlled synthetic ROI experiment, not a clean NWPU/VOC detector
result.  Local `data/` and `runs/` artifacts were unavailable in this checkout,
so the next real experiment should attach `roi_dual_energy` logging to
proposal-aligned ROI feature extraction and report it next to AP50/AP75,
precision, recall, false positives, ECE, and prediction count.
