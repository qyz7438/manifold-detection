# NNI Quality Matrix Results - 2026-06-03

## Run Summary

- Experiment ID: `cerkvsuz`
- NNI status: `DONE`
- Runtime: 2026-06-03 17:47:32 to 18:23:35
- Search size: 108 trials
- Trial status: 108 succeeded, 0 failed
- Result file: `runs/nni_quality_matrix/nni_matrix_results.jsonl`
- Environment: `E:\anaconda\01\envs\RLimage`

## Search Space

| Variable | Values |
| --- | --- |
| detector baseline epochs | 1, 3, 5 |
| quality head input | ROI-only, ROI+Amp, ROI+Amp+Struct |
| QH max epochs | 8, 20, with early stopping |
| rerank alpha | 0.95, 0.9, 0.85, 0.8, 0.75, 0.7 |

Objective used by NNI:

```text
AP50 + Precision@Recall=0.85 - ECE - High-conf FP rate
```

## Best By Objective

| detector epochs | QH | QH epochs | alpha | AP50 | Precision | Recall | High-conf FP | ECE | Precision@R=0.85 | Objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ROI-only | 8 | 0.70 | 0.8595 | 0.6752 | 0.8681 | 2 | 0.0656 | 0.8764 | 1.6425 |
| 1 | ROI-only | 20 | 0.70 | 0.8595 | 0.6752 | 0.8681 | 2 | 0.0656 | 0.8764 | 1.6425 |
| 1 | ROI-only | 8 | 0.75 | 0.8595 | 0.6695 | 0.8681 | 2 | 0.0625 | 0.8667 | 1.6359 |
| 1 | ROI-only | 20 | 0.75 | 0.8595 | 0.6695 | 0.8681 | 2 | 0.0625 | 0.8667 | 1.6359 |
| 1 | ROI+Amp+Struct | 8 | 0.75 | 0.8581 | 0.6695 | 0.8681 | 2 | 0.0736 | 0.8667 | 1.6238 |

## Best Per Detector Epoch

| detector epochs | best QH | QH epochs | alpha | AP50 | Precision | Recall | High-conf FP | ECE | Precision@R=0.85 | Objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ROI-only | 8 | 0.70 | 0.8595 | 0.6752 | 0.8681 | 2 | 0.0656 | 0.8764 | 1.6425 |
| 3 | ROI+Amp | 8 | 0.95 | 0.8366 | 0.7476 | 0.8462 | 4 | 0.0638 | N/A | 0.7180 |
| 5 | ROI+Amp+Struct | 8 | 0.80 | 0.8328 | 0.7624 | 0.8462 | 3 | 0.0472 | N/A | 0.7456 |

## Best Per Quality Head

| QH | detector epochs | QH epochs | alpha | AP50 | Precision | Recall | High-conf FP | ECE | Precision@R=0.85 | Objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROI-only | 1 | 8 | 0.70 | 0.8595 | 0.6752 | 0.8681 | 2 | 0.0656 | 0.8764 | 1.6425 |
| ROI+Amp | 1 | 8 | 0.95 | 0.8581 | 0.6529 | 0.8681 | 2 | 0.0566 | 0.8041 | 1.5783 |
| ROI+Amp+Struct | 1 | 8 | 0.75 | 0.8581 | 0.6695 | 0.8681 | 2 | 0.0736 | 0.8667 | 1.6238 |

## Readout

In this Penn-Fudan matrix, the best overall configuration is `detector_epochs=1`, `QH=ROI-only`, `QH epochs=8`, `alpha=0.7`. It keeps the highest AP50 and recall while improving Precision@Recall=0.85.

The 3-epoch and 5-epoch detector baselines improve precision in some settings, and 5 epochs gives the lowest ECE in several rows, but both reduce recall to about `0.8462`. Because the fixed-recall target is `0.85`, their `Precision@Recall=0.85` is not available, which makes them weaker under the current objective.

The frequency/structure branches do not beat ROI-only on this clean Penn-Fudan validation matrix. ROI+Amp has slightly better ECE in its best 1-epoch row, while ROI+Amp+Struct reaches the same fixed-recall precision band as ROI-only at alpha `0.75`, but neither provides a clear objective gain here.

QH max epochs `8` and `20` produce identical results in most paired rows because early stopping records the effective QH training at epoch 6 or 7 across the matrix.

## Next Analysis Step

Use these 108 rows to plot alpha trade-off curves per detector/QH group, then repeat the same matrix on patch perturbation splits. Clean validation alone currently favors ROI-only; the real value of amplitude/structure evidence should be judged on random/checkerboard/object-edge patch tests.
