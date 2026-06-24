# Round 2221 Interpretable Reward Signal Diagnostics

Date: 2026-06-21

## Setup

- Cache: `runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz`
- Output: `runs/round2221_interpretable_reward_signal_diagnostics/`
- Task: offline ranking/gating diagnostics for LC-HI proposals on NWPU
- Train proposals: 1484, LC-HI positives: 38
- Val proposals: 641, LC-HI positives: 41
- Calibration: train threshold at target precision 0.7, then fixed-threshold eval on val

## Signals

Seven interpretable, non-network signals were implemented:

1. Boundary phase coherence
2. Interior-exterior texture contrast
3. Aspect-ratio plausibility
4. Multi-scale saliency consistency
5. Score-edge alignment
6. NMS survivor density
7. Activation-centroid consistency

`activation_centroid_consistency` is a non-network proxy for CAM consistency: edge/phase saliency centroid should align with the candidate box center.

The existing raw-iFFT three-feature recipe was also evaluated as a reference:

- `fft_edge_truncation@64`
- `phase_edge@64`
- `phase_abs_high@11`

## Leaderboard

| Signal | Val AP | Val R@P0.7 (oracle threshold) | Fixed Val P | Fixed Val R | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| fusion_interpretable_logistic | 0.451 | 0.317 | 0.812 | 0.317 | Best train-threshold transfer among useful signals |
| score_edge_alignment | 0.498 | 0.390 | 1.000 | 0.220 | Best fixed-threshold transfer, but sign is inverted by train orientation |
| fusion_interpretable_effect_sum | 0.447 | 0.317 | 0.410 | 0.390 | High recall but too many FP under train threshold |
| reference_raw_ifft_recipe | 0.393 | 0.293 | 0.600 | 0.293 | Still useful, but train P0.7 threshold over-selects FP on val |
| boundary_phase_coherence | 0.460 | 0.366 | 0.000 | 0.000 | Strong val ranker, poor train-threshold transfer |
| interior_exterior_texture_contrast | 0.322 | 0.293 | 0.000 | 0.000 | Useful as rank/sample-weight feature, not hard gate |
| multi_scale_saliency_consistency | 0.127 | 0.000 | 0.000 | 0.000 | Weak standalone signal |
| activation_centroid_consistency | 0.107 | 0.000 | 0.000 | 0.000 | Weak standalone signal |
| aspect_ratio_plausibility | 0.077 | 0.000 | 0.000 | 0.000 | Near-random standalone signal |
| nms_survivor_density | 0.060 | 0.000 | 0.000 | 0.000 | Fails to transfer; val direction is unstable |

## Interpretation

The best new signal is `score_edge_alignment`, but its useful direction is inverted relative to the naive design. The raw feature was `boundary_edge_ratio * (1 - class_prob)`, and train orientation flipped the sign. This means it should not be inserted as a positive reward without orientation/calibration.

`boundary_phase_coherence` has strong val ranking behavior but fails train P0.7 threshold calibration. It is better treated as a sample-weight/ranking component than as a hard gate.

The pure geometry and density signals are not reliable enough as standalone verifier signals on this proposal cache.

## Recommendation

The quick fusion check used `score_edge_alignment`, `boundary_phase_coherence`, `interior_exterior_texture_contrast`, and the raw-iFFT recipe. Balanced logistic fusion improved fixed-threshold transfer to P=0.812/R=0.317, so the next training-side integration should use this fused score as a soft sample weight or pairwise ranker, not as a hard rescue gate.

## Fusion Sweep Addendum

After the implementation review, a broader fusion sweep was added to `fusion_sweep.csv`.

Best fixed-threshold transfer:

| Group | Method | Features | Val AUC | Val AP | Fixed Val P | Fixed Val R |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| new_top3_plus_raw_ifft_recipe | logistic C=0.25 | 4 | 0.836 | 0.451 | 0.812 | 0.317 |
| new_top3_plus_raw_ifft_individual3 | logistic C=0.05 | 6 | 0.831 | 0.435 | 0.846 | 0.268 |
| raw_ifft_individual3 | logistic C=0.05 | 3 | 0.667 | 0.341 | 1.000 | 0.122 |

Using all seven new interpretable signals or all 115 legacy iFFT features reduced fixed-threshold transfer quality. The useful combination is narrow: the top three new signals plus the existing raw-iFFT recipe.
