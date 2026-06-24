# DPO Short Sweep 2218-2220 Summary

## Context

This sweep tested whether DPO contributes a clean short-run signal on NWPU before committing to 10+ epoch training.

Baseline AP75:

```text
0.2939084504400764
```

The sweep used full train/full val, clean eval settings, and 5 epochs per run.

## Best Configuration

New version alias:

```text
det.dpo.smoke.001
```

Historical run:

```text
round2219_pre_nms_dpo_rescue_w003_5ep
```

Recipe:

- round2211 rescue configuration
- `pre_nms_dpo_loss_weight=0.03`
- `pre_nms_topk_per_gt=2`
- `pre_nms_dpo_max_pairs_per_gt=2`
- `trainable_mode=predictor`
- `lr=1e-4`
- `epochs=5`

Result:

| Run | Setup | AP75 | Delta vs baseline |
|---|---:|---:|---:|
| `round2219_pre_nms_dpo_rescue_w003_5ep` | DPO + rescue, weight 0.03 | `0.298896` | `+0.004988` |
| `round2218_pre_nms_dpo_only_w01_5ep` | DPO-only, weight 0.1 | `0.298816` | `+0.004907` |
| `round2218_pre_nms_dpo_only_w003_5ep` | DPO-only, weight 0.03 | `0.298642` | `+0.004734` |

## Interpretation

DPO has a real but small short-run positive signal. DPO-only improves AP75 monotonically with weight from `0.01` to `0.1`, but it does not directly rescue LC-HI samples.

DPO + rescue at weight `0.03` is the best current tradeoff. It shows a verifier-positive LC-HI score lift:

```text
verifier_positive_lchi_prob_delta_mean = 0.009477330105645316
```

However, the total LC-HI score shift is still negative:

```text
lchi_prob_delta_mean = -0.0020188747382745512
```

So the current bottleneck is not only DPO strength. The bottleneck is coverage: the verifier-positive subset moves, but the broader LC-HI pool does not.

## Negative Result

`cls_score`-only DPO underperformed.

Best cls-score-only AP75:

```text
0.29581237153139567
```

Conclusion:

Strict `cls_score`-only training does not provide enough capacity for this objective. The predictor/adapter path should remain enabled for the next DPO smoke tests.

## Promotion Rule

`det.dpo.smoke.001` should become the default short-run DPO smoke recipe, but not yet a validated recipe.

Validation requires at least one of:

- repeated seed or repeated clean run,
- same-epoch improvement over the round2211 curve,
- short-run AP75 clearly above `0.300`,
- improved LC-HI score shift without FP/ECE regression.
