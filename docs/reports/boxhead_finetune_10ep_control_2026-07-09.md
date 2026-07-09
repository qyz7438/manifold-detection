# Box-Head 10-Epoch Control Against RLVR

Date: 2026-07-09
Dataset: NWPU VHR-10 full validation
Model: Faster R-CNN MobileNetV3-320-FPN
Seeds: 42, 2024, 999

## Question

This group tests whether the longer supervised `box_head_only` control explains
the apparent action-local and RLVR gains.

All runs start from the corresponding 12-epoch NWPU baseline checkpoint.

## Mean Results

All rows are mean deltas over seeds 42, 2024, and 999 versus the corresponding
baseline checkpoint.

| Group | AP50 delta | AP75 delta | Best AP75 delta | Precision delta | Recall delta | FPR delta | ECE delta | Prediction delta | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_zero_fulltrain_10ep` | +0.0047 | +0.0304 | +0.0308 | +0.0023 | +0.0039 | -0.0023 | -0.0014 | -3.0 | 9.3 |
| `rlvr_random_fulltrain_10ep` | +0.0063 | +0.0350 | +0.0359 | +0.0042 | +0.0051 | -0.0042 | -0.0031 | -11.3 | 9.0 |
| `box_head_only_10ep` | +0.0426 | +0.0792 | +0.0858 | +0.0641 | +0.0205 | -0.0641 | +0.0160 | -354.0 | 7.3 |

## Per-Seed `box_head_only_10ep`

| Seed | AP75 delta | Best AP75 delta | Best epoch | ECE delta | Prediction delta |
|---:|---:|---:|---:|---:|---:|
| 42 | +0.0545 | +0.0565 | 7 | +0.0480 | -45 |
| 2024 | +0.0877 | +0.0985 | 7 | -0.0196 | -754 |
| 999 | +0.0953 | +0.1025 | 8 | +0.0195 | -263 |

## Interpretation

`box_head_only_10ep` is much stronger than RLVR and direct box-action training
on AP50 and AP75. It also moves detector-level decision metrics much more:
precision increases, false-positive rate drops, recall rises, and prediction
count falls substantially.

This closes the current independent action-head and RLVR path as the main AP
improvement route. RLVR still has weak positive signal, but it is much lower
information density than ordinary supervised detector fine-tuning.

The unresolved issue is calibration and epoch selection. `box_head_only_10ep`
has a mean ECE delta of `+0.0160`, but the per-seed signs are mixed:
seed 42 worsens strongly, seed 2024 improves, and seed 999 worsens moderately.
The best AP75 epochs are also earlier than the final epoch, around epochs 7-8.

## DeepSeek V4 Pro Review

DeepSeek judged that:

- `box_head_only_10ep` decisively closes independent action-head and RLVR as
  the main AP-improving path.
- The likely explanation is that the detector's ROI box head or the baseline
  training schedule is under-trained after 12 epochs.
- The ECE concern is real enough to track, but not yet statistically proven
  because the per-seed ECE deltas have mixed signs.
- The dangerous overclaim is to call `box_head_only` itself a method; it is a
  baseline adequacy and attribution control.

DeepSeek's next priorities:

1. Run a full-model 10-epoch continuation from the same 12-epoch checkpoints.
   If full-model continuation matches `box_head_only_10ep`, the issue is likely
   general training insufficiency rather than a box-head-specific structure.
2. Test a minimal calibration intervention such as label smoothing only after
   the full-model attribution control is understood.

## Next Action

The full-model continuation control was queued with:

```bash
scripts/experiments/run_full_model_finetune_10ep_control.sh
```

It completed on GPU2 and is now recorded in:

```bash
docs/reports/full_model_finetune_10ep_control_2026-07-09.md
```

The result further weakens the independent action-head story: full-model
continuation reaches `+0.1058` mean AP75, above `box_head_only_10ep` at
`+0.0792`.
