# DeepSeek V4 Pro Research Reviews

Date: 2026-07-09
Project line: action-local ROI transport for detection

This note records external DeepSeek V4 Pro reviews requested after completed
experiment groups. The reviews are advisory; final experiment decisions remain
with Codex and the project evidence.

## Review 1: HWM action-local NWPU full-val group

Reviewed group:

- `remote_hwm_preserve2_fullnw0_nwpu_s42_bs8_ep4`
- `remote_hwm_preserve2_fullnw0_nwpu_s2024_bs8_ep4`
- `remote_hwm_preserve2_fullnw0_nwpu_s999_bs8_ep4`

Key metrics over three seeds:

- AP50: 0.5271 -> 0.5420, delta +0.0148
- AP75: 0.1561 -> 0.2101, delta +0.0540
- Precision: 0.3179 -> 0.3817
- Recall: 0.6282 -> 0.6299
- False-positive rate: 0.6821 -> 0.6183
- Prediction count: 2309.0 -> 1922.0
- ECE: 0.0635 -> 0.0556

DeepSeek's judgment:

- The improvement is likely real, but the main driver may be supervised proposal
  box refinement plus high-IoU preservation rather than the HWM teacher itself.
- The strongest confound is that the action head receives direct GT box-delta
  supervision on ROI features already used by the detector's box regressor.
- `class_agnostic` rematching is another possible hidden lever and should be
  treated as part of the method, not an incidental detail.
- HWM has not yet been proven causal; `hwm_weight=0.05` is small relative to
  `box_weight=1.0` and `high_iou_preserve_weight=2.0`.

Recommended next experiment:

- Run a loss-causal ablation group:
  1. box loss only
  2. box loss plus preserve
  3. box loss plus HWM
  4. current best
- Success criterion: identify whether box-only recovers most of the AP75 gain,
  whether preserve explains stability, and whether HWM adds measurable final or
  best-final-gap value.

## Review 2: boxRLVR/proposal-action single-seed group

Reviewed group:

- `remote_boxrlvr_ms_zero_s999_ltr256_fullval_3ep`

Reference default for seed 999:

- AP50: 0.5294
- AP75: 0.1292
- Precision: 0.3455
- Recall: 0.6294
- False-positive rate: 0.6545
- ECE: 0.0624
- Prediction count: 2133

boxRLVR final:

- AP50: 0.5318
- AP75: 0.1666
- Precision: 0.3463
- Recall: 0.6302
- False-positive rate: 0.6537
- ECE: 0.0617
- Prediction count: 2131
- Best AP75: 0.1669 at epoch 1

DeepSeek's judgment:

- The result is a weak structure signal rather than pure noise: AP75 improved by
  about +0.037 on the seed-999 reference.
- It is not yet evidence that the RLVR formulation is working well, because
  AP50, precision, false-positive rate, ECE, and prediction count barely moved.
- `score_scale=0` prevents decision-level score correction, which likely limits
  detector-level impact even if boxes improve.
- `move_penalty=0` means the current formulation does not yet realize the
  original minimal-energy-action idea.
- Epoch-1 saturation suggests a sparse or flat local reward landscape.

Recommended diagnostic before more boxRLVR training:

- Run an offline reward-landscape sensitivity scan around proposals:
  sample many box deltas per proposal, measure IoU reward standard deviation,
  best-minus-mean reward gap, and the rank of the zero action across IoU buckets.

Recommended priority:

1. HWM loss-causal ablation
2. boxRLVR reward-landscape diagnostic
3. stronger-baseline comparison

## Review 3: HWM loss-causal ablation

Reviewed group:

- `abl_hwm_loss_boxonly_e0_fullnw0_nwpu_s42_bs8_ep4`
- `abl_hwm_loss_box_preserve2_e0_fullnw0_nwpu_s42_bs8_ep4`
- `abl_hwm_loss_box_hwm_e0_fullnw0_nwpu_s42_bs8_ep4`
- `abl_hwm_loss_current_e001_fullnw0_nwpu_s42_bs8_ep4`
- `abl_hwm_loss_boxonly_e0_fullnw0_nwpu_s2024_bs8_ep4`
- `abl_hwm_loss_box_preserve2_e0_fullnw0_nwpu_s2024_bs8_ep4`
- `abl_hwm_loss_box_hwm_e0_fullnw0_nwpu_s2024_bs8_ep4`
- `abl_hwm_loss_current_e001_fullnw0_nwpu_s2024_bs8_ep4`

Mean AP75 deltas:

- `boxonly`: +0.0416
- `box+preserve`: +0.0505
- `box+hwm`: +0.0429
- `current`: +0.0490

DeepSeek's judgment:

- The main contribution is direct supervised box refinement. `boxonly` recovers
  about 85% of the `current` AP75 gain.
- High-IoU preservation is the only meaningful auxiliary term. It adds about
  +0.009 AP75 over `boxonly` and removes the best-final gap.
- HWM is effectively a dead component in this configuration: HWM loss is around
  2e-05 to 3e-05 and does not improve over `box+preserve`.
- Energy is also too small to claim a positive role.

DeepSeek's recommended narrative change:

- Do not lead with "energy-guided" or HWM as the active contribution.
- Reframe the active method as decoupled action-local box refinement with
  high-IoU consistency regularization.
- Keep HWM and energy as exploratory or negative results until new evidence
  shows a causal contribution.

Recommended next experiment:

1. `class_aware` versus `class_agnostic` matching ablation
2. stronger baseline or same-budget box-head fine-tuning comparison
3. seed999 completion only after the mechanism checks are clean
4. boxRLVR reward-landscape diagnostics after the supervised mechanism is clear

## Review 4: match-mode ablation

Reviewed group:

- `abl_match_class_aware_boxonly_e0_fullnw0_nwpu_s42_bs8_ep4`
- `abl_match_class_aware_box_preserve2_e0_fullnw0_nwpu_s42_bs8_ep4`
- `abl_match_class_aware_boxonly_e0_fullnw0_nwpu_s2024_bs8_ep4`
- `abl_match_class_aware_box_preserve2_e0_fullnw0_nwpu_s2024_bs8_ep4`

Comparison against the previous `class_agnostic` ablation:

- `class_aware boxonly`: AP75 delta +0.0431
- `class_agnostic boxonly`: AP75 delta +0.0416
- `class_aware box+preserve`: AP75 delta +0.0507
- `class_agnostic box+preserve`: AP75 delta +0.0505

DeepSeek's judgment:

- The result basically rules out the concern that the gain is caused by
  `class_agnostic` matching relaxation.
- The main candidate should use `class_aware box+preserve` because it is no worse
  on AP75 and is easier to defend methodologically.
- The revised candidate is:

```bash
--box-base decoded \
--match-mode class_aware \
--energy-weight 0.0 \
--hwm-weight 0.0 \
--high-iou-preserve-weight 2.0
```

Next priority from DeepSeek:

1. same-budget standard `box_head_only` fine-tune control
2. seed999 completion for statistical completeness
3. VOC/COCO generalization
4. boxRLVR reward-landscape diagnostics

DeepSeek's summary:

> The method delivers consistent AP75 improvements under class-aware matching,
> ruling out agnostic-relaxation confounds; causal attribution against an
> equal-budget fine-tune control is the next necessary experiment.

## Review 5: boxRLVR/direct box action matrix

Reviewed completed groups:

- `rlvr_zero_ltr256_3ep`
- `rlvr_random_ltr256_3ep`
- `rlvr_spectral_ltr256_3ep`
- `rlvr_spectral_shuffled_ltr256_3ep`
- `rlvr_random_ltr256_10ep`
- `direct_zero_ltr256_3ep`
- `direct_zero_fulltrain_3ep`
- `direct_zero_fulltrain_10ep`
- `rlvr_random_fulltrain_3ep`

Key mean AP75 deltas:

- real RLVR contexts: about +0.026 to +0.032
- shuffled spectral control: about +0.002
- direct zero fulltrain 10ep: about +0.030
- supervised `class_aware box+preserve`: about +0.0507

DeepSeek's judgment:

- RLVR has weak but real signal, because real contexts outperform the shuffled
  spectral control.
- The signal is not clearly RLVR-specific; it mostly comes from the IoU reward
  structure.
- Spectral context does not add value over random or zero context and should not
  remain in the active method path.
- Direct box action and RLVR appear to converge toward a similar +0.03 AP75
  ceiling, still below supervised `box+preserve`.
- The equal-budget standard `box_head_only` fine-tune control remains the next
  required causal experiment.

If RLVR is revisited later, DeepSeek recommended:

1. an offline reward-landscape scan around proposals
2. enabling score actions, such as `score_scale=0.05`
3. adding a small move penalty, such as `move_penalty=0.01`

Do not run another broad RLVR matrix before the box-head control clarifies
whether the supervised action-local gain is more than standard fine-tuning
capacity.

## Review 6: completed `rlvr_random_fulltrain_10ep`

Reviewed group:

- `remote_boxrlvr_full_ms_random_s42_fulltrain_fullval_10ep`
- `remote_boxrlvr_full_ms_random_s2024_fulltrain_fullval_10ep`
- `remote_boxrlvr_full_ms_random_s999_fulltrain_fullval_10ep`

Mean deltas:

- AP50: +0.0063
- AP75: +0.0350
- Best AP75: +0.0359
- Precision: +0.0042
- Recall: +0.0051
- False-positive rate: -0.0042
- Prediction count: -11.3
- ECE: -0.0031
- Mean best epoch: 9.0

DeepSeek's updated judgment:

- The completed 10-epoch group confirms RLVR has weak but real positive signal:
  all three seeds improve AP75.
- The conclusion does not change: the improvement still looks like localization
  micro-adjustment rather than detector-level decision improvement.
- RLVR remains much weaker than supervised `class_aware box+preserve` on
  precision, false-positive-rate reduction, prediction suppression, and ECE.
- The next required causal control is still `box_head_only`.

Action taken:

- The existing 4-epoch two-seed `box_head_only` control remains queued.
- A new 10-epoch three-seed `box_head_only` control is queued after the 4-epoch
  control, to match the RLVR fulltrain 10-epoch budget.
