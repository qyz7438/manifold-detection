# Round 2.226: NWPU Oracle Continuous Reward Re-ranking

## Objective

Establish an upper bound for AP75 improvement on NWPU VHR-10 if we had a perfect
LC-HI signal.  We disable the detector's internal NMS, compute an oracle reward
per proposal from ground-truth IoU, re-score proposals with several fusion
strategies, re-run class-wise NMS, and re-evaluate.

This is a Phase 0 sanity check for the RLVR/DPO rescue direction: if even an
oracle reward does not lift AP75, the problem is not the reward model.

## Method

- Baseline: `runs/round2100_nwpu_baseline/checkpoint_best.pth`
- Dataset: NWPU VHR-10 full validation split (no `limit_val`)
- Detector config: Faster R-CNN MobileNetV3-Large-320-FPN, 11 classes, max size 480
- Proposal extraction: `score_thresh=0.0`, `nms_thresh=1.0`, `detections_per_img=1000`
- Oracle reward: `R = IoU(box, GT) * (1 - score)`
- Re-scoring strategies tested:
  - `baseline`: original detector score
  - `iou_only`: replace score with IoU
  - `oracle_replace`: replace score with `R`
  - `oracle_add`: `score + alpha * R` for `alpha` in {0.1, 0.3, 0.5, 0.7, 0.9}
  - `oracle_mul`: `score^alpha * R^(1-alpha)` for `alpha` in {0.1, 0.3, 0.5, 0.7, 0.9}
- Post-processing: class-wise NMS at IoU 0.5, eval score threshold 0.05

## Key Results

| Strategy | alpha | AP50 | AP75 | #Pred | FP Rate | dAP50 | dAP75 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | - | 0.6548 | **0.2939** | 1425 | 0.478 | - | - |
| oracle_add | 0.1 | 0.6671 | 0.2988 | 9641 | 0.919 | +0.012 | +0.005 |
| oracle_add | 0.3 | 0.6678 | 0.3099 | 17829 | 0.956 | +0.013 | +0.016 |
| oracle_add | 0.5 | 0.6657 | 0.3153 | 19029 | 0.958 | +0.011 | +0.021 |
| oracle_add | 0.7 | 0.6585 | 0.3320 | 19538 | 0.959 | +0.004 | +0.038 |
| **oracle_add** | **0.9** | **0.6310** | **0.3459** | **19811** | **0.960** | **-0.024** | **+0.052** |
| oracle_mul | 0.5 | 0.6681 | 0.2653 | 1846 | 0.580 | +0.013 | -0.029 |
| oracle_mul | 0.9 | 0.6774 | 0.2984 | 1389 | 0.456 | +0.023 | +0.004 |
| iou_only | - | 0.1192 | 0.0918 | 19757 | 0.959 | -0.536 | -0.202 |
| oracle_replace | - | 0.0407 | 0.0085 | 19646 | 0.963 | -0.614 | -0.285 |

Best result: `oracle_add` with `alpha=0.9` gives **AP75 = 0.3459 (+0.0520)**.

## Interpretation

1. **Oracle reward has a real, but bounded, ceiling.**  A perfect LC-HI signal
can lift AP75 by about +5% on NWPU, but at the cost of AP50 (-2.4%) and a
massive increase in the number of predictions (14x).  This tells us reward
design is important, but naive rescue is not enough.

2. **Additive fusion outperforms multiplicative fusion.**  `oracle_add` is
monotonically better as `alpha` increases, while `oracle_mul` is unstable and
often worse than baseline.  For RLVR/DPO, additive score adjustments are the
safer action space.

3. **Replacing score with IoU/oracle is catastrophic.**  `iou_only` and
`oracle_replace` collapse AP75.  The detector's original score must remain the
dominant term; any rescue signal should act as a residual correction.

4. **The rescue ceiling comes with a calibration cost.**  High `alpha` rescues
many LC-HI boxes but also lets through many false positives, hurting AP50 and
FP rate.  A production rescue policy must constrain the number of rescued boxes
or explicitly penalize false positives.

## Implications for RLVR/DPO

- **Use additive score adjustments**, not score replacement.
- **Keep alpha small-to-moderate** (target dAP75 ~+0.02 to +0.03 without AP50
  collapse) rather than chasing the oracle upper bound.
- **Constrain the rescue budget**: only adjust scores for boxes that pass a
  verifier gate, and/or add a KL anchor to the baseline score distribution.
- **Reward should include a calibration term**: reward low-confidence high-IoU
  boxes, but also penalize rescuing low-IoU boxes.
- The oracle ceiling of +5.2% AP75 is the target; any learned RLVR/DPO policy
  that achieves +2-3% AP75 without AP50 collapse is a success.

## Next Step

Round 2.227 will train a GRPO policy that predicts additive score adjustments
for LC candidates, using a structured verifier-weighted reward:

```
R = w_verifier * [ IoU + (1 - s_old) * IoU - s_old * (1 - IoU) ]
```

with group-relative baseline and KL anchor to the baseline detector.

## Artifacts

- Script: `scripts/round2226_nwpu_oracle_rerank.py`
- Full-val report: `runs/round2226_nwpu_oracle_rerank_full/oracle_rerank_report.json`
- Smoke report: `runs/round2226_nwpu_oracle_rerank_smoke/oracle_rerank_report.json`
