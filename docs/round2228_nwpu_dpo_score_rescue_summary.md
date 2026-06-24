# Round 2.228: NWPU DPO Pair-Wise Score-Rescue Policy

## Objective

Use Direct Preference Optimization (DPO) to learn a score-rescue policy.  For
each image we construct preference pairs of low-confidence candidates:

- preferred: IoU >= 0.5
- dispreferred: IoU <= 0.3

The policy is trained so that the preferred box receives a higher rescued score
than the dispreferred box.

## Method

- Baseline: `runs/round2100_nwpu_baseline/checkpoint_best.pth`
- Dataset: NWPU VHR-10 smoke split (16 train, 32 val, seed 42)
- Detector: frozen Faster R-CNN MobileNetV3-Large-320-FPN
- Trainable head: 2-layer MLP outputting deterministic `delta` in [-0.3, 0.3]
- Preference pairs: Cartesian product of LC-HI (IoU>=0.5) and LC-LI (IoU<=0.3)
  within each image
- DPO loss: `-log sigmoid(beta * ((s_pos - s_neg) - (s_pos_ref - s_neg_ref)))`
- Reference policy: frozen initial policy
- Hyperparameters: lr=1e-3, beta=0.1, max_delta=0.3, 3 epochs

## Results

| Epoch | Train Loss | Pair Acc | Pairs | Val AP50 | Val AP75 | #Pred | FP Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (baseline) | - | - | - | 0.4705 | 0.1811 | 368 | 0.614 |
| 1 | 0.6855 | 0.942 | 722337 | 0.4025 | 0.1771 | 1851 | 0.843 |
| 2 | 0.6706 | 0.975 | 722337 | 0.4102 | 0.1776 | 2583 | 0.887 |
| 3 | 0.6680 | 0.981 | 722337 | 0.4062 | 0.1768 | 2092 | 0.874 |

Final delta AP75: **-0.0043**.

## Interpretation

1. **DPO successfully learns the preference task.**  Pair accuracy reaches
   ~98%, so the policy reliably ranks LC-HI boxes above LC-LI boxes.

2. **Preference accuracy does not translate to AP75 gain.**  The rescue
   operation increases the number of predictions dramatically (368 -> 2000+)
   and introduces many false positives.  The policy raises the score of many
   LC-LI boxes as well, even though they are labeled "dispreferred".

3. **DPO only enforces relative ordering.**  It does not enforce that
   dispreferred boxes stay below the detection threshold.  As a result, a large
   number of low-IoU boxes cross the eval threshold and pollute the output.

4. **Smoke split may be too small** for the policy to generalize the right
   magnitude of rescue.  The high pair accuracy is achieved by small score
   deltas that still hurt NMS ranking.

## Lessons for Next Round

To make DPO rescue work, we need to add an **absolute constraint** in addition
to the relative preference:

- **Rescue budget**: only adjust the top-K highest-IoU LC candidates per image.
- **Threshold preservation**: penalize any LC-LI box that crosses the original
  score threshold after rescue.
- **Stronger reference / KL**: keep dispreferred boxes close to their baseline
  scores.
- **Use verifier consensus** to select which LC candidates enter the preference
  set, reducing pair count and noise.

## Next Step

Round 2.229 will combine DPO with a **rescue budget and absolute threshold
preservation**:

- Only rescue LC candidates that pass a verifier gate (e.g., verifier score
  > 0.6).
- Add a loss term that penalizes rescued LC-LI boxes for crossing the detection
  threshold.
- Cap the number of rescued boxes per image.

## Artifacts

- Script: `scripts/round2228_nwpu_dpo_score_rescue.py`
- Run: `runs/round2228_nwpu_dpo_score_rescue_smoke`
