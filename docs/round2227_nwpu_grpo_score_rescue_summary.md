# Round 2.227: NWPU GRPO Score-Rescue Policy

## Objective

Train a small neural policy that predicts an additive score adjustment for
low-confidence proposals.  The policy sees ROI box features and is optimized
with GRPO (group-relative policy optimization) plus a KL anchor to the initial
policy.

## Method

- Baseline: `runs/round2100_nwpu_baseline/checkpoint_best.pth`
- Dataset: NWPU VHR-10 smoke split (16 train images, 32 val images, seed 42)
- Detector: frozen Faster R-CNN MobileNetV3-Large-320-FPN
- Trainable head: 2-layer MLP policy outputting Gaussian `delta ~ N(mu, sigma)`
- Action: `score_new = score_old + delta`, clamped to [0, 1]
- Low-confidence candidates: `score_old < 0.5`
- Reward (final version):
  - `target_delta = sign(IoU - 0.5) * max_delta`
  - `R_action = -((delta - target_delta)^2) * |2*IoU - 1|`
  - `R_oracle = 0.5 * IoU * (1 - score_old)`
  - `R = R_action + R_oracle`
- GRPO: advantage = (R - image_mean(R)) / image_std(R)
- KL anchor to initial policy, weight 0.1
- Hyperparameters: lr=1e-3, max_delta=0.3, K=4 samples, 3 epochs

## Results

| Epoch | Train Loss | Active LC | Mean Delta | Std Delta | Val AP50 | Val AP75 | #Pred | FP Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (baseline) | - | - | - | - | 0.4705 | 0.1811 | 368 | 0.614 |
| 1 | -0.0154 | 667.6 | 0.0056 | 0.0509 | 0.4679 | 0.1811 | 355 | 0.589 |
| 2 | -0.0095 | 667.6 | 0.0012 | 0.0501 | 0.4689 | 0.1811 | 360 | 0.594 |
| 3 | -0.0130 | 667.6 | 0.0055 | 0.0496 | 0.4690 | 0.1818 | 348 | 0.580 |

Final delta AP75: **+0.0007** (no meaningful improvement).

## Interpretation

1. **The policy barely moves scores.**  Mean predicted `delta` stays near zero
   (~0.005) and the predicted standard deviation remains close to its initial
   value (0.05).  The rescue signal is not strong enough to push scores across
   the NMS/eval threshold.

2. **Smoke data is very sparse.**  With only 16 training images, the policy has
   ~670 low-confidence candidates per epoch, but these may not contain enough
   reliable LC-HI examples to learn a generalizable adjustment.

3. **KL anchor may be too strong.**  A KL weight of 0.1 on a per-sample basis
   can dominate the small GRPO gradient, keeping the policy close to the
   zero-mean initialization.

4. **Reward design still has issues.**  Even the target-based reward, which
   avoids the `delta=0` deadlock, did not produce a useful policy.  The policy
   may need a sharper reward (e.g., only reward top/bottom IoU buckets) or a
   different action space (e.g., direct score re-ranking via ranking loss).

## Next Step

Round 2.228 will try **DPO-style pair-wise ranking** instead of GRPO.  DPO is
naturally suited here because we can construct reliable preferences:

> A low-confidence box with IoU > 0.5 should have a higher rescued score than
> a low-confidence box with IoU < 0.3.

DPO avoids the continuous delta action space and directly optimizes the
relative ordering of boxes, which is what NMS/eval actually cares about.

## Artifacts

- Script: `scripts/round2227_nwpu_grpo_score_rescue.py`
- Run: `runs/round2227_nwpu_grpo_score_rescue_smoke`
- Alt run with target reward: `runs/round2227b_nwpu_grpo_target_reward_smoke`
