# Action-Local HWM NWPU Full-Val Result

Date: 2026-07-09
Dataset: NWPU VHR-10
Detector: frozen Faster R-CNN MobileNetV3 FPN checkpoints
Training: action-local ROI transport head only, 4 epochs
Evaluation: full validation split, no `limit_val`

## Candidate configuration

- `box_base=decoded`
- `match_mode=class_agnostic`
- `hwm_weight=0.05`
- `hwm_epsilon=0.01`
- `high_iou_preserve_weight=2.0`
- `gate_actions=false`
- `num_workers=0` on the remote server to avoid DataLoader file-descriptor failures

This configuration keeps the detector frozen and learns bounded local ROI actions.
The high-water-mark teacher anchors the action head to the best AP75 epoch, while
the high-IoU preserve loss prevents later epochs from damaging already-good boxes.

## Full-val metrics

| Seed | Run | Default AP50 | Final AP50 | Delta AP50 | Default AP75 | Final AP75 | Delta AP75 | Best AP75 | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | `remote_hwm_preserve2_fullnw0_nwpu_s42_bs8_ep4` | 0.5188 | 0.5318 | +0.0130 | 0.1922 | 0.2211 | +0.0288 | 0.2269 | 1 |
| 2024 | `remote_hwm_preserve2_fullnw0_nwpu_s2024_bs8_ep4` | 0.5332 | 0.5480 | +0.0148 | 0.1468 | 0.2052 | +0.0584 | 0.2079 | 3 |
| 999 | `remote_hwm_preserve2_fullnw0_nwpu_s999_bs8_ep4` | 0.5294 | 0.5461 | +0.0167 | 0.1292 | 0.2039 | +0.0748 | 0.2039 | 4 |

Mean over the three seeds:

| Metric | Default | Final | Delta |
|---|---:|---:|---:|
| AP50 | 0.5271 | 0.5420 | +0.0148 |
| AP75 | 0.1561 | 0.2101 | +0.0540 |
| Precision | 0.3179 | 0.3817 | +0.0638 |
| Recall | 0.6282 | 0.6299 | +0.0017 |
| False-positive rate | 0.6821 | 0.6183 | -0.0638 |
| Prediction count | 2309.0 | 1922.0 | -387.0 |
| ECE | 0.0635 | 0.0556 | -0.0079 |

Mean best-final AP75 gap: 0.0029.

## Interpretation

The improvement is not an AP75-only tradeoff. AP75, AP50, precision, false-positive
rate, prediction count, and ECE all move in the desired direction, while recall is
approximately preserved. This supports the current thesis that action-local
transport is learning conservative local corrections rather than flooding NMS with
rescued predictions.

The earlier drift was mainly caused by weak protection of already high-IoU
proposals. Increasing `high_iou_preserve_weight` to 2.0 was more effective than
only strengthening the high-water-mark anchor.

## Reproduction command

Example for seed 2024:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_energy_transport_action.py \
  --run-name remote_hwm_preserve2_fullnw0_nwpu_s2024_bs8_ep4 \
  --dataset nwpu \
  --checkpoint runs/nwpu_mob_baseline_s2024_12ep/checkpoint_best.pth \
  --seed 2024 \
  --epochs 4 \
  --batch-size 8 \
  --num-workers 0 \
  --box-base decoded \
  --match-mode class_agnostic \
  --hwm-weight 0.05 \
  --hwm-epsilon 0.01 \
  --high-iou-preserve-weight 2.0
```
