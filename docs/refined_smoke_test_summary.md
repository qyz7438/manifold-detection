# Refined Smoke Test Summary

Date: 2026-06-03

Purpose:

- Verify the refined method framing where Fourier only generates frequency views.
- Verify post-training logs use `loss_view_consistency`.
- Verify the full 1 epoch smoke chain still runs after renaming config and loss fields.

Commands verified:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.train_baseline --config mfvpt/configs/smoke.yaml --run-name refined_smoke_baseline --limit-train 64 --limit-val 32 --epochs 1
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.post_train --config mfvpt/configs/smoke.yaml --baseline runs/refined_smoke_baseline/checkpoint_last.pth --run-name refined_smoke_posttrain --limit-train 32 --limit-val 32 --epochs 1
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.eval --config mfvpt/configs/smoke.yaml --checkpoint runs/refined_smoke_posttrain/checkpoint_last.pth --run-name refined_smoke_eval_posttrain --limit-val 32
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.visualize --config mfvpt/configs/smoke.yaml --baseline runs/refined_smoke_baseline/checkpoint_last.pth --ours runs/refined_smoke_posttrain/checkpoint_last.pth --run-name refined_smoke_visualize --limit-val 16
```

Unit tests:

```text
13 passed
```

Post-train loss log:

```json
{
  "loss_total": 4.26608681678772,
  "loss_ce": 4.2523956298828125,
  "loss_view_consistency": 0.0010081841974169947,
  "loss_confidence": 0.025365993613377213,
  "epoch": 1,
  "val_clean_acc": 0.0625
}
```

Post-train eval metrics:

```json
{
  "clean_acc": 0.0625,
  "low_acc": 0.09375,
  "high_acc": 0.0625,
  "patch_acc": 0.0625,
  "cons_low": 0.90625,
  "cons_high": 1.0,
  "cons_patch": 0.75,
  "hce_clean": 0.0,
  "hce_low": 0.0,
  "hce_high": 0.0,
  "hce_patch": 0.0,
  "ece_clean": 0.03510143607854843,
  "ece_low": 0.06645344197750092,
  "ece_high": 0.035090573132038116,
  "ece_patch": 0.03465292602777481
}
```

Generated local artifacts:

- `runs/refined_smoke_baseline/checkpoint_last.pth`
- `runs/refined_smoke_posttrain/checkpoint_last.pth`
- `runs/refined_smoke_eval_posttrain/eval_metrics.json`
- `runs/refined_smoke_visualize/perturbation_grid.png`
- `runs/refined_smoke_visualize/prediction_compare.md`
