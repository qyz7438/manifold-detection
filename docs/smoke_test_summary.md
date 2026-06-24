# Smoke Test Summary

Date: 2026-06-03

Environment:

- Conda env: `RLimage`
- Python: `E:\anaconda\01\envs\RLimage\python.exe`
- Torch: `2.1.0+cu121`
- Torchvision: `0.16.0+cu121`
- CUDA available: `true`

Commands verified:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.train_baseline --config mfvpt/configs/smoke.yaml --run-name smoke_baseline --limit-train 64 --limit-val 32 --epochs 1
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.post_train --config mfvpt/configs/smoke.yaml --baseline runs/smoke_baseline/checkpoint_last.pth --run-name smoke_posttrain --limit-train 32 --limit-val 32 --epochs 1
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.eval --config mfvpt/configs/smoke.yaml --checkpoint runs/smoke_baseline/checkpoint_last.pth --run-name smoke_eval_baseline --limit-val 32
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.eval --config mfvpt/configs/smoke.yaml --checkpoint runs/smoke_posttrain/checkpoint_last.pth --run-name smoke_eval_posttrain --limit-val 32
E:\anaconda\01\envs\RLimage\python.exe -m mfvpt.visualize --config mfvpt/configs/smoke.yaml --baseline runs/smoke_baseline/checkpoint_last.pth --ours runs/smoke_posttrain/checkpoint_last.pth --run-name smoke_visualize --limit-val 16
```

Note: after the refined method naming update, post-training logs use `loss_view_consistency` instead of `loss_consistency`.

Unit tests:

```text
13 passed
```

Post-train smoke metrics:

```json
{
  "clean_acc": 0.0625,
  "low_acc": 0.09375,
  "high_acc": 0.0625,
  "patch_acc": 0.0625,
  "cons_low": 0.90625,
  "cons_high": 1.0,
  "cons_patch": 0.78125,
  "hce_clean": 0.0,
  "hce_low": 0.0,
  "hce_high": 0.0,
  "hce_patch": 0.0,
  "ece_clean": 0.035090990364551544,
  "ece_low": 0.066431425511837,
  "ece_high": 0.03507808595895767,
  "ece_patch": 0.03471405804157257
}
```

Generated local artifacts:

- `runs/smoke_baseline/checkpoint_last.pth`
- `runs/smoke_posttrain/checkpoint_last.pth`
- `runs/smoke_eval_baseline/eval_metrics.json`
- `runs/smoke_eval_posttrain/eval_metrics.json`
- `runs/smoke_visualize/perturbation_grid.png`
- `runs/smoke_visualize/prediction_compare.md`

Generated artifacts are intentionally ignored by git because they include checkpoints and run outputs.
