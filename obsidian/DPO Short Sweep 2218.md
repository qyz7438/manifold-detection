# DPO Short Sweep 2218

Active sweep:

- `round2218`: DPO-only weights `0.01 / 0.03 / 0.1`
- `round2219`: DPO plus rescue weights `0.01 / 0.03 / 0.1`
- `round2220`: cls_score-only DPO weights `0.01 / 0.03 / 0.1`

Supervisor:

- `scripts/run_round2218_short_dpo_sweep.py`
- summary: `runs/round2218_short_dpo_sweep_summary.json`

Completed early result:

- `round2218_pre_nms_dpo_only_w001_5ep`: AP75 `0.2980`, delta `+0.0041`
- `round2218_pre_nms_dpo_only_w003_5ep`: AP75 `0.2986`, delta `+0.0047`
- `round2218_pre_nms_dpo_only_w01_5ep`: AP75 `0.2988`, delta `+0.0049`
- `round2219_pre_nms_dpo_rescue_w003_5ep`: AP75 `0.2989`, delta `+0.0050`

Interpretation:

DPO-only shows a small clean positive signal, but it has not shown direct LC-HI rescue. The best current short-run recipe is [[Versioning|det.dpo.smoke.001]]: DPO plus rescue with `pre_nms_dpo_loss_weight=0.03`.

Source summary:

- [docs/dpo_short_sweep_2218_2220_summary.md](../docs/dpo_short_sweep_2218_2220_summary.md)
