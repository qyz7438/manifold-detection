# Active Reports Index

This directory holds experiment plans, interim results, and analysis reports.
Obsolete round artifacts are archived in `docs/reports/archive/`.

## Plans and Roadmaps

- `top_tier_experiment_roadmap.md` — 顶刊路线图：VOC/NWPU/COCO 实验矩阵、对比基线、消融计划与可视化清单。
- `voc_matrix_plan.md` — VOC 20-class 主矩阵与对比矩阵设计。

## Interim Results

- `voc_matrix_interim_results.md` — VOC 矩阵已完成的组及初步 AP/AP75/ECE 结果。

## NWPU VHR-10

- `nwpu_strong_native_candidate_energy_review_2026-07-10.md` - strong-baseline, native-parity, discrete candidate-energy, spatial-feature, dense-gain, and context-only decision record.

- `nwpu_c0_native_parity_review_2026-07-10.md` - active baseline-convergence and native-parity decision record.

- `nwpu_matrix_analysis.md` / `nwpu_matrix_summary.txt` — NWPU 主矩阵分析。
- `nwpu_mob_480x800_analysis.md` / `nwpu_mob_480x800_summary.txt` — MobileNet 480×800 分辨率矩阵分析。

## Aggregated / Tooling

- `aggregate_results.py` (in `scripts/`) produces cross-run markdown tables.
- `eval_per_size.py` produces per-size AP breakdowns.
- `profile_model.py` produces efficiency profiles.

## Historical Archives

See `docs/reports/archive/` for older Penn-Fudan round summaries and deprecated plans.
