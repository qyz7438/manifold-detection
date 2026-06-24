# RLIimage Map

## Core

- [[Architecture]]
- [[Versioning]]
- [[Directory Refactor]]
- [[Canonical Runner]]
- [[Experiment Registry]]

## Methods

- [[AFM]]
- [[RLVR]]
- [[DPO]]
- [[Signals]]
- [[Segmentation]]

## Tasks

- [[Detection Task]]
- [[Segmentation Task]]

## Active Lines

- [[NWPU Clean Posttrain]]
- [[DPO Short Sweep 2218]]
- [[AFM Detection Evidence]]
- [[Segmentation Roadmap]]

## Key Claims

- In-network FFT/AFM has evidence on detection.
- External FFT verifier as scalar reward was weak in early Penn-Fudan experiments.
- NWPU clean eval pollution affected many historical runs before the canonical clean eval fix.
- Current DPO work should be judged on clean 5-epoch short runs before any long training.

## Source Documents

- [Architecture](../docs/architecture.md)
- [Versioning Scheme](../docs/versioning_scheme.md)
- [Directory Refactor Plan](../docs/refactor_directory_versioning_plan.md)
- [Segmentation Technical Plan](../docs/segmentation_technical_plan.md)
- [DPO Short Sweep Summary](../docs/dpo_short_sweep_2218_2220_summary.md)
- [Eval Pollution Audit](../docs/round2129_eval_pollution_audit.md)
