# DPO

DPO is pairwise preference optimization.

Detection variants:

- pre-NMS cls_score DPO,
- DPO plus rescue,
- cls_score-only DPO,
- bbox/action DPO as a later step.

Current active sweep:

- [[DPO Short Sweep 2218]]

Judgment criteria:

- clean AP75,
- DPO pair count,
- DPO gradient into intended module,
- LC-HI score delta,
- safety guards: prediction count, FP rate, ECE.

Related:

- [[RLVR]]
- [[Signals]]
- [[NWPU Clean Posttrain]]
