# RLVR

RLVR is verifier-guided post-training.

Current detection pattern:

```text
baseline rollout -> proposals -> verifier/reward/rescue signal -> KL-anchored update -> clean eval
```

Key risks:

- sparse LC-HI reward,
- verifier precision high but recall low,
- KL/detection loss can dominate policy gradients,
- eval pollution can hide failures.

Related:

- [[Signals]]
- [[NWPU Clean Posttrain]]
- [[DPO]]
- [[Canonical Runner]]
