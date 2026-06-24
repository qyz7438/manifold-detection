# Segmentation

Segmentation must be a first-class task line.

Source plan:

- [docs/segmentation_technical_plan.md](../docs/segmentation_technical_plan.md)

Method families:

- `seg.afm.*`: in-network FFT modules for dense prediction.
- `seg.rlvr.*`: dense verifiable rewards such as Dice, IoU, boundary IoU.
- `seg.dpo.*`: mask/component/patch preference optimization.

Key difference from detection:

Segmentation should not depend on proposal mining, NMS, or detection confidence thresholds.

Related:

- [[AFM]]
- [[RLVR]]
- [[DPO]]
- [[Segmentation Roadmap]]
