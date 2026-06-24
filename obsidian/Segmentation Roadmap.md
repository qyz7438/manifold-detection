# Segmentation Roadmap

Source:

- [docs/segmentation_technical_plan.md](../docs/segmentation_technical_plan.md)

Initial version sequence:

1. `seg.afm.proto.001`: MicroAFM before segmentation classifier.
2. `seg.afm.proto.002`: phase-only AFM.
3. `seg.rlvr.proto.001`: dense IoU/Dice reward with KL anchor.
4. `seg.rlvr.proto.002`: boundary reward.
5. `seg.dpo.proto.001`: per-image mask DPO using GT IoU.
6. `seg.dpo.proto.002`: connected-component DPO.

Do not reuse detection proposal assumptions for segmentation.
