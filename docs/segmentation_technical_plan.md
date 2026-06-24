# Segmentation Technical Plan

The project should treat segmentation as a first-class task line, not as a side experiment. Segmentation has different signal geometry from detection: dense masks provide pixel-level supervision, while detection post-training depends on proposals, matching, NMS, and confidence calibration.

## Segmentation AFM

Goal: test whether in-network FFT modules help dense prediction when inserted at spatially aligned feature maps.

Candidate insertion points:

- FCN classifier input.
- DeepLab ASPP input.
- U-Net bottleneck.
- Mask head features if using Mask R-CNN.

Variants:

- `seg.afm.proto.001`: MicroAFM before classifier.
- `seg.afm.proto.002`: phase-only AFM.
- `seg.afm.proto.003`: multiscale AFM with weak gate.
- `seg.afm.proto.004`: AFM with feature consistency constraint.

Metrics:

- mIoU
- boundary F1
- ECE for pixel probabilities
- foreground recall
- small-object mask IoU

Known risk: early FCN experiments showed AFM did not improve mIoU. This may be architecture-dependent, so segmentation AFM should focus on boundary metrics instead of only mIoU.

## Segmentation RLVR

Goal: use verifiable dense rewards where the reward is mask quality rather than proposal quality.

Possible rewards:

- IoU reward against GT mask.
- Boundary IoU reward.
- Dice/F1 reward.
- Connected-component correctness.
- Frequency-domain boundary consistency.

Implementation direction:

1. Freeze baseline segmenter.
2. Run policy segmenter.
3. Compute dense reward map or region-level reward.
4. Apply KL anchor against baseline logits.
5. Use safety guards: foreground area ratio, false positive component count, calibration drift.

Candidate versions:

- `seg.rlvr.proto.001`: dense IoU/Dice reward with KL anchor.
- `seg.rlvr.proto.002`: boundary reward with AFM features.
- `seg.rlvr.proto.003`: component-level verifiable reward.

## Segmentation DPO

Goal: compare two masks, crops, or components and prefer the one with better verifiable quality.

Pair construction:

- Chosen: higher IoU mask or component.
- Rejected: lower IoU mask or component from same image/class.
- Optional no-GT variant: chosen has stronger boundary/spectral consistency and stable baseline confidence.

Loss:

```text
loss = -logsigmoid(beta * ((logp_chosen - logp_rejected) - (ref_chosen - ref_rejected)))
```

Candidate versions:

- `seg.dpo.proto.001`: per-image mask DPO using GT IoU.
- `seg.dpo.proto.002`: connected-component DPO.
- `seg.dpo.proto.003`: boundary patch DPO.
- `seg.dpo.proto.004`: AFM-feature-guided DPO.

## Segmentation Data Flow

```text
image
  -> baseline segmenter logits
  -> policy segmenter logits
  -> mask/component/patch candidates
  -> verifiable quality score
  -> RLVR reward or DPO pair
  -> KL/safety-guarded update
  -> clean segmentation eval
```

## Minimum Viable Segmentation Runner

The first canonical segmentation runner should support:

- dataset config,
- model config,
- checkpoint config,
- clean eval config,
- objective config: `supervised`, `afm`, `rlvr`, `dpo`,
- metadata recording.

It should not reuse detection proposal code.
