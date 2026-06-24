# Architecture

Project architecture is tracked in [[RLIimage Map]] and the source document:

- [docs/architecture.md](../docs/architecture.md)

Main runtime boundary:

```text
config -> model/checkpoint validation -> train/posttrain -> clean eval -> metadata -> registry
```

Current canonical package paths:

- `spectral_detection_posttrain.core`
- `spectral_detection_posttrain.methods.afm`
- `spectral_detection_posttrain.methods.rlvr`
- `spectral_detection_posttrain.methods.dpo`
- `spectral_detection_posttrain.signals.fft`
- `spectral_detection_posttrain.trainers.detection`

The target structure separates:

- [[AFM]]
- [[RLVR]]
- [[DPO]]
- [[Signals]]
- [[Segmentation]]
- [[Experiment Registry]]
- [[Directory Refactor]]

Legacy paths are compatibility shims until the archive pass is complete.
