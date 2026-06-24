# Signals

Signal families:

- FFT/raw iFFT features,
- high-dimensional manifold features,
- geometry,
- IoU oracle / GT-verifiable reward,
- confidence calibration,
- NMS fate tracing.

Important distinction:

Signals can be useful offline but fail online if they do not enter the gradient path strongly enough.

Related:

- [[AFM]]
- [[RLVR]]
- [[DPO]]
- [[Experiment Registry]]
