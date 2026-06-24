# AFM

AFM is the in-network FFT path.

Core idea:

FFT inside the forward pass can receive useful gradients through normal supervised losses. This is different from external FFT verifier rewards, which were weak in earlier experiments.

Known evidence:

- Penn-Fudan detection had strong AFM evidence.
- Segmentation FCN experiments were negative or inconclusive.
- Phase-only variants appeared important in historical detection runs.

Related:

- [[Detection Task]]
- [[Segmentation]]
- [[Signals]]
- [docs/segmentation_technical_plan.md](../docs/segmentation_technical_plan.md)
