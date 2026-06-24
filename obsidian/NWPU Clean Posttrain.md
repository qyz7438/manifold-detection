# NWPU Clean Posttrain

Current reliable baseline:

- baseline AP75 around `0.2939`

Known clean improvements:

- `round2211`: lr `1e-4`, AP75 around `0.3026` at 15 epochs.
- `round2214`: same recipe extended to 30 epochs, best AP75 around `0.3066` at epoch 23.

Current policy:

Prioritize short 5-epoch experiments for new mechanisms. Avoid 10+ epoch runs until a short-run signal is promising.

Related:

- [[DPO Short Sweep 2218]]
- [[RLVR]]
- [[DPO]]
- [[Experiment Registry]]
