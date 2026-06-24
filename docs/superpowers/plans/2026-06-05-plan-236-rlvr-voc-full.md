# Plan 2.36: RLVR 后训练 — VOC 全 20 类

> **核心问题:** Penn-Fudan 太小（136 张），RL 探索空间为零。VOC full（5.7K/20 类）提供足够的探索空间验证 RLVR。

---

## Phase 1 — VOC Full 收敛基线

MobV3 × VOC2012 full 20-class × 15 epoch。每 epoch 后记录 val AP。

**预计:** 3-5 epoch 收敛（数据量大，收敛更快），baseline AP50 约 0.45-0.55。

---

## Phase 2 — 2.31 频谱 reward（对照）

从 Phase 1 checkpoint 出发。loss reweighting 在 VOC full 上的表现。

---

## Phase 3 — 2.33/2.35 RLVR

从 Phase 1 checkpoint 出发。REINFORCE/PPO + 频谱 quality reward。在大数据上验证 RL 是否超过 loss reweighting。

---

## 前置修复

VOC 20 类：`round28_train_eval.py` 已有 `--dataset voc`，需确认 20 类 mapping 正确，训练集用 `train`（非 subset）。
