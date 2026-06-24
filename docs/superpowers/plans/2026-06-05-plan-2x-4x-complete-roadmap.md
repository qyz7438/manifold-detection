# Plan 2.27: 后训练有效性质控

> **核心问题:** 2.26 C_5ep 的 AP75=0.674 提升，来自 AFM 特征约束还是多训 5 个 epoch？

---

## Phase 1 — 收敛验证（先跑）

**问题:** 当前不知道 PF+MobV3 训练几 epoch 收敛。必须先确定收敛点，后训练才有意义——否则任何续训都会涨，分不清是 AFM 还是多训。

**设计:** baseline（无 AFM）× seed42 × PF+MobV3 × 20 epoch。每 epoch 记录 val AP50/AP75。

**判定:** AP 连续 3 epoch 不涨 → 该 epoch 即为收敛点。后续后训练从此 checkpoint 出发。

**前置实现:** `round28_train_eval.py` 需加 per-epoch val eval（目前只在最后 eval 一次）。

---

## Phase 2 — 后训练质控（Phase 1 跑完后）

从 Phase 1 的收敛 checkpoint 出发：

| 组 | 操作 | 回答什么问题 |
|----|------|-------------|
| eval | 冻结全部，直接 eval | 收敛基线 |
| cont_full | **全模型续训** 5ep，无 AFM | 多训能不能涨？ |
| cont_afm_A | 冻结全部 + 嵌入新 AFM(弱门控) + 只训 AFM，5ep | AFM 有没有用？（A 路线） |
| cont_afm_C | 冻结全部 + 嵌入新 AFM(特征约束) + 只训 AFM，5ep | AFM 有没有用？（C 路线） |

每组 3 seed。

**判定逻辑：**
- 若 cont_afm > cont_full > eval → AFM 有效，且超过"多训几轮"的收益
- 若 cont_afm ≈ cont_full > eval → 后训练有效，但不是 AFM 的功劳，是续训的功劳
- 若 cont_full ≈ eval → 模型已收敛，干净基线

---

## 后续（2.28+，视 Phase 2 结果决定）

**若 AFM 后训练有效:** 扫 AFM 类型（pass-through/weak/mid/strong/mag-only/phase-only）、扫 epoch 数、跨模型跨数据集泛化。

**若 AFM 后训练无效:** 回到结构层面——当前的 "冻结+微调 AFM" 范式需要重新设计，考虑真正的高维 reward signal。
