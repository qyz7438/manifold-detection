# Plan 3.5: 频谱质量 Reward 跨数据集/跨 Backbone 验证

> **目标:** 验证 2.31（FFT spectral quality → bbox loss weight）在 PF+ResNet50 和 VOC+MobV3 上是否普遍有效。

---

## Phase 1 — 收敛基线（同 2.27 模式）

两个组合各跑收敛基线：

| 组 | 数据集 | backbone | epoch |
|----|--------|----------|-------|
| 3.5_pf_r50_conv | PF | ResNet50 | 20 |
| 3.5_voc_mob_conv | VOC 3-class | MobV3 | 20 |

保存 best checkpoint 供 Phase 2。

---

## Phase 2 — 频谱质量 reward（同 2.31 模式）

从 Phase 1 收敛 checkpoint 出发：

```
冻结 = backbone
训练 = RPN + box_head
loss = det_loss + alpha × quality × loss_box_reg
quality = FFT spectral heuristic (HF energy + entropy + phase coherence)
epoch = 15, alpha = [0.1, 0.5, 1.0], seed = [42, 123, 456]
```

两组各 9 组，共 18 组。

---

## 成功标准

AP75 对比冻结基线提升 > 5 点 → 方案跨 backbone/数据集有效。
