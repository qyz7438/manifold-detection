# Plan 2.33: RPN 采样 + REINFORCE

> **目标:** 将 RPN 从确定性 top-K 改为按 objectness 概率分布随机采样，频谱 quality reward 做 REINFORCE 更新。验证 RLVR 在检测上是否通则。

---

## 设计

```
每个 FPN level:
  RPN → objectness logits (A, H, W)
    → softmax → 概率分布
    → categorical 采样 K 个位置
    → 每个位置生成 proposal → ROI Align → ROI features
    → spectral_quality(reward)
    → REINFORCE: ∇J = (reward - baseline) × ∇log P(采样位置)
```

**冻结:** backbone + box_head。只更新 RPN head。

**超参:** K=20（每层采样数），reward 基线=EMA(baseline=0.9)，alpha=[0.1, 0.5]

**实现要点:**
- `torch.multinomial(probs, K)` 做无放回采样
- bbox delta 解码：`dx = wx*aw + ax`（标准 Faster R-CNN 解码）
- `roi_align` 提取采样位置的 ROI 特征
- `spectral_quality()` 用已实现的固定 heuristic
- baseline subtraction 降方差

---

## 验证

从 2.27 收敛 checkpoint 出发，5 epoch RL × 2 alpha × 1 seed = 2 组。
对比冻结基线 AP50=0.887, AP75=0.623。
