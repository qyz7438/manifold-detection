# Plan 2.34: BBox 探索 + 频谱 Reward

> **目标:** box_head 输出 Gaussian policy（μ, logσ²），每个 proposal 采样多个候选 delta，频谱 reward 打分 → REINFORCE 更新。

> **前提:** 2.33 成功。

---

## 设计

```
对每个正样本 proposal:
  box_head → (μ_x, μ_y, μ_w, μ_h, logσ²)    ← Gaussian policy
  采样 M 个 delta ~ N(μ, diag(σ²))
  → M 个候选框 b'₁...b'ₘ
  → 每个候选框 ROI → spectral_quality(b'_j)
  → REINFORCE: ∇J = mean((r_j - baseline) × ∇log P(delta_j|μ,σ))
```

**冻结:** backbone + RPN（RPN 由 2.33 训好）。只更新 box_head + box_predictor。

**超参:** M=10, σ_init=0.1, baseline=EMA(0.9)

**实现要点:**
- box_head 最后一层拆成 μ_head 和 logσ_head（双输出）
- `torch.distributions.Normal(μ, σ)` 采样
- 多个 delta 的 reward 取 mean 做 advantage

---

## 验证

从 2.33 最佳 checkpoint 出发，5 epoch × 3 seed = 3 组。
AP75 应优于 2.33——频谱 reward 同时改善 RPN 看哪里 + bbox 调到哪。
