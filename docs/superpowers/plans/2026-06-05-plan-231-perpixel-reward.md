# Plan 2.31: 逐像素频域 Reward

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按任务执行。

**目标:** 不改模型结构。用频域信息做成逐像素 quality map，当 RPN anchor 的训练权重——频域"热闹"的区域多看。

**架构:** 从 2.27 收敛 checkpoint 出发，附加 light-weight quality scorer（2 层 MLP）在 FPN 输出上，冻结 backbone + box_head，训练 RPN + scorer。5 epoch × 3 seed × 3 α。

**技术栈:** Python, PyTorch, 独立 runner 脚本

---

## 设计

```
FPN 特征图 (C, H, W)
  │
  └─→ quality_scorer: 每个空间位置取 (C,) → 1D FFT（通道维度）
          │
      score = σ(W × [高频能量, 相位一致性])   # 标量
          │
      quality_map (H, W)      ← dense reward
          │
      RPN anchor 覆盖区域的 quality 均值 → anchor_weight
          │
      RPN loss = anchor_weight × BCE(anchor_is_object)
```

```
起点 = 2.27 checkpoint_best.pth
模型 = 2.27 模型 + quality_scorer（附加，不改原有结构）
冻结 = backbone + box_head + box_predictor
训练 = RPN + quality_scorer
loss = RPN loss × anchor_weight + 检测 loss
epoch = 5, seed = [42, 123, 456]
α = [0.1, 0.5, 1.0]
共 9 组。
```

## quality_scorer 结构

```
nn.Sequential(
    nn.Linear(C, C//4),    # C = 256 (FPN 通道)
    nn.ReLU(),
    nn.Linear(C//4, 1),
    nn.Sigmoid(),
)
```

输入: FPN 每层特征的 per-pixel 1D FFT 统计量（高频能量占比、相位一致性指数）
输出: [0,1] quality score

---

## 成功标准

AP50/AP75 对比 2.28（冻结基线）。若 2.31 > 2.28 → 逐像素 reward 有效——教会模型看哪里。
