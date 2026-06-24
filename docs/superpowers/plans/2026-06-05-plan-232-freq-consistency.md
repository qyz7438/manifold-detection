# Plan 2.32: ROI 频谱一致性 Reward

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按任务执行。

**目标:** 不改模型结构。对正样本 ROI 在 loss 中约束频谱一致性——同类物体有相似频谱结构。

**架构:** 从 2.27 收敛 checkpoint 出发，不改结构，loss 加 freq_consistency_loss。冻结 backbone，训练 RPN + box_head。5 epoch × 3 seed × 3 λ。

**技术栈:** Python, PyTorch, 独立 runner 脚本

---

## 设计

```
loss = 检测 loss + λ × freq_consistency_loss

freq_consistency:
  对每个正样本 ROI，计算 fft_mag
  同一 batch 同类 ROI 的平均 fft_mag → 作为 "prototype"
  loss = -cos_sim(当前 ROI fft_mag, prototype)
```

```
起点 = 2.27 checkpoint_best.pth
模型 = 2.27 模型（不改结构，无 AFM）
冻结 = backbone
训练 = RPN + box_head + box_predictor
loss = 检测 loss + λ × freq_consistency_loss
epoch = 5, seed = [42, 123, 456]
λ = [0.01, 0.05, 0.1]
共 9 组。
```

## 实现要点

- 需要在训练循环中获取 ROI 特征（box_head 输入前 hook）
- 同类 ROI 只在同一 batch 内聚合（无需跨 batch memory bank）
- 正样本判定: 与 GT 的 IoU > 0.5

---

## 成功标准

AP50/AP75 对比 2.28（冻结基线）。若 2.32 > 2.28 → 频谱一致性约束有效——不改结构也能用频域信息驱动模型。
