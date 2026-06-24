# Plan 2.27: 后训练探索

> **核心问题:** 当前"后训练"没有 reward signal——只是冻结主体微调 AFM 模块。真正后训练需要一个 signal 驱动模型行为变化。

---

## 前置：实现 per-epoch val eval

**文件:** `scripts/round28_train_eval.py`

当前训练循环（L172-200）只记 train_loss，最后统一 eval。需改为每 epoch 后 eval val set，记录 val AP50/AP75 到 history。

**改动点:**

1. 训练循环中，每个 epoch 结束后跑 val eval（复用已有的 eval 代码块）
2. history row 新增 `val_ap50`、`val_ap75` 字段
3. 追踪 best_ap50，保存 best checkpoint
4. 去掉训练结束后的重复 eval（最后 epoch 的 eval 即为最终结果）

**改动范围:** L172-224（训练循环 + 最终 eval）

**实现后验收:** 跑一次 `--epochs 2`，确认 eval_metrics.json 中 history 数组包含 per-epoch val_ap50/val_ap75。

---

## V1 — 收敛基线

**问题:** 训练几 epoch 收敛？

**设计:** baseline（无 AFM）× seed42 × PF+MobV3 × 20 epoch。每 epoch 记录 val AP50/AP75。

**判定:** AP 连续 3 epoch 不涨 → 该点为收敛点。保存 checkpoint 供后续版本使用。

---

## V2 — 冻结续训对照

**问题:** 收敛后多训几轮会不会自然涨？

**设计:** 从 V1 收敛 checkpoint 出发，冻结全部，续训 5 epoch。

**断言:** V2 ≈ V1 → 确实收敛。V2 > V1 → 基线重定。

---

## V3 — AFM 特征精炼（当前最优，作为对照）

**设计:** V1 收敛模型 + 嵌入 AFM(gate=0.6) 在 ROI box_head 前。冻结全部，只训 AFM。标准检测 loss。5 epoch。

---

## V4 — AFM 特征约束（当前最优，作为对照）

**设计:** 同 V3，loss 加 0.05 × MSE(AFM输入, AFM输出)。

---

## V5 — 逐像素频域 Reward（核心探索）

**问题:** 不修改模型结构。频域信息做成 dense reward map，当 RPN anchor 的训练权重——频域"热闹"的区域要多看。

**设计:**

```
FPN 特征图 (C, H, W)
  │
  └─→ 每个空间位置取 (C,) 向量 → 1D FFT（通道维度）
          │
      quality_score = σ(W × [高频能量, 相位一致性])
          │
      ↓
      per-pixel quality_map (H, W)           ← dense reward
          │
      RPN 每个 anchor 聚合其覆盖区域的 reward
          │
      anchor_weight = 1 + α × mean(quality_map[anchor区域])
          │
      RPN loss = anchor_weight × BCE(anchor_is_object)
```

```
模型 = V1 收敛模型（不嵌入 AFM，不改结构）
冻结 = backbone + box_head + box_predictor
训练 = RPN + quality_scorer（轻量 2 层 MLP，附加在 FPN 输出后）
loss = RPN loss × anchor_weight
epoch = 5
```

**需扫超参:** α (0.1, 0.5, 1.0)

**组数:** 3 seed × 3 α = 9 组

---

## 总览

| 版本 | 类型 | 核心问题 |
|------|------|---------|
| V1 | 诊断 | 什么时候收敛？ |
| V2 | 对照 | 多训涨不涨？ |
| V3 | 对照 | AFM 特征精炼在干净基线下的真实表现 |
| V4 | 对照 | AFM 特征约束在干净基线下的真实表现 |
| V5 | **reward** | 逐像素频域 reward 能教会 RPN 看哪里吗？ |

---

### V6 — ROI 频域一致性 reward

**问题:** 不改结构。对正样本 ROI 在 loss 里约束频谱一致性——同类物体的 ROI 应该有相似的频谱结构。

**设计:**

```
loss = 检测 loss + λ × freq_consistency_loss

freq_consistency = -cos_sim(
    当前正样本 ROI 的 fft_mag,
    同一 batch 同类正样本 ROI 的平均 fft_mag
)
```

```
模型 = V1 收敛模型（不嵌入 AFM，不改结构）
冻结 = backbone（RPN + box_head 可训练）
loss = 标准检测 loss + λ × freq_consistency_loss
epoch = 5
```

**需扫超参:** λ (0.01, 0.05, 0.1)

**组数:** 3 seed × 3 λ = 9 组

---

## 总览

| 版本 | 类型 | 核心问题 |
|------|------|---------|
| V1 | 诊断 | 什么时候收敛？ |
| V2 | 对照 | 多训涨不涨？ |
| V3 | 对照 | AFM 特征精炼在干净基线下的真实表现 |
| V4 | 对照 | AFM 特征约束在干净基线下的真实表现 |
| V5 | **reward** | 逐像素频域 reward 能教会 RPN 看哪里吗？ |
| V6 | **reward** | ROI 频谱一致性约束能改善特征质量吗？ |

**两个 reward 方向:** V5 干预 RPN 阶段（看哪里），V6 干预 ROI 阶段（特征好不好）。

**执行顺序:** V1 → V2 → V3/V4/V5/V6 并行
