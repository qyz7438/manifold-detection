# Plan 4.1-4.2: 分割基线 + AFM 收敛验证

> **For agentic workers:** 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务执行。复选框（`- [ ]`）用于跟踪。

**目标:** 在预训练 FCN-ResNet50 起点上，用 per-epoch val mIoU 监控收敛，得到可靠的 baseline vs mid06 对比。

**架构:** 2 配置 (baseline, mid06) × 3 seed × Penn-Fudan 分割 × FCN-ResNet50（COCO 预训练） × 15 epoch。每 epoch 记录 train_loss + val mIoU，收敛判定：连续 3 epoch mIoU 不涨即停止。

**技术栈:** Python, PyTorch, torchvision FCN-ResNet50, `round4x_seg_runner.py`

---

## 设计

| 组 | AFM | 起点 |
|----|-----|------|
| 4.1 baseline | none | COCO 预训练 FCN-ResNet50 |
| 4.2 mid06 | MPLSegAFMBlock(gate=0.6, in_ch=2048) | COCO 预训练 FCN-ResNet50 |

两者同 epoch、同 lr、同数据，干净对比。

## 需要修复的 bug

- [ ] `weights=None, weights_backbone=None` → `weights=FCN_ResNet50_Weights.DEFAULT`
- [ ] epoch 3 → 15，早期停止逻辑
- [ ] 每 epoch 后 eval val mIoU，写入 history
- [ ] 15 epoch 后 save checkpoint（供 4.5 后训练用）

## 成功标准

1. baseline 和 mid06 都在 val mIoU 上收敛（最后 3 epoch 不涨）
2. mid06 vs baseline 的 mIoU 差异有统计意义（3 seed 算 mean±std）
