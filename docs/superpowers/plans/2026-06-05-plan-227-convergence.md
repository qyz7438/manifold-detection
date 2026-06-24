# Plan 2.27: 收敛基线

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按任务执行。

**目标:** 确定 PF+MobV3 baseline 训练几 epoch 收敛。不知道收敛点，后训练结论不可靠。

**架构:** baseline（无 AFM）× seed42 × PF+MobV3 × 20 epoch。每 epoch 后 eval val AP50/AP75，记录 history。

**技术栈:** Python, PyTorch, `round28_train_eval.py`

---

## 设计

```
模型 = Faster R-CNN MobileNetV3（COCO 预训练）
AFM = none
训练模式 = full
epoch = 20
seed = 42
```

**判定:** AP 连续 3 epoch 不涨 → 该点为收敛点。保存 `checkpoint_best.pth` 供后续版本使用。

---

## 前置依赖

`round28_train_eval.py` 已改为 per-epoch eval（每 epoch 后自动 eval val set，记录 val_ap50/val_ap75 到 history）。

---

## 执行

```bash
python scripts/round28_train_eval.py \
  --run-name round227_v1_convergence \
  --afm-type none --trainable-mode full \
  --epochs 20 --seed 42
```

---

## 成功标准

- eval_metrics.json 中 history 包含 20 个 epoch 的 val AP
- 明确收敛 epoch（AP 曲线 plateau）
- checkpoint 保存在 `runs/round227_v1_convergence/checkpoint_best.pth`
