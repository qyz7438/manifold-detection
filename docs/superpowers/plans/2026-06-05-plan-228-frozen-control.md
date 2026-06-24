# Plan 2.28: 冻结续训对照

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按任务执行。

**目标:** 确认 2.27 收敛判断正确。如果收敛后续训还能涨 → 收敛判断有误，基线需重定。

**架构:** 从 2.27 收敛 checkpoint 出发，冻结全部（无 AFM），续训 5 epoch × 3 seed。

**技术栈:** Python, PyTorch, `round28_train_eval.py`

---

## 设计

```
起点 = 2.27 checkpoint_best.pth
模型 = 同 2.27（无 AFM）
冻结 = backbone + RPN + box_head + box_predictor（全部）
训练 = 无 trainable params（纯 eval only）
epoch = 5, seed = [42, 123, 456]
```

**注意:** 全部冻结意味着没有可训练参数。这是特意设计——如果模型确实收敛了，冻结续训不应该改变任何参数，AP 不变。如果 `trainable_mode=full` 但全冻结报错，改为只 eval 一次（epochs=0 with checkpoint）。

**断言:** 冻结后的 AP = 2.27 收敛点 AP。如果有变化 → 收敛判断不成立。

---

## 成功标准

3 seed 的 eval AP 与 2.27 收敛点一致（AP50 差异 < 0.01）。
