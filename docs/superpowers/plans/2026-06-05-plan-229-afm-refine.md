# Plan 2.29: AFM 特征精炼

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按任务执行。

**目标:** 在 2.27 确认收敛的前提下，嵌入 AFM 做特征精炼——复现当前"后训练"方案的真实表现。

**架构:** 从 2.27 收敛 checkpoint 出发，嵌入 MPLSegAFMBlock(gate=0.6) 在 ROI box_head 前，冻结全部其他参数，只训 AFM，5 epoch × 3 seed。

**技术栈:** Python, PyTorch, `round28_train_eval.py`

---

## 设计

```
起点 = 2.27 checkpoint_best.pth
模型 = 2.27 模型 + 嵌入新 AFM(gate=0.6) 在 box_head 前
冻结 = backbone + RPN + box_head + box_predictor
训练 = AFM 全部参数
loss = 标准检测 loss（分类 + bbox 回归）
epoch = 5, seed = [42, 123, 456]
```

**注意:** AFM 是新建的（随机初始化），不是加载已训练好的 AFM。这样才叫"后训练"——对收敛模型加新模块，只训新模块。

---

## 成功标准

AP50/AP75 对比 2.28 的冻结基线。若 2.29 > 2.28 → AFM 有效。若 2.29 ≈ 2.28 → AFM 无用。
