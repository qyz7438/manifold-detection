# Plan 2.30: AFM 特征约束

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按任务执行。

**目标:** 同 2.29，loss 加 MSE(AFM输入, AFM输出) 防止 FFT 把特征搞飞。

**架构:** 与 2.29 相同，loss 加 0.05 × MSE。

**技术栈:** Python, PyTorch, `round28_train_eval.py`（需修改——当前不支持 MSE 约束。如果改动量大，用独立 runner 脚本）

---

## 设计

```
起点 = 2.27 checkpoint_best.pth
模型 = 2.27 模型 + 嵌入新 AFM(gate=0.6)
冻结 = backbone + RPN + box_head + box_predictor
训练 = AFM 全部参数
loss = 检测 loss + 0.05 × MSE(AFM输出, AFM输入)
epoch = 5, seed = [42, 123, 456]
```

---

## 实现

复用 2.26 的 `train_post_C` 模式——hook AFM 的输入输出，在 loss 里加 feat_loss。独立 runner 脚本或扩展 `round28_train_eval.py`。

---

## 成功标准

AP75 对比 2.29（无约束 AFM）。若 2.30 > 2.29 → 特征约束有效。
