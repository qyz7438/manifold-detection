# Plan 4.4: 分割专用 Gate Strength Sweep

> **For agentic workers:** 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务执行。

**目标:** 在 4.3 的最优插入位置上，扫描 gate_strength 找到分割的最优门控强度。

**架构:** 4 gate_strength × 3 seed × PF 分割 × FCN-ResNet50（预训练） × 15 epoch。使用 4.3 确定的最优 `insert_layer` 和 `in_ch`。

**技术栈:** Python, PyTorch, `round4x_seg_runner.py`

---

## 门控值

| gate_strength | 含义 |
|---------------|------|
| 0.1 | 极弱（接近 pass-through） |
| 0.3 | 弱 |
| 0.6 | 中（检测最优） |
| 1.0 | 全量 |

## 成功标准

找到分割任务的最优 gate_strength。
