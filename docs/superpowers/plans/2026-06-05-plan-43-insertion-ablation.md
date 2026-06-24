# Plan 4.3: AFM 插入位置消融

> **For agentic workers:** 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务执行。

**目标:** 确定 AFM 在 FCN 中的最优插入位置（检测经验：256ch 最有效，分割应在接近此通道数的层做实验）。

**架构:** 4 位置 × 3 seed × PF 分割 × FCN-ResNet50（预训练） × 15 epoch。

**技术栈:** Python, PyTorch, torchvision FCN, `round4x_seg_runner.py`（扩展 `FCNSegAFM` 支持多位置插入）

---

## 插入位置

FCN-ResNet50 backbone 输出四层特征：

| 位置 | 通道数 | 空间尺度 | 语义 |
|------|--------|----------|------|
| layer1 | 256 | H/4 | 浅层纹理 |
| layer2 | 512 | H/8 | 中层形状 |
| layer3 | 1024 | H/16 | 高层语义 |
| layer4 ("out") | 2048 | H/32 | 最高语义（当前） |

在 `FCNSegAFM` 中加 `insert_layer` 参数，将 AFM 插在指定 backbone 层后、classifier 前。对于非 "out" 层，需额外处理跳连结构。

## 实验矩阵

| 组名 | insert_layer | in_ch |
|------|-------------|-------|
| pos_layer1 | layer1 | 256 |
| pos_layer2 | layer2 | 512 |
| pos_layer3 | layer3 | 1024 |
| pos_out | out | 2048（对照） |

## 成功标准

1. 找到最优插入位置（最高 mIoU）
2. 验证"低通道层更有效"假说
