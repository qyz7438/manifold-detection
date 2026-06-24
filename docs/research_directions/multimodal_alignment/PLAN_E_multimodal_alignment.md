# Plan E：自然语言文本与图像文本对齐任务

## 目标

将 ChordEdit 的低能量最优传输思想应用于图像-文本对齐任务，探索用 OT 距离替代或补充对比学习（InfoNCE），实现更结构化、更低方差的跨模态对齐。

## 核心问题

现有 CLIP 风格的对齐是**样本对级别的内积最大化**：

$$\mathcal{L}_{\mathrm{InfoNCE}} = -\log \frac{\exp(f(x_i) \cdot g(t_i) / \tau)}{\sum_j \exp(f(x_i) \cdot g(t_j) / \tau)}$$

问题：
- 只关注配对样本的拉近/推开，忽略了**分布级别的结构**；
- 对比损失对噪声标签和 hard negatives 敏感；
- 文本到图像的编辑/生成中，drift 差分不稳定（这正是 ChordEdit 解决的问题）。

ChordEdit 提示：**把图像-文本对齐重新建模为两个分布（图像特征分布、文本特征分布）之间的低能量最优传输**。

## 模块设计

### 模块位置

```
spectral_detection_posttrain/methods/multimodal/
├── __init__.py
├── ot_alignment.py            # OT-based 图像-文本对齐损失
├── chord_text_guided.py       # ChordEdit 式文本引导图像生成/编辑
├── cross_modal_transport.py   # 跨模态低能量传输
└── eval_retrieval.py          # 检索评估
```

### 1. OTImageTextAlignment

用 Sinkhorn 距离度量 batch 内图像特征分布与文本特征分布的对齐程度。

```python
class OTImageTextAlignment(nn.Module):
    def __init__(self, feature_dim, eps=0.01, max_iter=50):
        # feature_dim: 图像/文本特征维度
        # eps: entropic regularization

    def forward(self, image_features, text_features):
        # image_features: (B, D)
        # text_features: (B, D)
        # 1. 归一化
        # 2. 计算 pairwise cost matrix C = 1 - cosine_similarity
        # 3. 计算 Sinkhorn 距离
        # 4. 对角线配对给予更低成本
        return loss
```

**与 InfoNCE 的关系**：
- InfoNCE 是 hard 0-1 对齐（配对=1，非配对=0）；
- OT 是 soft 结构化对齐，考虑所有样本之间的传输成本。

### 2. ChordTextGuidedEdit

把 ChordEdit 从“文本编辑图像”扩展到“文本引导图像生成/检索后精炼”。

```python
class ChordTextGuidedEdit(nn.Module):
    def __init__(self, image_encoder, text_encoder, transport):
        # image_encoder: 图像编码器
        # text_encoder: 文本编码器
        # transport: ChordTransport from Plan A

    def forward(self, x_source, text_source, text_target):
        # 1. 编码源/目标文本 -> 文本特征 v_src, v_tar
        # 2. 在图像特征空间构造源/目标 drift
        # 3. 用 Chord 传输得到低能量编辑方向
        # 4. 解码回图像或特征
        return x_edited
```

### 3. CrossModalTransport

学习一个从文本特征空间到图像特征空间的低能量传输映射。

```python
class CrossModalTransport(nn.Module):
    def __init__(self, text_dim, image_dim, manifold, transport):
        # 把文本特征映射为图像特征流形上的位移场

    def forward(self, text_feature, image_feature):
        # 计算从 image_feature 向 text_feature 引导方向的低能量移动
        return image_feature_refined
```

## 实现路线图

### Phase 1：数据与基线（2 天）
- [ ] 准备小规模图像-文本数据集：
  - Flickr30k 子集；
  - 或本地标注数据；
  - 或 COCO Captions 子集。
- [ ] 建立 CLIP-style 基线（可冻结 CLIP 或训练小 encoder）。

### Phase 2：OT 对齐损失（2-3 天）
- [ ] 实现 `OTImageTextAlignment`；
- [ ] 替换 InfoNCE，训练并对比；
- [ ] 尝试 InfoNCE + OT 联合损失。

### Phase 3：跨模态 Chord 传输（3-4 天）
- [ ] 实现 `CrossModalTransport`；
- [ ] 在文本-图像检索任务上验证；
- [ ] 可视化：文本引导下的图像特征移动路径。

### Phase 4：文本引导图像编辑（3-4 天）
- [ ] 在 diffusion/flow 模型上实现 ChordEdit 式的文本引导编辑；
- [ ] 重点验证：单步编辑的稳定性（与原始 ChordEdit 一致）；
- [ ] 与现有 one-step 编辑方法对比。

### Phase 5：遥感/多模态扩展（可选，2 天）
- [ ] 与 Plan F 结合：遥感图像-文本描述对齐。

## 验证方式

| 验证项 | 通过标准 |
|--------|---------|
| 基线复现 | CLIP-style 检索 R@1 达到合理水平 |
| OT 对齐有效 | R@1 ≥ InfoNCE baseline，或训练更稳定 |
| 跨模态传输 | 检索 R@1 提升 ≥ 1% |
| 文本引导编辑 | 单步编辑 PSNR/CLIP 优于朴素 drift 差分 |
| 可视化 | 文本引导的特征移动路径平滑、低能量 |

## 风险与依赖

| 风险 | 缓解 |
|------|------|
| 计算资源不足训练 CLIP | 冻结 CLIP，只训练对齐头；或用小模型 |
| OT 在 batch 小时效果差 | 用大 batch 或 memory bank |
| 文本-图像分布差异大 | 先在公共 embedding 空间（CLIP）做 OT |
| 单步编辑质量差 | 与 Plan B 的防御模块共享 refinement 思路 |

## 预计时间

**12–15 天**。

## 与 Plan A/B/C/D 的依赖

- 依赖 Plan A 的 `ChordTransport` 和 `SinkhornOT`；
- 可与 Plan B 共享 refinement 模块；
- 可与 Plan F 结合做遥感图像-文本对齐。
