# Plan D：图像分类方向的模块集成与验证

## 目标

将频域流形结构与 ChordEdit 式低能量传输应用于图像分类，探索 magnitude/phase 解耦分类头、OT-based 原型分类器，以及频域数据增强对分类准确率和鲁棒性的影响。

## 核心思路

分类任务通常只需要语义信息，但：
- 细粒度分类需要纹理/边界信息（phase）；
- 低分辨率/退化输入需要频域鲁棒性；
- 对抗样本会扰动特定频率。

ChordEdit 提示：**把分类看作“样本特征到类别原型分布的低能量传输”**。

## 模块设计

### 模块位置

```
spectral_detection_posttrain/methods/classification/
├── __init__.py
├── spectral_classifier_head.py    # 频域 magnitude/phase 分类头
├── ot_prototype_classifier.py     # OT-based 原型分类器
├── spectral_mixup.py              # OT-based 频域 mixup
└── eval_classification.py         # 分类评估
```

### 1. SpectralClassifierHead

在 backbone 最后一层特征后，分别对 magnitude 和 phase 做全局池化与分类。

```python
class SpectralClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=256):
        # 1. 对输入特征做 2D-DFT
        # 2. magnitude 分支：全局统计池化 -> MLP -> logits
        # 3. phase 分支： circular statistics -> MLP -> logits
        # 4. 可学习融合权重

    def forward(self, x):
        F = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(F)
        phase = torch.angle(F)
        logits_mag = self.mag_head(self.mag_pool(mag))
        logits_phase = self.phase_head(self.phase_pool(phase))
        alpha = torch.sigmoid(self.fusion_weight)
        return alpha * logits_mag + (1 - alpha) * logits_phase
```

### 2. OTPrototypeClassifier

用每个类别的特征分布作为原型，分类时计算样本到各类原型的 Sinkhorn 距离。

```python
class OTPrototypeClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, n_prototypes=4):
        # 每个类别维护 n_prototypes 个分布（如高斯混合）

    def update_prototypes(self, features, labels):
        # 在线更新类别原型

    def forward(self, features):
        # 计算样本到每个原型的 OT 距离
        # return logits = -distance
```

### 3. SpectralMixup

用 OT 在频域做 mixup，生成更语义一致的增强样本。

```python
class SpectralMixup:
    def __init__(self, alpha=1.0):
        # alpha: Beta 分布参数

    def __call__(self, x1, x2, y1, y2):
        # 1. DFT
        # 2. 在 magnitude/phase 上分别做 OT 插值
        # 3. iDFT
        # 4. 标签按 lambda 混合
        return x_mix, y_mix
```

## 实现路线图

### Phase 1：频域分类头（2-3 天）
- [ ] 实现 `SpectralClassifierHead`；
- [ ] 在 CIFAR-100 ResNet-18 上训练；
- [ ] 对比 baseline：标准线性头 vs SpectralHead。

### Phase 2：OT 原型分类器（2-3 天）
- [ ] 实现 `OTPrototypeClassifier`；
- [ ] 在 CIFAR-100 上训练；
- [ ] 分析学到的原型结构。

### Phase 3：频域数据增强（2 天）
- [ ] 实现 `SpectralMixup`；
- [ ] 对比标准 mixup / cutmix；
- [ ] 测试 corruption robustness（CIFAR-100-C 子集）。

### Phase 4：细粒度分类（3 天）
- [ ] 在 CUB-200 或 Stanford Cars 上验证；
- [ ] 分析 magnitude-only / phase-only / combined 的贡献。

### Phase 5：对抗鲁棒性连接（2 天）
- [ ] 与 Plan B 的防御模块联合：先用 SpectralChordDefense 净化，再分类；
- [ ] 测量分类准确率恢复。

## 验证方式

| 验证项 | 通过标准 |
|--------|---------|
| CIFAR-100 baseline | ResNet-18 准确率 ≥ 75% |
| SpectralHead 提升 | 准确率提升 ≥ 0.5% 或鲁棒性提升 ≥ 2% |
| OT 原型有效 | 准确率 ≥ baseline，且可解释性更好 |
| SpectralMixup | 优于标准 mixup 或 cutmix |
| 细粒度验证 | CUB-200 上准确率 ≥ baseline |

## 风险与依赖

| 风险 | 缓解 |
|------|------|
| 分类任务对 phase 不敏感 | 先做 ablation，确定 phase 有效场景 |
| OT 原型更新不稳定 | 用 EMA 更新，小学习率 |
| 频域 mixup 产生伪影 | 约束插值范围，只在 magnitude 上做 |
| 计算开销 | 用 entropic OT，采样计算 |

## 预计时间

**10–13 天**。

## 与 Plan A/B/C 的依赖

- 依赖 Plan A 的流形与 OT 工具；
- 可与 Plan B 共享对抗防御评估；
- 可与 Plan C 共享频域特征分析脚本。
