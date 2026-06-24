# Plan C：语义分割方向的模块集成与验证

## 目标

将 Plan A 的可学习频域流形结构与 ChordEdit 式低能量传输集成到语义分割网络中，解决 MPLSeg-AFM 在分割任务上失效的问题，提升边界定位质量与 mIoU。

## 核心假设

MPLSeg-AFM 在检测上有效（AP75 +12.7%），但在分割上失效（mIoU 下降），可能原因：
1. 分割 backbone 特征通道数高（2048 vs 检测 256），单点门控信号被淹没；
2. 分割特征分辨率低，频谱信息稀疏；
3. magnitude/phase 门控是高能量局部扰动，破坏精细的分割结构。

ChordEdit 提示：**把门控替换为流形上的低能量传输**，可能保持结构的同时调整语义/定位。

## 模块设计

### 模块位置

```
spectral_detection_posttrain/methods/segmentation/
├── __init__.py
├── manifold_afm.py            # 基于流形的 AFM 模块
├── ot_segmentation_loss.py    # OT 分割损失
└── eval_segmentation.py       # 分割评估
```

### 1. ManifoldAFMBlock

替换现有 `MPLSegAFMBlock`，在流形上做 magnitude/phase 调制。

```python
class ManifoldAFMBlock(nn.Module):
    def __init__(self, channels, latent_dim=32, gate_strength=0.6):
        # channels: 输入特征通道数
        # latent_dim: 流形潜维度
        # gate_strength: 门控强度

    def forward(self, x):
        # x: (B, C, H, W)
        # 1. 对每个空间位置做 2D-DFT
        F = torch.fft.rfft2(x, norm='ortho')
        # 2. 把复数谱展平 (B, C, H, W//2+1) -> (B*H*(W//2+1), C)
        F_flat = F.permute(...).reshape(-1, C)
        # 3. 嵌入到流形
        z = self.manifold.encode(F_flat)
        # 4. 在流形上做 magnitude/phase 解耦调制
        rho = torch.abs(z)
        theta = torch.angle(z)
        rho_new = rho * self.mag_gate(rho)
        theta_new = theta + self.phase_residual(theta)
        z_new = rho_new * torch.exp(1j * theta_new)
        # 5. Chord 传输：低能量修正
        z_refined = self.transport(z_new, z)
        # 6. 解码回谱空间
        F_refined = self.manifold.decode(z_refined).reshape(...)
        # 7. iDFT
        x_out = torch.fft.irfft2(F_refined, s=x.shape[-2:])
        return x_out + x  # 残差连接
```

### 2. OTSegmentationLoss

在预测掩码与 GT 之间引入 Sinkhorn 距离，替代或补充交叉熵。

```python
class OTSegmentationLoss(nn.Module):
    def __init__(self, num_classes, eps=0.01):
        # 把预测和 GT 都视为空间上的类别分布

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) long
        # 1. softmax 预测
        # 2. GT 转成 one-hot 分布
        # 3. 计算空间位置上的 Sinkhorn 距离
        return loss
```

**关键点**：
- 不要对所有像素对计算完整 OT（计算量太大）；
- 用 mini-batch 像素采样或小 patch 级别的 OT；
- 或者只对预测不确定区域做 OT。

## 实现路线图

### Phase 1：复现分割基线（1-2 天）
- [ ] 复现 FCN-ResNet50 on Penn-Fudan 分割 baseline（mIoU=0.5208）；
- [ ] 确认 mid06 AFM 失效现象可复现。

### Phase 2：实现 ManifoldAFMBlock（2-3 天）
- [ ] 接入 Plan A 的流形模块；
- [ ] 实现空间维度的频谱处理；
- [ ] 保证 identity 初始化，训练稳定。

### Phase 3：训练与消融（3-4 天）
- [ ] 训练 ManifoldAFM + FCN-ResNet50；
- [ ] 消融：
  - latent_dim（16, 32, 64）；
  - 是否用 Chord 传输；
  - magnitude-only vs phase-only；
  - 插入位置（layer1/2/3/4）。

### Phase 4：OT 分割损失（2 天）
- [ ] 实现 `OTSegmentationLoss`；
- [ ] 与交叉熵联合训练；
- [ ] 测量边界 IoU（Boundary IoU）提升。

### Phase 5：扩展 VOC / 可视化（2 天）
- [ ] 在 Pascal VOC 上验证；
- [ ] 可视化流形坐标与分割错误区域的关系。

## 验证方式

| 验证项 | 通过标准 |
|--------|---------|
| 基线复现 | PF mIoU ≈ 0.52 |
| ManifoldAFM 有效 | mIoU 较 mid06 AFM 提升 ≥ 3% |
| Chord 传输贡献 | 去掉 Chord 传输后 mIoU 下降 |
| OT 损失贡献 | 边界 IoU 提升 ≥ 2% |
| 可视化 | 流形坐标与错误区域有对应关系 |

## 风险与依赖

| 风险 | 缓解 |
|------|------|
| 2048ch 维度太高 | 对通道分组后分别做流形嵌入 |
| 空间分辨率低 | 在 stride 8/16 特征上插入，不在 stride 32 |
| OT 损失计算慢 | 只采样不确定像素或小 patch |
| 训练不稳定 | identity 初始化 + 冻结 backbone 前几层 |

## 预计时间

**10–13 天**。

## 与 Plan A/B 的依赖

- 依赖 Plan A 的 `ComplexSpectralManifold` 和 `ChordTransport`；
- 可与 Plan B 共享频域分析工具。
