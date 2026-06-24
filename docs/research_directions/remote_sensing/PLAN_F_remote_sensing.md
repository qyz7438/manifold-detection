# Plan F：遥感领域专门方向

## 目标

将 Plan A 的可学习频域流形结构、ChordEdit 式低能量传输，以及 magnitude/phase 解耦思想，专门应用于高分辨率遥感图像理解任务，重点解决遥感目标检测中的小目标、旋转目标、复杂背景问题。

## 为什么遥感需要独立 plan

遥感图像与自然图像的关键差异：

1. **多尺度极端**：同一幅图像中可能同时包含像素级小船和占画面 1/4 的机场；
2. **方向任意**：车辆、飞机、船只朝向任意；
3. **频域结构强**：农田纹理、城市网格、海岸线等具有显著周期性；
4. **背景噪声复杂**：云层、阴影、大气散射、传感器噪声；
5. **对抗场景特殊**：物理对抗补丁可能以地面标志、车辆贴纸、建筑涂鸦形式出现。

这些差异意味着：直接把自然图像上的 AFM/流形模块搬到遥感上可能失效，需要专门设计。

## 模块设计

### 模块位置

```
spectral_detection_posttrain/methods/remote_sensing/
├── __init__.py
├── remote_sensing_afm.py        # 遥感专用 AFM
├── multiscale_spectral_head.py  # 多尺度频谱检测头
├── rotation_equivariant_fft.py  # 旋转等变频域模块
├── rs_manifold.py               # 遥感频域流形
└── eval_remote_sensing.py       # 遥感评估
```

### 1. RemoteSensingAFM

针对遥感特性改造 AFM：

```python
class RemoteSensingAFM(nn.Module):
    def __init__(self, channels, scales=[1, 2, 4]):
        # scales: 多尺度频域处理
        # 对不同尺度分别做 FFT，再融合

    def forward(self, x):
        # 1. 多尺度金字塔
        # 2. 对每个尺度做 FFT
        # 3. 分别 magnitude/phase 调制
        # 4. 跨尺度 attention 融合
        # 5. iDFT 后上采样/下采样回原尺寸
        return x + residual
```

**与自然图像 AFM 的区别**：
- 必须多尺度，因为遥感目标尺度差异大；
- magnitude gate 需要尺度自适应；
- phase residual 需要考虑旋转不变性。

### 2. RotationEquivariantFFT

遥感目标方向任意，需要旋转等变或旋转不变的频域表示。

```python
class RotationEquivariantFFT(nn.Module):
    def __init__(self, n_angles=8):
        # 把图像旋转 n_angles 次，分别做 FFT
        # 在频域对齐后取最大值或平均值

    def forward(self, x):
        # 1. rotate x by angles [0, 360/n, ...]
        # 2. FFT for each rotated version
        # 3. align phase in polar frequency coordinates
        # 4. pool across rotations
        return F_rot_inv
```

### 3. RemoteSensingManifold

遥感专用的频域流形，在 Plan A 基础上增加：
- 多尺度坐标；
- 方向坐标；
- 地物类别先验（如水域、植被、建筑等）。

```python
class RemoteSensingManifold(ComplexSpectralManifold):
    def __init__(self, in_dim, latent_dim, n_scales, n_orientations):
        # latent 坐标显式编码尺度、方向、语义
```

## 实现路线图

### Phase 1：遥感数据基线（2 天）
- [ ] 在 NWPU VHR-10 上建立 Faster R-CNN baseline；
- [ ] 在 VisDrone 上建立 baseline；
- [ ] 记录 AP50/AP75，分析错误模式（小目标、密集目标、旋转目标）。

### Phase 2：多尺度频谱检测头（3-4 天）
- [ ] 实现 `RemoteSensingAFM`；
- [ ] 插入 Faster R-CNN 的 FPN/ROI 特征；
- [ ] 消融：单尺度 vs 多尺度、不同插入位置。

### Phase 3：旋转等变频域模块（3-4 天）
- [ ] 实现 `RotationEquivariantFFT`；
- [ ] 在飞机、船只等旋转敏感类别上验证；
- [ ] 评估计算开销。

### Phase 4：遥感流形与 Chord 传输（3-4 天）
- [ ] 在 Plan A 基础上扩展 `RemoteSensingManifold`；
- [ ] 用流形上的低能量传输做特征精炼；
- [ ] 评估对小目标 AP 的提升。

### Phase 5：对抗防御在遥感场景（2-3 天）
- [ ] 与 Plan B 结合：在遥感图像上生成对抗补丁；
- [ ] 验证 `SpectralChordDefense` 的跨域迁移能力；
- [ ] 分析遥感频谱特性对防御的影响。

### Phase 6：图像-文本对齐（可选，3 天）
- [ ] 与 Plan E 结合：遥感图像-地理文本描述对齐；
- [ ] 做图像-文本检索验证。

## 验证方式

| 验证项 | 通过标准 |
|--------|---------|
| NWPU baseline | AP50 ≥ 0.70（取决于具体模型） |
| RemoteSensingAFM | AP50 提升 ≥ 2% |
| 旋转等变模块 | 飞机/船只类别 AP 提升 ≥ 3% |
| 小目标检测 | 小尺度目标 AP 提升 ≥ 3% |
| 对抗防御 | 遥感场景下 AP 恢复 ≥ 60% |

## 风险与依赖

| 风险 | 缓解 |
|------|------|
| 遥感数据标注不一致 | 先用 NWPU VHR-10 和 VisDrone 标准化评估 |
| 多尺度模块计算量大 | 只在 FPN 特定层级插入，不全局使用 |
| 旋转等变实现复杂 | 先做 4/8 个离散角度，再做连续近似 |
| 与小目标检测冲突 | 对小目标尺度单独保留高频信息 |

## 预计时间

**13–18 天**。

## 与 Plan A/B/C/D/E 的依赖

- 依赖 Plan A 的流形基础设施；
- 与 Plan B 共享对抗防御模块；
- 与 Plan C 共享分割模块（变化检测可视为分割）；
- 与 Plan D 共享分类模块（遥感场景分类）；
- 与 Plan E 共享多模态对齐模块。
