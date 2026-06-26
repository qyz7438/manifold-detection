# 检测组件实现逻辑与数据流报告

本报告介绍当前 `manifold-main` 分支下检测器各个组件的实现逻辑与数据流，仅做技术说明，不做实验分析或后续计划。

---

## 1. 总体架构与数据流

检测器基于 TorchVision 的 Faster R-CNN MobileNetV3-Large-FPN。在标准 Faster R-CNN 中，RPN 产生 proposals，ROI Align 把每个 proposal 映射为固定大小的 ROI feature，然后送入 `box_head` 提取特征，最后由 `box_predictor` 输出分类和回归结果。

当前实现允许在以下位置插入可选模块：

```
backbone + FPN
    ↓
RPN → proposals
    ↓
ROI Align → ROI feature (B, C, 7, 7)
    ↓
[FSBG / PBG]  ← 可选，作用于 ROI feature
    ↓
[AFM]         ← 可选，作用于 ROI feature
    ↓
box_head → flattened feature (B, in_features)
    ↓
[box_predictor 或 PAH] → cls_logits, bbox_deltas
    ↓
[TAM]         ← 可选，作用于 box_head 输出特征
    ↓
最终 cls_logits, bbox_deltas
```

模块插入顺序由 `spectral_detection_posttrain/core/models/build_detector.py` 控制，CLI 标志由 `scripts/round28_train_eval.py` 提供。

---

## 2. Frequency-Spatial Boundary Gate (FSBG) / Phase Boundary Gate (PBG)

### 2.1 文件位置

`spectral_detection_posttrain/methods/detection/pbg.py`

### 2.2 模块定义

```python
class FrequencySpatialBoundaryGate(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 0.0):
        super().__init__()
        hid = max(1, channels // 4)
        self.spatial_edge = nn.Sequential(
            nn.Conv2d(channels, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 1, 3, padding=1),
        )
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(channels, hid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 1, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
```

### 2.3 前向传播

输入：`x`，形状 `(B, C, H, W)`，在当前配置中通常为 `(B, 256, 7, 7)`。

**空间分支**：

```
x ──→ Conv2d(C, hid, 3, padding=1) ──→ ReLU ──→ Conv2d(hid, 1, 3, padding=1) ──→ spatial_map (B, 1, H, W)
```

`hid = max(1, C // 4)`，对于 `C=256` 时 `hid=64`。该分支是一个小型卷积边缘检测器，直接对 ROI feature 提取空间边界响应。

**相位分支**：

```
x ──→ torch.fft.rfft2(norm="ortho")
       ↓
    计算相位角 angle(freq)
       ↓
    phase_only = exp(1j * angle)   # 幅度被强制置为 1
       ↓
    torch.fft.irfft2(s=x.shape[-2:], norm="ortho")
       ↓
    ReLU
       ↓
    Conv2d(C, hid, 1) ──→ ReLU ──→ Conv2d(hid, 1, 1) ──→ phase_map (B, 1, H, W)
```

该分支先对输入做实数二维 FFT，保留相位信息并将所有频率分量的幅度置为 1，再通过反 FFT 重建出仅含相位结构的特征图。这一操作抑制了平滑区域、突出了结构边界。随后用两层 1×1 卷积将多通道特征编码为单通道相位边界图。

**融合分支**：

```
spatial_map (B, 1, H, W)
phase_map   (B, 1, H, W)
      ↓ concat 在通道维度
(B, 2, H, W)
      ↓
Conv2d(2, 1, 3, padding=1) ──→ Sigmoid ──→ gate (B, 1, H, W)
```

融合分支把空间边界图和相位边界图拼接后，通过一个 3×3 卷积和 Sigmoid 激活生成取值在 `(0, 1)` 之间的空间注意力门控。

**残差输出**：

```
output = x + alpha * gate * x
```

`alpha` 是一个可学习的标量参数，初始化值为 `alpha_init`（默认 0）。当 `alpha=0` 时，模块在初始时刻等价于恒等映射，可以安全插入预训练检测器而不会破坏已有特征分布。

### 2.4 别名

为兼容旧配置和 CLI 标志，模块同时暴露别名：

```python
PhaseBoundaryGate = FrequencySpatialBoundaryGate
```

因此 `--use-pbg` 标志实际调用的是 `FrequencySpatialBoundaryGate`。

---

## 3. Task-Aligned Manifold (TAM)

### 3.1 文件位置

`spectral_detection_posttrain/methods/detection/tam.py`

### 3.2 模块定义

```python
class TaskAlignedManifold(nn.Module):
    def __init__(self, in_features: int, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, in_features),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
```

### 3.3 前向传播

输入：`z`，形状 `(B, in_features)`，即 `box_head` 输出的展平特征。

```
z ──→ Linear(in_features, in_features//2) ──→ GELU ──→ Linear(in_features//2, latent_dim) ──→ latent (B, latent_dim)
                                                                                                      ↓
latent ──→ Linear(latent_dim, in_features//2) ──→ GELU ──→ Linear(in_features//2, in_features) ──→ residual (B, in_features)
                                                                                                      ↓
                                                                                              初始权重/偏置为 0
                                                                                                      ↓
output = z + scale * residual
```

编码器把高维特征压缩到 `latent_dim` 维的低维流形，解码器再映射回原维度。解码器最后一层的权重和偏置被初始化为 0，使得模块在初始时刻输出恒等残差（`residual ≈ 0`）。

`scale` 是一个可学习的标量，初始化为 0，用于控制残差流形的贡献程度。整个模块可视为在 `box_head` 特征空间上学习一个任务对齐的残差扰动。

---

## 4. Prototype-Aware Head (PAH)

### 4.1 文件位置

`spectral_detection_posttrain/methods/detection/pah.py`

### 4.2 模块定义

```python
class PrototypeAwareHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)
        self.temperature = temperature
```

### 4.3 前向传播

输入：`x`，形状 `(B, in_features)`，即 `box_head` 输出的展平特征。

**分类分支**：

```
x ──→ L2 归一化 ──→ x_norm (B, in_features)
prototypes ──→ L2 归一化 ──→ proto_norm (num_classes, in_features)

cls_logits = (x_norm @ proto_norm.T) / temperature   # 形状 (B, num_classes)
```

该分支用可学习的 prototype 向量替代传统分类层的权重。每个类别对应一个 prototype，分类分数通过查询特征与 prototype 之间的余弦相似度（经 temperature 缩放）得到。

**回归分支**：

```
x ──→ Linear(in_features, num_classes * 4) ──→ bbox_deltas (B, num_classes * 4)
```

回归分支与传统 Fast R-CNN 回归头一致，为每个类别预测 4 维边界框偏移量。

### 4.4 与标准预测头的替换关系

PAH 在 `build_detector.py` 中替换标准的 `FastRCNNPredictor`。当启用 `--use-pah` 时，分类和回归均由 PAH 完成；否则使用 TorchVision 默认的 `FastRCNNPredictor`。

---

## 5. AFM 基线模块

### 5.1 文件位置

`spectral_detection_posttrain/methods/afm/micro_afm.py`

### 5.2 模块概述

AFM（Amplitude-Frequency Modulation）模块继承自 MPLSeg 设计，作用于 ROI feature。它把 FFT 操作嵌入网络前向传播中，通过幅度门控和相位残差对频域特征进行调制，再用反 FFT 回到空间域。

### 5.3 数据流

输入：`x`，形状 `(B, C, H, W)`。

```
x ──→ torch.fft.rfft2(norm="ortho") ──→ freq (B, C, H, W//2+1) 复数
            ↓
      分离幅度 mag 和相位 phase
            ↓
      mag_gate  = conv_encoder(mag)      # 学习幅度门控
      phase_res = conv_encoder(phase)    # 学习相位残差
            ↓
      modulated_freq = (mag * mag_gate + mag * phase_res) * exp(1j * phase)
            ↓
      torch.fft.irfft2(s=x.shape[-2:], norm="ortho")
            ↓
      ReLU
            ↓
      output
```

具体实现支持多种残差模式（`current`、`delta`、`norm_delta`），可通过配置选择。默认情况下模块作为一个结构保持的特征扰动器工作。

---

## 6. 检测器构建与模块插入逻辑

### 6.1 文件位置

`spectral_detection_posttrain/core/models/build_detector.py`

### 6.2 构建流程

```python
def build_faster_rcnn_mobilenet(..., use_pbg=False, use_tam=False, use_pah=False,
                                afm_type='none', afm_residual_mode='current', ...):
    # 1. 创建标准 Faster R-CNN
    model = fasterrcnn_mobilenet_v3_large_fpn(...)

    # 2. 获取 ROI feature 通道数 C
    box_head = model.roi_heads.box_head
    C = box_head.fc7.in_features if hasattr(box_head, 'fc7') else box_head[0].in_features

    # 3. 可选：用 PAH 替换 box_predictor
    if use_pah:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        num_classes = model.roi_heads.box_predictor.cls_score.out_features
        model.roi_heads.box_predictor = PrototypeAwareHead(in_features, num_classes)

    # 4. 包装 box_head，按顺序插入 PBG/FSBG、AFM、TAM
    wrapper = BoxHeadFeatureWrapper(
        box_head=box_head,
        pbg=FrequencySpatialBoundaryGate(C) if use_pbg else None,
        afm=build_afm(C, afm_type, afm_residual_mode) if afm_type != 'none' else None,
        tam=TaskAlignedManifold(in_features) if use_tam else None,
    )
    model.roi_heads.box_head = wrapper
    return model
```

### 6.3 BoxHeadFeatureWrapper 数据流

```python
class BoxHeadFeatureWrapper(nn.Module):
    def __init__(self, box_head, pbg=None, afm=None, tam=None):
        self.box_head = box_head
        self.pbg = pbg
        self.afm = afm
        self.tam = tam

    def forward(self, x, proposals):
        # x: (B*R, C, 7, 7)
        if self.pbg is not None:
            x = self.pbg(x)
        if self.afm is not None:
            x = self.afm(x)

        # box_head 通常接收 (features, proposals)
        features = self.box_head(x)

        if self.tam is not None:
            features = self.tam(features)
        return features
```

注意：TAM 作用于 `box_head` 输出的展平特征，而不是 ROI feature 张量。

---

## 7. 训练脚本 CLI 映射

### 7.1 文件位置

`scripts/round28_train_eval.py`

### 7.2 相关参数

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `--use-pbg` | flag | False | 启用 FSBG/PBG 模块 |
| `--use-tam` | flag | False | 启用 TAM 模块 |
| `--use-pah` | flag | False | 启用 PAH 头 |
| `--afm-type` | str | `'none'` | AFM 类型，`none` 表示不启用 |
| `--afm-residual-mode` | str | `'current'` | AFM 残差模式 |
| `--pbg-alpha-init` | float | `0.0` | FSBG 的 alpha 初始化值 |
| `--tam-latent-dim` | int | `256` | TAM 的隐空间维度 |
| `--pah-temperature` | float | `0.1` | PAH 的 temperature |

### 7.3 调用示例

启用 FSBG 独立训练（无 AFM）：

```bash
python scripts/round28_train_eval.py \
  --run-name indep_fsbg_s42 \
  --afm-type none \
  --use-pbg \
  --trainable-mode box_head_only \
  --epochs 3 \
  --seed 42
```

启用 AFM + FSBG：

```bash
python scripts/round28_train_eval.py \
  --run-name afm_fsbg_s42 \
  --afm-type micro_afm \
  --use-pbg \
  --trainable-mode box_head_only \
  --epochs 3 \
  --seed 42
```

---

## 8. 测试覆盖

### 8.1 文件位置

`tests/methods/test_pbg.py`

### 8.2 测试内容

```python
def test_pbg_identity_at_init():
    # 验证 alpha=0 时 FSBG 输出等于输入
    m = FrequencySpatialBoundaryGate(256)
    x = torch.randn(2, 256, 7, 7)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)

def test_pbg_shape_and_gradient():
    # 验证非零 alpha 时梯度可正常回传
    m = FrequencySpatialBoundaryGate(128, alpha_init=0.1)
    x = torch.randn(2, 128, 7, 7, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in m.parameters())
```

---

## 9. 代码目录结构（与本报告相关部分）

```
spectral_detection_posttrain/
├── core/
│   └── models/
│       └── build_detector.py      # 检测器构建与模块插入
├── methods/
│   ├── afm/
│   │   └── micro_afm.py           # AFM 基线
│   └── detection/
│       ├── pbg.py                 # FSBG / PBG
│       ├── tam.py                 # TAM
│       └── pah.py                 # PAH
└── ...

scripts/
└── round28_train_eval.py          # 训练入口与 CLI

tests/
└── methods/
    └── test_pbg.py                # FSBG 基础测试
```

---

## 10. 数据维度速查

| 阶段 | 默认形状 | 说明 |
|------|---------|------|
| ROI feature | `(B*R, 256, 7, 7)` | R 为每张图的 ROI 数 |
| FSBG 输出 | `(B*R, 256, 7, 7)` | 与输入同形 |
| AFM 输出 | `(B*R, 256, 7, 7)` | 与输入同形 |
| box_head 输出 | `(B*R, in_features)` | 对于 MobileNetV3 通常为 1024 |
| TAM 输出 | `(B*R, in_features)` | 与输入同形 |
| PAH cls_logits | `(B*R, num_classes)` | num_classes 包含背景 |
| PAH bbox_deltas | `(B*R, num_classes * 4)` | 每类 4 维偏移 |
