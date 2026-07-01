# Image-level vs ROI-level Spectral Manifold 设计报告

> 目标：把“傅里叶频谱分析”放在原始图像层，把“数据特征维度的流形分析”放在倒数第二层（RoI feature 后、分类头前），并与已有的 FPN-level 流形形成对照。本阶段暂时剥离 RLVR verifier，仅作为可微分的网络内模块进行标准检测后训练。

---

## 一、整体思路

### 1.1 两个不同位置的流形

| 层级 | 名称 | 位置 | 处理对象 | 目的 |
|---|---|---|---|---|
| **Image-level** | `ImageSpectralManifold` | backbone 之前 | 原始图像 `(B,3,H,W)` | 在数据最源头对全局/局部频谱进行可学习重构，让 backbone 接收到经过频域筛选的输入 |
| **ROI-level** | `ROISpectralManifold` | RoI Align 之后、box_head 之前 | RoI 特征 `(N,C,7,7)` | 在位置信息已充分展开的倒数第二层，对每个候选框的频谱特征做低维流形压缩与增强 |
| **FPN-level** | `FPNSpectralManifold` | backbone 与 RPN/ROI 之间 | FPN 多尺度特征 | 已有对照模块，用于对比“在 neck 处做频谱流形”是否不如 ROI-level |

### 1.2 为什么选择这两个点？

- **原始图像层**：图像是数据最原始、维度最低（仅 3 通道）的地方。FFT 在这里能直接捕获边缘、纹理、周期性的全局频率分布。如果频谱流形在这里有效，说明它可以作为**预处理式的可学习增强**。
- **RoI 层**：此时每个 RoI 已经对齐到固定大小（如 7×7），背景被显著裁剪，位置/尺度信息已充分展开。频谱分析集中在“候选框内部”，对分类和定位更直接。这是用户明确指出的“倒数第二层”位置。
- **与 RLVR 解耦**：先验证模块本身作为监督/后训练的特征增强是否有效，再考虑把它作为 verifier。避免 RLVR 的梯度方差问题干扰模块有效性的判断。

### 1.3 共同设计原则

- **复数域 FFT + ComplexSpectralManifold**：保留幅度和相位信息，利用已有的 identity-initialized 复数 MLP。
- **残差连接 + 零初始化 scale**：保证训练起步时模块近似恒等映射，不会破坏预训练 backbone/head。
- **通道维度作为特征向量**：对每个空间-频率位置，把通道维（image 为 3，ROI 为 256）作为输入特征，过同一个共享的复数自编码器。

---

## 二、数据流动报告

### 2.1 完整前向流程

```text
输入图像 batch
    │
    ▼
[GeneralizedRCNNTransform]  # resize / normalize
    │
    ▼
+-------------------------------------------+
│  ImageSpectralManifold (可选)             │
│  输入: (B, 3, H, W) 实数                  │
│  输出: (B, 3, H, W) 实数                  │
+-------------------------------------------+
    │
    ▼
Backbone + FPN
    │
    ▼
+-------------------------------------------+
│  FPNSpectralManifold (可选，已有)         │
│  输入: dict{level: (B, C, H_l, W_l)}      │
│  输出: dict{level: (B, C, H_l, W_l)}      │
+-------------------------------------------+
    │
    ▼
RPN → proposals
    │
    ▼
RoI Align
    │
    ▼
RoI feature: (N, C, 7, 7)
    │
    ▼
+-------------------------------------------+
│  RefinedBoxHead wrapper                   │
│  内部顺序:                                │
│    PBG (可选)                             │
│    LSG (可选)                             │
│    AFM (可选)                             │
│    ROISpectralManifold (可选)  ◄── 新增   │
│    downsample → box_head (flatten+MLP)    │
│    TAM (可选)                             │
+-------------------------------------------+
    │
    ▼
box_predictor → cls_logits + bbox_regression
```

### 2.2 Image-level 分支细节

```text
x: (B, 3, H, W)
  │
  ▼ rfft2(norm="ortho")
F: (B, 3, H, W//2+1)  complex
  │
  ▼ permute + reshape
F_flat: (B*H*(W//2+1), 3)  complex
  │
  ▼ ComplexSpectralManifold(3 → latent → 3)
F_rec_flat: (B*H*(W//2+1), 3)  complex
  │
  ▼ reshape + permute
F_rec: (B, 3, H, W//2+1)  complex
  │
  ▼ irfft2(s=(H,W), norm="ortho")
x_freq: (B, 3, H, W)  real
  │
  ▼ x + scale * x_freq
out: (B, 3, H, W)
```

### 2.3 ROI-level 分支细节

```text
x: (N, C, 7, 7)          # N = 当前 batch 所有 RoI 总数
  │
  ▼ rfft2(norm="ortho")
F: (N, C, 7, 4)  complex # 7x7 → rfft2 → 7x4
  │
  ▼ permute + reshape
F_flat: (N*7*4, C)  complex
  │
  ▼ ComplexSpectralManifold(C → latent → C)
F_rec_flat: (N*7*4, C)  complex
  │
  ▼ reshape + permute
F_rec: (N, C, 7, 4)  complex
  │
  ▼ irfft2(s=(7,7), norm="ortho")
x_freq: (N, C, 7, 7)  real
  │
  ▼ x + scale * x_freq
out: (N, C, 7, 7)
```

> 若 `N == 0`（推理时无 proposal），直接返回 `x`，避免空张量 FFT 报错。

---

## 三、伪代码报告

### 3.1 ImageSpectralManifold

```python
class ImageSpectralManifold(nn.Module):
    def __init__(self, channels=3, latent_dim=None):
        self.channels = channels
        self.latent_dim = latent_dim or channels
        self.manifold = ComplexSpectralManifold(
            in_dim=channels,
            latent_dim=self.latent_dim,
            hidden_dim=channels,
        )
        self.scale = nn.Parameter(torch.zeros(1))   # 残差系数，初始为 0

    def forward(self, x: (B, C, H, W)):
        assert C == self.channels
        F = torch.fft.rfft2(x, norm="ortho")        # (B, C, H, W//2+1)
        b, c, h, w_r = F.shape
        F_flat = F.permute(0, 2, 3, 1).reshape(b*h*w_r, c)
        F_rec = self.manifold(F_flat)                # (B*H*W_r, C)
        F_rec = F_rec.reshape(b, h, w_r, c).permute(0, 3, 1, 2)
        x_freq = torch.fft.irfft2(F_rec, s=(h, w), norm="ortho")
        return x + self.scale * x_freq
```

### 3.2 ROISpectralManifold

```python
class ROISpectralManifold(nn.Module):
    def __init__(self, channels=256, latent_dim=None):
        self.channels = channels
        self.latent_dim = latent_dim or channels
        self.manifold = ComplexSpectralManifold(
            in_dim=channels,
            latent_dim=self.latent_dim,
            hidden_dim=channels,
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: (N, C, H, W)):
        if x.shape[0] == 0:
            return x
        assert C == self.channels
        F = torch.fft.rfft2(x, norm="ortho")        # (N, C, H, W//2+1)
        n, c, h, w_r = F.shape
        F_flat = F.permute(0, 2, 3, 1).reshape(n*h*w_r, c)
        F_rec = self.manifold(F_flat)
        F_rec = F_rec.reshape(n, h, w_r, c).permute(0, 3, 1, 2)
        x_freq = torch.fft.irfft2(F_rec, s=(h, w), norm="ortho")
        return x + self.scale * x_freq
```

### 3.3 build_detector 中的接入逻辑

```python
# 1) Image-level：在 backbone.forward 外包一层
if model_cfg["image_spectral_manifold"]:
    img_sm = ImageSpectralManifold(
        channels=model_cfg["img_sm_channels"],
        latent_dim=model_cfg["img_sm_latent_dim"],
    )
    old_backbone_forward = model.backbone.forward
    def new_backbone_forward(x):
        x = img_sm(x)                       # 先过图像级频谱流形
        features = old_backbone_forward(x)  # 再走原有 backbone
        return features
    model.backbone.forward = new_backbone_forward

# 2) ROI-level：加入 RefinedBoxHead 的流水线
if model_cfg["roi_spectral_manifold"]:
    roi_sm = ROISpectralManifold(
        channels=model_cfg["roi_sm_channels"],
        latent_dim=model_cfg["roi_sm_latent_dim"],
    )

class RefinedBoxHead(nn.Module):
    def __init__(self):
        self.pbg = pbg
        self.lsg = lsg
        self.spatial_afm = spatial_afm
        self.roi_sm = roi_sm
        self.head = original_box_head

    def forward(self, x, proposals=None):
        if self.pbg is not None:        x = self.pbg(x)
        if self.lsg is not None:        x = self.lsg(x)
        if self.spatial_afm is not None: x = self.spatial_afm(x)
        if self.roi_sm is not None:     x = self.roi_sm(x)   # ROI 级流形
        x = self.downsample(x)
        z = self.head(x)
        return z
```

### 3.4 训练启动命令

```bash
# baseline
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/ps/lzz/RLimage \
  python scripts/round28_train_eval.py \
  --run-name nwpu_baseline_10ep_s42 \
  --dataset nwpu \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 10 --seed 42 --trainable-mode all_except_backbone

# image-level spectral manifold
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/ps/lzz/RLimage \
  python scripts/round28_train_eval.py \
  --run-name nwpu_image_sm_10ep_s42 \
  --dataset nwpu \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 10 --seed 42 --trainable-mode all_except_backbone \
  --image-spectral-manifold --img-sm-channels 3 --img-sm-latent-dim 3

# ROI-level spectral manifold
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/ps/lzz/RLimage \
  python scripts/round28_train_eval.py \
  --run-name nwpu_roi_sm_10ep_s42 \
  --dataset nwpu \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 10 --seed 42 --trainable-mode all_except_backbone \
  --roi-spectral-manifold --roi-sm-channels 256 --roi-sm-latent-dim 256
```

---

## 四、设计取舍与后续可扩展点

### 4.1 当前取舍

- **Image-level 使用整图 FFT**：实现简单、梯度完整，但会混入大量背景频率，小目标信号容易被稀释。后续可考虑改成 **patch-wise FFT**（类似 FANet MSFFEM）。
- **Latent dim 先取 identity**：`latent_dim == in_dim`，保证模块初始为恒等映射，先验证“可学习频谱重构”本身是否有用；后续可尝试压缩 latent 维度的真正流形降维。
- **训练模式用 `all_except_backbone`**：backbone 冻结，保证 image/roi 模块和 box head/predictor 一起训练，控制变量公平。

### 4.2 可扩展点（为发一区预留）

1. **Patch-wise image-level**：把原图分块 FFT，增强局部性。
2. **Phase / Magnitude 分离约束**：显式让流形分别编码幅度和相位，提升可解释性。
3. **跨尺度频谱一致性**：让 image-level 和 ROI-level 学到的频谱表征互相约束。
4. **重新接入 RLVR**：如果 ROI-level 有效，可用其重构误差或 latent 置信度作为 in-network verifier reward，再论证 RLVR 频谱 verifier。

---

## 五、文件改动清单

| 文件 | 说明 |
|---|---|
| `spectral_detection_posttrain/methods/manifold/image_spectral_manifold.py` | 新增图像级频谱流形 |
| `spectral_detection_posttrain/methods/manifold/roi_spectral_manifold.py` | 新增 ROI 级频谱流形 |
| `spectral_detection_posttrain/methods/manifold/__init__.py` | 导出两个新模块 |
| `spectral_detection_posttrain/core/models/build_detector.py` | 接入 image/roi 模块到 backbone 和 box head |
| `scripts/round28_train_eval.py` | 新增 CLI 参数和 `image_sm_only` / `roi_sm_only` 训练模式 |
| `docs/reports/image_roi_spectral_manifold_design.md` | 本报告 |
