# FPN-level / ROI-level 低秩频域残差适配器：实现思路、预期目标与伪代码

> 状态：代码已实现并通过本地/远程 smoke test，**暂不启动训练**，待 GPU2 空闲后按本报告执行对照实验。

---

## 1. 总体定位

本方案的核心模块不再称为“频谱流形重构器”，而是：

> **FPN-level 可学习低秩频域残差适配器**  
> `ROI-level` 版本仅作为 7×7 倒数第二层对照。

模块插入位置：

```text
Backbone + FPN
   ↓
FPNSpectralManifold  (低秩频域残差适配器)
   ↓
RPN + RoI Align + Box Head
```

因此它可以同时影响：

1. RPN objectness
2. proposal localization
3. ROI classification
4. box regression

这比仅作用于 `box_head` 前的 ROI-level 版本更符合检测定位瓶颈。

---

## 2. 关键设计修正（相对上一版）

| 问题 | 修正 |
|---|---|
| 残差写成 `x + scale * irfft2(F_rec)`，退化为输入缩放 | 改为 **ΔF = adapter(F) − F**，输出 **x + α·irfft2(ΔF)** |
| `latent_dim = 256` 无压缩 | 默认 **`latent_dim = C // 4`**（256→64），hidden_dim = C // 2 |
| 无频率位置信息 | 加入 **radius / angle 频率坐标门控**，并可附加 **level 坐标** |
| identity 初始化导致 delta=0 梯度死锁 | `ComplexMLP` 采用 **near-identity + 小噪声** 初始化 |
| alpha 按索引 `alpha[i]`，跨 backbone 可能错位 | 改为 **按 FPN key 绑定**（`alpha['0']`, `alpha['1']`, `alpha['pool']`），并在 `build_detector` 中通过 dummy forward 自动推断真实 keys |
| DC 分量可能被任意改动，导致全局 mean shift | 增加 `suppress_dc` 选项，强制 DC bin 残差为 0 |
| 每 batch 调用 `.item()` 造成 GPU 同步 | 改为 **epoch 级 running average**：forward 只累积 detached tensor，epoch 末一次性 `.item()` |
| 缺少非频谱对照组 | 新增 `FPNRealAdapter`：与 FPN-SM 同形状，但用 1×1 bottleneck conv 而非 FFT，用于区分“低秩 adapter”与“频域操作”的贡献 |

---

## 3. 概念澄清：不是 autoencoder reconstruction

由于 bottleneck 维度 64 < 256，`adapter(F)` 不可能严格等于 `F`。因此：

- 模块**不是**在最小化频谱重构误差；
- `adapter(F) − F` 应被理解为一个**低秩、频率坐标条件化的频域修正量**；
- 训练信号只来自下游 detection loss，没有 reconstruction loss；
- 因此 `freq_residual_ratio` 不要求随训练下降，它只是观察残差强度的诊断量。

---

## 4. 数据流

### 4.1 FPN-level

```text
输入图像
    ↓
[backbone] ──► FPN 特征 {key: (B, C, H, W)}
    ↓
[FPNSpectralManifold]
    对每层 key:
        F      = rfft2(x)                         # (B, C, H, W_r)
        F_flat = reshape(F, (B*H*W_r, C))
        F_rec  = adapter(F_flat)                   # low-rank complex adapter
        ΔF     = F_rec − F_flat                    # 频域修正量
        ΔF     = ΔF * gate(radius, angle, level)   # 频率/层坐标门控
        ΔF[DC] = 0  (可选)
        Δx     = irfft2(reshape(ΔF), s=(H,W))
        x'     = x + alpha[key] * Δx
    ↓
{RPN, RoI Align, Box Head}
```

### 4.2 ROI-level

```text
…
[RoI Align] ──► x (N, C, 7, 7)
    ↓
[ROISpectralManifold]
    与 FPN-level 相同，但单 α、无 level 坐标、分辨率 7×7
    ↓
[box_head]
```

### 4.3 非频谱对照 `FPNRealAdapter`

```text
x (B, C, H, W)
    ↓ 1×1 conv  C → 64
    ↓ ReLU
    ↓ 1×1 conv  64 → C
    ↓
x' = x + alpha[key] * adapter(x)
```

---

## 5. 关键模块伪代码

### 5.1 `FPNSpectralManifold`

```python
class FPNSpectralManifold(nn.Module):
    def __init__(self, channels=256, level_keys=None,
                 latent_dim=64, hidden_dim=128,
                 use_freq_coords=True, use_level_coords=True,
                 gate_activation="sigmoid", suppress_dc=False,
                 init_alpha=1e-3):
        self.level_keys = level_keys or ["0", "1", "2", "3"]
        self.adapter = ComplexSpectralManifold(
            in_dim=channels, latent_dim=latent_dim,
            hidden_dim=hidden_dim, near_identity=True)
        self.alpha = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(init_alpha))
            for key in self.level_keys
        })
        if use_freq_coords:
            gate_in = 2 + (1 if use_level_coords else 0)
            self.gate = nn.Sequential(
                nn.Linear(gate_in, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, channels), activation)

    def forward(self, features):
        assert set(self.level_keys) <= set(features.keys())
        for level_idx, key in enumerate(self.level_keys):
            x = features[key]
            F = torch.fft.rfft2(x.float(), norm="ortho")
            F_flat = F.permute(0,2,3,1).reshape(-1, C)
            dF = self.adapter(F_flat) - F_flat
            if self.use_freq_coords:
                gate = self._make_gate(h, w, b, level_idx)
                dF = dF * gate
            if self.suppress_dc:
                dF[::h_f*w_r] = 0.0
            dF = dF.reshape(b, h_f, w_r, c).permute(0,3,1,2)
            dx = torch.fft.irfft2(dF, s=(h,w), norm="ortho")
            out[key] = x + self.alpha[key] * dx
            # accumulate detached tensor stats, no .item() here
        return out

    def epoch_stats(self):
        return {k: float(torch.stack(v).mean().item())
                for k, v in self._stats_buffer.items()}
```

### 5.2 `ROISpectralManifold`

结构与 FPN 版一致，但：

- 输入 `(N, C, H, W)`，通常 `H=W=7`；
- 单标量 `alpha`；
- 无 level 坐标；
- 空 RoI 保护。

### 5.3 `FPNRealAdapter`

```python
class FPNRealAdapter(nn.Module):
    def __init__(self, channels=256, level_keys=None,
                 latent_dim=64, init_alpha=1e-3):
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, latent_dim, 1), nn.ReLU(),
            nn.Conv2d(latent_dim, channels, 1))
        self.alpha = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(init_alpha))
            for key in level_keys})

    def forward(self, features):
        for key in self.level_keys:
            delta = self.adapter(features[key])
            out[key] = features[key] + self.alpha[key] * delta
        return out
```

---

## 6. 预期目标（已下调并拆分）

### 6.1 主假设

1. **FPN-SM 应能在不牺牲 AP50 的前提下，提升 AP75 0.02~0.04**。这是比较稳健的期望，不应期待 0.58→0.65 的跳跃。
2. **ROI-SM 效果应弱于 FPN-SM**。7×7 频率分辨率太低，且不影响 RPN proposal。
3. **频率坐标门控是必要条件**。`fpn_sm_no_coords` 组若接近 `FPNRealAdapter`，说明 freq coords 的增量贡献有限。
4. **FPN-SM 若有效，应先体现在 RPN proposal 质量或 box regression loss 上**，而非仅体现在最终 AP75。

### 6.2 诊断指标解释（重要修正）

| 指标 | 原解释 | 修正后解释 |
|---|---|---|
| `alpha` | 应单调增长 | **不要求单调**。关键是部分 level 显著偏离 `1e-3`，且不同 level 出现分化 |
| `raw_rel_delta` | 0.05~0.2 | 仅表示残差本身强度 |
| `eff_rel_delta = \|α·dx\|/\|x\|` | 未记录 | **这是真正重要的扰动强度**，应稳定在 `1e-3 ~ 5e-2`，既非 0 也非过大 |
| `cos_delta_x` | 应趋于正相关 | **不要求正相关**。长期接近 1 反而可能意味着缩放式增强；接近 0 或小幅震荡更合理 |
| `freq_residual_ratio` | 应随训练下降 | **不要求下降**。没有 reconstruction loss，该指标只反映频域残差强度 |
| level 分化 | — | 若 P2/P3 的 `alpha/eff_rel_delta` 大于 P4/P5，说明模块偏向小目标/边界频率，符合预期 |

### 6.3 定量预期（Penn-Fudan，5 epoch fine-tune，3 seeds 平均）

| 组 | AP50 | AP75 | ECE | 说明 |
|---|---|---|---|---|
| baseline | ~0.86 | ~0.58 | ~0.10 | 已有结果 |
| FPN-SM coords | ≥0.85 | **+0.02~0.04** | ≤0.10 | 主方法 |
| FPN-SM no coords | ~0.86 | ~0.59 | ~0.10 | freq coords 对照 |
| ROI-SM | ~0.86 | ~0.59 | ~0.10 | 位置对照 |
| FPN-RealAdapter | ~0.86 | ~0.59 | ~0.10 | **最关键对照：区分低秩 adapter vs 频域** |

> 接受标准：AP50 不下降，AP75 有稳定提升，ECE 不恶化；若 FPN-SM 不优于 FPN-RealAdapter，则证明有效的是 adapter 而非频域。

---

## 7. 实验计划（待 GPU2 空闲后执行）

### 7.1 第一轮最小矩阵

| 组 | 命令关键参数 | 目的 |
|---|---|---|
| `baseline` | 无 | 基准 |
| `fpn_sm` | `--fpn-spectral-manifold` | 主方法 |
| `fpn_sm_no_coords` | `--fpn-spectral-manifold --no-fpn-sm-use-freq-coords` | freq coords 必要性 |
| `fpn_real` | `--fpn-real-adapter` | **非频谱低秩 adapter 对照** |
| `roi_sm` | `--roi-spectral-manifold` | 位置对照 |

**暂时不做 `fpn + roi` 组合实验**，避免两个模块耦合后难以归因。

### 7.2 第二轮扩展（视第一轮结果）

| 组 | 目的 |
|---|---|
| `fpn_sm_suppress_dc` | 验证 DC 分量影响 |
| `fpn_sm_gate_sigmoid2` / `tanh` | gate 输出范围 |
| `fpn_sm_p2p3_only` | 仅对 P2/P3 层启用 |
| `fpn + roi` | 叠加验证 |

### 7.3 运行命令示例

```bash
python scripts/round28_train_eval.py \
  --dataset pennfudan \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 5 \
  --seed 42 \
  --device cuda:2 \
  --run-name fpn_sm_pf_s42_5ep \
  --fpn-spectral-manifold
```

- seeds: `42, 2024, 999`
- 指标：AP50、AP75、P@R=0.85、ECE、Pred、box regression loss、RPN recall@0.75，以及模块诊断指标。

### 7.4 诊断指标落盘

训练脚本已修改：

- 每个 epoch 开始时调用 `_reset_spectral_stats(model)`；
- 每个 epoch 结束时调用 `_epoch_spectral_stats(model)`；
- 结果合并进 `history.json` / `eval_metrics.json`。

---

## 8. 与 RLVR 的关系

本次方案**暂时完全剥离 RLVR verifier**：频谱模块作为网络内部可微模块，通过标准 detection loss 端到端训练。只有当 FPN-SM 单独被验证有效后，才会考虑将其作为 RLVR 的“可学习 verifier”或“奖励特征提取器”重新接入。

---

## 9. 代码清单

新增/修改文件：

- `spectral_detection_posttrain/methods/manifold/complex_manifold.py`
- `spectral_detection_posttrain/methods/manifold/fpn_spectral_manifold.py`
- `spectral_detection_posttrain/methods/manifold/roi_spectral_manifold.py`
- `spectral_detection_posttrain/methods/manifold/fpn_real_adapter.py`（新增）
- `spectral_detection_posttrain/methods/manifold/__init__.py`
- `spectral_detection_posttrain/core/models/build_detector.py`
- `scripts/round28_train_eval.py`
- `docs/reports/fpn_roi_spectral_manifold_plan.md`

本地与远程均已通过 import、shape、backward、`build_detector` smoke test。
