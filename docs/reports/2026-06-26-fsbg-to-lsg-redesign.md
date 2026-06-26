# FSBG → LSG 重构设计报告（v1 修正版）

## 1. FSBG 的问题与重构方向

### 1.1 实验结果

NWPU 50 epoch（bs=16, lr=0.024）：

| 方法 | mean AP50 | mean AP75 |
|---|---|---|
| Baseline v3 | 0.2005 | 0.0742 |
| FSBG v3 | 0.1852 | 0.0667 |

FSBG v3 不但没有提升，反而下降了 **AP50 -0.0153 / AP75 -0.0075**。

### 1.2 根因

FSBG 的核心问题是**把频域问题绕回了空间边界检测**：

```
phase reconstruction → spatial edge map → spatial gate
```

相位重建与真实 Sobel 边缘的相关性只有 ~0.3，hard/none mask 几乎等同于噪声。后续的空间 gate 本质上只起到了弱边缘注意力的作用，没有利用频域的尺度分离和噪声分离能力。

### 1.3 正确方向

**直接在频域学习频率响应**，而不是用相位重建近似边界。

| 方法 | 建模对象 | 问题 |
|---|---|---|
| FSBG | 用相位重建近似空间边界 | proxy 弱 |
| LSG | 直接学习幅度谱的频率响应 | 更贴近频域优势 |

这与 AFM、AFNO 等频域模块的经验一致：FFT → 可学习变换 → iFFT 作为网络前向链路是合理的。

---

## 2. LSG v1 设计

### 2.1 数据流

```
x (B, C, H, W)
  ↓
rfft2 → X (B, C, H, W//2+1)
  ↓
|X| → log1p → per-sample/channel norm
  ↓
[concat radius_map]
  ↓
Conv1x1(hid) → BN → ReLU → Conv1x1(C, zero-init) → tanh → ΔG
  ↓
G = 1 + ΔG
  ↓
X' = G * X
  ↓
irfft2 → x_filt
  ↓
y = x + α * (x_filt - x)
```

### 2.2 关键修正（相比第一版 LSG）

| 问题 | 第一版 LSG | LSG v1 |
|---|---|---|
| 初始输出 | `x + 1.0 * x' ≈ 2x` | `x + α * (x_filt - x) = x` |
| α 初始值 | 1.0 | **0.1**（0 会导致梯度死锁） |
| 最后一层初始化 | 随机 | **零初始化** -> ΔG=0 |
| 输入幅度 | raw \|X\| | **log1p + 标准化** |
| 频率位置感知 | 无 | **加入 radius_map** |
| phase 分支 | 默认开启 | **先关闭** |
| FFT 精度 | 输入 dtype | **显式 float32** |

### 2.3 关于 α 初始值的说明

理论上 α=0 可以保持严格 identity，但它同时会让 `(x_filt - x)` 前的系数为零，导致 gate 网络的梯度也归零，模块无法学习。因此选择 **α=0.1** 作为初始：

- 输出近似 identity（误差 < 1e-7）
- gate 网络仍能获得有效梯度
- 随着 gate 偏离 1，α 也会获得梯度并自适应调整

---

## 3. 代码实现

```python
class LearnedSpectralGate(nn.Module):
    def __init__(self, channels, alpha_init=0.1, use_radius=True):
        super().__init__()
        self.use_radius = use_radius
        self.eps = 1e-6
        hid = max(1, channels // 4)

        in_ch = channels + (1 if use_radius else 0)
        self.mag_gate = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, 1, bias=False),
        )
        # 关键：零初始化 -> 初始 gate 严格为 1
        nn.init.zeros_(self.mag_gate[-1].weight)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def _radius_map(self, h, w_half, device, dtype):
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(1, 1, h, 1)
        fx = torch.fft.rfftfreq((w_half - 1) * 2, device=device, dtype=dtype).view(1, 1, 1, w_half)
        r = torch.sqrt(fx ** 2 + fy ** 2)
        r = r / (r.max() + self.eps)
        return r

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.float()  # FFT 用 float32

        X = torch.fft.rfft2(x_float, norm="ortho")

        # log + normalization：学相对频谱结构
        mag = torch.log1p(torch.abs(X))
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (
            mag.std(dim=(-2, -1), keepdim=True) + self.eps
        )

        if self.use_radius:
            r = self._radius_map(mag.size(-2), mag.size(-1), mag.device, mag.dtype)
            r = r.expand(mag.size(0), 1, mag.size(-2), mag.size(-1))
            gate_input = torch.cat([mag, r], dim=1)
        else:
            gate_input = mag

        delta = torch.tanh(self.mag_gate(gate_input))
        gate = 1.0 + delta  # 初始为 1

        X_out = gate * X
        x_filt = torch.fft.irfft2(X_out, s=x_float.shape[-2:], norm="ortho")

        # 残差修正：严格 identity
        y = x_float + self.alpha * (x_filt - x_float)
        return y.to(orig_dtype)
```

---

## 4. 实验计划

按问题优先级排序：

| 组 | 配置 | 要回答的问题 |
|---|---|---|
| A | Baseline v3 | 参照 |
| B | AFM hard-coded | 频域门控本身是否有用？ |
| C | LSG-mag-only（无 radius） | learned gate 是否比 hard-coded 强？ |
| D | LSG-mag-only + radius | 显式频率坐标是否必要？ |
| E | LSG-mag-only + radius + α=0.1 | 验证最终版本 |
| F | LSG-mag+phase（后续） | phase 是否真贡献？ |

**第一阶段先做 A/B/C/D/E，F 留到后面。**

### 4.1 启动命令

```bash
# Baseline（已有）
python scripts/round28_train_eval.py --run-name nwpu_baseline_v3_50ep_bs16_s42 \
  --dataset nwpu --trainable-mode box_head_only --epochs 50 --seed 42

# LSG v1 mag-only + radius
python scripts/round28_train_eval.py --run-name nwpu_lsg_v1_radius_s42 \
  --dataset nwpu --use-lsg --lsg-alpha-init 0.1 --lsg-use-radius \
  --trainable-mode box_head_only --epochs 50 --seed 42

# LSG v1 mag-only 无 radius
python scripts/round28_train_eval.py --run-name nwpu_lsg_v1_noradius_s42 \
  --dataset nwpu --use-lsg --lsg-alpha-init 0.1 \
  --trainable-mode box_head_only --epochs 50 --seed 42
```

### 4.2 评估指标

除了 AP50/AP75，还应关注：

- **Precision / Recall / High-conf FP**：LSG 若抑制背景噪声，应先看 precision
- **Radial gate profile**：low / mid / high frequency gate mean
- **Gate entropy / std**：是否退化为 uniform scaling
- **Small-object AP**（如可能）：遥感场景中边界/高频结构是否改善
- ** per-image gate profile**：TP-heavy vs FP-heavy 图像的 gate 差异

---

## 5. 本地探针验证结果

```
=== LSG v1 probe (use_radius=True) ===
input shape       : (2, 256, 14, 14)
output shape      : (2, 256, 14, 14)
max |y - x|       : 1.19e-07  (identity at init)
input grad ok?    : True
alpha value       : 0.1000
alpha grad        : -0.0000

gate stats:
  mean : 1.0000
  std  : 0.0000
  min  : 1.0000
  max  : 1.0000
```

- 初始输出严格等于 identity
- 梯度正常流通
- gate 初始为 1（uniform），训练过程中会学习偏离

---

## 6. 风险与注意事项

1. **α=0 会死锁**：已从默认值中排除，用 0.1。
2. **gate 可能学不会频率选择**：需要 radial profile 监控。
3. **NWPU 数据量小**： learned gate 可能过拟合；若无效，不一定是思路错，可能是数据不足。
4. **phase 分支风险高**：跳变、弱幅度相位不稳定、Hermitian 约束，先不做。
5. **FFT 精度**：已强制 float32，避免 AMP 下 half FFT 尺寸限制。

---

## 7. 结论

FSBG 的失败说明**相位重建边界不是稳定的检测特征增强路径**。LSG 改为直接在频域学习幅度谱门控，方向正确。

当前 LSG v1 已修正第一版的 identity、phase、频率坐标和数值稳定性问题，可以进入实验验证阶段。第一阶段应优先对比：

- Baseline v3
- AFM hard-coded
- LSG-mag-only（无 radius）
- LSG-mag-only + radius

以回答“频域门控是否有用”、“learned gate 是否优于 hard-coded gate”、“频率坐标是否必要”三个核心问题。
