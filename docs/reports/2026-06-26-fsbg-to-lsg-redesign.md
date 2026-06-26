# FSBG → LSG 重构设计报告

## 1. 当前 FSBG v3 的问题

### 1.1 实验结果

NWPU 50 epoch（bs=16, lr=0.024）：

| 方法 | mean AP50 | mean AP75 |
|---|---|---|
| Baseline v3 | 0.2005 | 0.0742 |
| FSBG v3 | 0.1852 | 0.0667 |

FSBG v3 不但没有提升，反而在 3 seeds 上平均下降了 **AP50 -0.0153 / AP75 -0.0075**。

### 1.2 根因分析

通过梯度探针和统计分析，我们发现 FSBG v3 的两个核心问题：

1. **phase boundary 信号弱**
   - phase boundary 输出与真实 Sobel 边缘的相关性只有 ~0.3
   - hard/none mask 几乎等同于随机噪声
   - 相位重建不是一条稳定的边界提取路径

2. **gate 设计偏离频域优势**
   - FSBG 先把频域信息转回空间域（spatial edge + phase reconstruction），再做一个空间注意力 gate
   - 本质上退化为“空间边缘注意力”，没有利用频域的尺度/噪声分离能力
   - 相比直接做 FFT→门控→iFFT（AFM 思路），FSBG 多了一层不可靠的相位近似

## 2. 重构方向：Learned Spectral Gate (LSG)

### 2.1 核心思想

**不再用相位重建近似边界，直接在频域学习一个可学习的门控函数。**

AFM（MPLSeg）已经证明：FFT→门控→iFFT 作为网络前向的一部分是有效的，梯度可以正常回传。AFM 使用硬编码的幅度门控（基于频率半径）。LSG 把门控换成**可学习的 1x1 卷积**，让网络自己决定哪些频率成分应该被保留、抑制或增强。

### 2.2 数据流

```
输入 x (B, C, H, W)
    ↓
rfft2 → X (B, C, H, W//2+1) 复数张量
    ↓
分离幅度 |X| 和相位 ∠X
    ↓
magnitude gate:  G_mag = 1 + tanh(Conv1x1(|X|))   ∈ (0, 2)
phase gate:      Δφ   = tanh(Conv1x1(∠X)) * π     ∈ (-π, π)   (可选)
    ↓
X' = G_mag * X * exp(j * Δφ)
    ↓
irfft2 → x' (B, C, H, W)
    ↓
输出 y = x + α * x'
```

### 2.3 与 FSBG 的本质区别

| 维度 | FSBG v3 | LSG |
|---|---|---|
| 操作空间 | 空间域 | 频域 |
| 门控对象 | spatial edge map + phase reconstruction | 幅度谱 / 相位谱 |
| 是否可学习 | 是，但基于弱相位近似 | 是，直接在频域学习 |
| 初始状态 | gate≈0，接近 identity | G_mag≈1，严格 identity |
| 梯度链路 | 经过相位重建，较绕 | FFT→Conv→iFFT，直接 |
| 参数量 | ~2C²/16 | ~C²/8 (mag+phase) |

## 3. 架构设计

```python
class LearnedSpectralGate(nn.Module):
    def __init__(self, channels, alpha_init=1.0, use_phase=True):
        super().__init__()
        self.use_phase = use_phase
        hid = max(1, channels // 4)

        # Magnitude gate: per-frequency scaling
        self.mag_gate = nn.Sequential(
            nn.Conv2d(channels, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, 1, bias=False),
        )

        # Phase gate: per-frequency rotation
        if use_phase:
            self.phase_gate = nn.Sequential(
                nn.Conv2d(channels, hid, 1, bias=False),
                nn.BatchNorm2d(hid),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid, channels, 1, bias=False),
            )

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, x):
        X = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(X)
        phase = torch.angle(X)

        # Gate centered at 1 → identity at init; can suppress (<1) or boost (>1)
        mag_gate = 1.0 + torch.tanh(self.mag_gate(mag))
        X_out = mag_gate * X

        if self.use_phase:
            phase_shift = torch.tanh(self.phase_gate(phase)) * math.pi
            X_out = X_out * torch.exp(1j * phase_shift)

        x_out = torch.fft.irfft2(X_out, s=x.shape[-2:], norm="ortho")
        return x + self.alpha * x_out
```

### 3.1 关键设计选择

1. **1x1 卷积**：频率谱每个位置对应一个频率/方向，1x1 conv 在不混淆空间位置的前提下学习 per-channel 的频响。
2. **G_mag 中心化为 1**：初始严格等于 identity，避免破坏预训练特征。
3. **α 初始为 1.0**：允许 LSG 在第一轮就有显著影响，同时 residual 形式保证稳定性。
4. **相位分支可选**：如果相位分支引入噪声，可以关闭只保留幅度门控。
5. **BN + ReLU**：在小 batch（16）下稳定训练，比 FSBG 的 3x3 conv 更适合处理谱图。

## 4. 预期优势与风险

### 4.1 优势

- 直接利用 AFM 已验证有效的 FFT/iFFT 梯度链路
- 门控是可学习的，网络可以自己找到对检测有用的频率
- 不依赖相位重建这种弱信号
- 初始 identity，训练稳定

### 4.2 风险

- NWPU 数据集小，可学习门控可能过拟合
- 如果网络学到 uniform gate，会退化为 AFM 的均匀缩放
- 相位分支可能引入额外噪声
- 需要更多 epoch 才能体现学习门控的优势

## 5. 验证计划

1. **单元测试**：梯度探针确认 FFT/iFFT 链路梯度流通
2. **统计分析**：检查 mag_gate 的均值/方差/频率选择性
3. **NWPU 50ep 实验**：baseline + LSG (mag only) + LSG (mag+phase)，3 seeds
4. **与 AFM 对比**：确认 learned gate 是否优于 hard-coded AFM

## 6. 结论

FSBG 系列的问题不在“梯度是否流通”，而在**问题建模错误**：试图用相位重建做空间边界检测，偏离了频域操作的本质优势。LSG 通过直接在频域学习门控，回归了 AFM 的成功路径，同时赋予网络自适应选择频率的能力，是更值得验证的方向。
