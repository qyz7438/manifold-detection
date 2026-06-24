# Plan A：可学习频域流形结构的基础模块实现

## 目标

构建一套可复用的底层基础设施，把复数谱系数/深度特征嵌入到可学习的高维流形上，并支持 ChordEdit 式的低能量最优传输。后续 Plan B/C/D/E/F 都依赖此模块。

## 核心设计

### 模块位置

```
spectral_detection_posttrain/methods/manifold/
├── __init__.py
├── complex_manifold.py       # 复数谱流形嵌入
├── riemannian_metric.py      # 自适应黎曼度量
├── chord_transport.py        # ChordEdit 式低能量传输
├── sinkhorn_ot.py            # 可微 Sinkhorn OT
└── tests/
    └── test_manifold.py      # 单元测试
```

### 1. ComplexSpectralManifold

把复数谱系数 $F \in \mathbb{C}^{d}$ 映射到低维潜流形坐标 $z \in \mathbb{C}^{k}$，再解码回原始空间。

```python
class ComplexSpectralManifold(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim=None):
        # encoder/decoder 都是复数 MLP
        # 输出满足 identity 初始化：初始时 decoder(encoder(F)) ≈ F

    def encode(self, F: Tensor[complex]) -> Tensor[complex]:
        # F: (..., d) -> z: (..., k)

    def decode(self, z: Tensor[complex]) -> Tensor[complex]:
        # z: (..., k) -> F: (..., d)

    def forward(self, F):
        return self.decode(self.encode(F))
```

**关键技术点**：
- 复数线性层：`ComplexLinear`（分别处理实部/虚部或直接用复数权重）；
- Identity 初始化：初始 decoder 权重近似 encoder 的伪逆，保证训练初期 $F_{\mathrm{recon}} \approx F$；
- 幅度-相位解耦：在 latent 空间中显式分离 $\rho = |z|$ 和 $\theta = \arg(z)$。

### 2. AdaptiveRiemannianMetric

学习流形上的局部度量 $M(z)$，使得不同方向（语义/结构/幅度/相位）有不同的距离代价。

```python
class AdaptiveRiemannianMetric(nn.Module):
    def __init__(self, latent_dim):
        # M(z) = U(z)^T U(z) + eps * I，保证正定

    def metric(self, z):
        # z: (..., k) -> M(z): (..., k, k)

    def local_distance(self, z1, z2):
        # d(z1, z2)^2 = (z1-z2)^* M(z) (z1-z2)
```

**初始化策略**：$M(z) = I$，初始退化为欧氏度量，训练后自适应。

### 3. ChordTransport

实现 ChordEdit 式的低能量传输。

```python
class ChordTransport(nn.Module):
    def __init__(self, manifold, metric, delta=0.15, lambda_step=1.0):
        # delta: 平滑窗口
        # lambda_step: 传输步长

    def forward(self, F_source, F_target_obs):
        # 1. 编码到流形: z_src, z_tar_obs
        # 2. 计算观测残差场 R
        # 3. 时间平滑得到 u_hat
        # 4. 单步传输: z_pred = z_src + lambda * u_hat
        # 5. 解码回谱系数
        return F_pred
```

**与图像编辑的区别**：
- ChordEdit 的源/目标是文本条件诱导的分布；
- 我们这里源/目标是具体的谱系数/特征（如干净图像 vs 对抗图像，或当前特征 vs 目标原型）。

### 4. SinkhornOT

PyTorch 原生实现的可微 Sinkhorn 距离，用于模块间解耦。

```python
class SinkhornOT(nn.Module):
    def __init__(self, eps=0.01, max_iter=100, p=2):
        # eps: entropic regularization
        # max_iter: Sinkhorn iterations

    def forward(self, mu, nu, cost_matrix):
        # mu, nu: probability distributions
        # cost_matrix: pairwise cost
        # return: Sinkhorn distance + gradients
```

**实现选择**：
- 优先手写 PyTorch 版本，不依赖外部包；
- 可选 `geomloss` 加速（如果允许安装）。

## 实现路线图

### Phase 1：复数 MLP 与流形嵌入（2 天）
- [ ] 实现 `ComplexLinear`, `ComplexMLP`；
- [ ] 实现 `ComplexSpectralManifold`；
- [ ] 验证重构误差：随机复数谱输入，重构误差 $< 1\%$；
- [ ] 验证 identity 初始化。

### Phase 2：黎曼度量（1-2 天）
- [ ] 实现 `AdaptiveRiemannianMetric`；
- [ ] 验证正定性；
- [ ] 在 toy 数据上验证学习后的度量能区分幅度/相位方向。

### Phase 3：Chord 传输（2 天）
- [ ] 实现 `ChordTransport`；
- [ ] 在 2D/3D toy 分布上做可视化，验证低能量路径；
- [ ] 对比硬差分 vs Chord 传输的稳定性。

### Phase 4：Sinkhorn OT（1-2 天）
- [ ] 实现 `SinkhornOT`；
- [ ] 与 `scipy.stats.wasserstein_distance` 在 1D 上对比验证；
- [ ] 验证梯度可传。

### Phase 5：单元测试与文档（1 天）
- [ ] 写 `tests/methods/test_manifold.py`；
- [ ] 写模块 README；
- [ ] 提交到 git。

## 验证方式

| 验证项 | 通过标准 |
|--------|---------|
| 复数流形重构 | 随机输入重构 MSE < 1e-3 |
| identity 初始化 | 训练 0 步时输出 ≈ 输入 |
| 度量正定性 | 对所有输入，M(z) 特征值 > 0 |
| Chord 传输能量 | 传输场 $L^2$ 范数 ≤ 朴素差分场 |
| Sinkhorn 精度 | 1D 分布距离与 scipy 一致 |

## 风险与依赖

| 风险 | 缓解 |
|------|------|
| 复数梯度支持不稳定 | 使用 PyTorch 复数张量，必要时拆实部/虚部 |
| 流形训练 collapse | identity 初始化 + 小的学习率 |
| OT 计算慢 | 先用小 batch，再考虑 `geomloss` |
| 高维 latent 学习困难 | 默认 $k=16$ 或 $32$，后续可调 |

## 预计时间

**7–10 天**（含测试与文档）。
