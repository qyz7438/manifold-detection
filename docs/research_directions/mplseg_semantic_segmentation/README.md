# MPLSeg 高维流形扩展：语义分割方向

## 目标

在 MPLSeg（*Decoupling Semantic and Localization for Semantic Segmentation via Magnitude-aware and Phase-sensitive Learning*, Information Fusion 2024）的 magnitude/phase 解耦框架基础上，引入**高维流形参数化**，进一步提升语义分割在边界定位与类别语义上的一致性。

## 核心思路

MPLSeg 的关键洞察：
- **幅度谱（magnitude）** 主要承载语义信息；
- **相位谱（phase）** 主要承载结构/定位信息。

现有 AFM（Adaptive Frequency-aware Module）通过可学习的 magnitude gate 与 phase residual 在单点频率上做门控。我们提出把这一操作提升到**高维流形层面**：

1. **谱系数流形（Spectral Coefficient Manifold）**
   - 将每个空间位置的频率响应视为复数向量 $z \in \mathbb{C}^d$；
   - 在通道维度上构造一个低维潜流形 $\mathcal{M} \subset \mathbb{C}^d$，用流形坐标 $(\rho, \theta)$ 参数化幅度与相位；
   - 学习一个可微映射 $f: \mathcal{M} \to \mathcal{M}$，在流形上完成 magnitude/phase 的解耦调制。

2. **黎曼门控（Riemannian Gate）**
   - 用流形上的测地距离替代欧氏门控：
     $$g_{\mathrm{mag}} = \sigma\left( -\frac{d_{\mathcal{M}}(z, z_{\mathrm{semantic}})^2}{2\tau^2} \right)$$
   - 该门控对幅度变化更敏感，同时保持相位结构不变。

3. **相位子空间约束（Phase Subspace Constraint）**
   - 将相位投影到切空间 $T_z \mathcal{M}$ 的一个子空间；
   - 通过正交补空间约束抑制高频噪声，同时保留边界结构。

## 与本项目资产的衔接

| 现有资产 | 复用方式 |
|---------|---------|
| `spectral_detection_posttrain/methods/afm/micro_afm.py` | `MPLSegAFMBlock`, `PhaseOnlyAFMBlock` 可直接迁移到分割 backbone |
| `spectral_detection_posttrain/methods/segmentation/` | 分割任务入口与数据加载 |
| `docs/segmentation_technical_plan.md` | 已有的 PF 分割 baseline 与失败分析 |
| `scripts/analyze_segmentation_signal_complementarity.py` | 信号互补性分析可复用 |

## 实现路线图

### Phase 1：基线复现（1-2 天）
- [ ] 在 `spectral_detection_posttrain/methods/segmentation/` 中建立 FCN-ResNet50 / DeepLabV3+ 的 MPLSeg-AFM 基线；
- [ ] 复现 PF 分割 baseline mIoU=0.5208 与 mid06 mIoU=0.4910；
- [ ] 定位 mid06 在分割上失效的根因（通道数 2048 vs 检测 256、特征分辨率差异等）。

### Phase 2：高维流形模块设计（2-3 天）
- [ ] 实现 `ManifoldAFMBlock`：将复数谱系数展平为流形坐标；
- [ ] 引入可学习的流形基 $U \in \mathbb{C}^{d \times k}$，$k \ll d$；
- [ ] 在潜坐标上做 magnitude/phase 解耦，再映射回像素谱；
- [ ] 保留 identity 初始化，确保训练稳定。

### Phase 3：实验与诊断（3-5 天）
- [ ] 在 Pascal VOC / Penn-Fudan 分割上跑 ablation：
  - 流形维度 $k$；
  - 是否对 magnitude/phase 分别约束；
  - 残差模式（current / delta / norm_delta）；
- [ ] 监控 ECE、mIoU、boundary IoU；
- [ ] 可视化流形上的 magnitude/phase 分布。

### Phase 4：扩展（可选）
- [ ] 将 Manifold-AFM 接入 mmsegmentation；
- [ ] 尝试 Cityscapes / ADE20K 验证通用性。

## 预期产出

- 一篇技术报告：高维流形参数化如何影响 magnitude/phase 解耦；
- 代码：可插拔的 `ManifoldAFMBlock`；
- 指标：在 PF/VOC 分割上 mid06 失败根因被解决，mIoU 提升 ≥3%。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 2048ch 特征维度太高，流形学习困难 | 先对通道做 group/channel-wise PCA 降维 |
| 分割特征分辨率低，频谱信息稀疏 | 在深层特征（stride 8/16）而非 stride 32 插入 AFM |
| 训练不稳定 | identity 初始化 + 小的学习率 + 冻结 backbone 前几层 |
