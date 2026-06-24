# MPLSeg 思路迁移：图像分类方向

## 目标

将 MPLSeg 中“幅度承载语义、相位承载结构”的解耦思想，迁移到图像分类任务。探索在分类 backbone 中插入基于 FFT 的 magnitude/phase 调制模块，是否能提升：
- 标准分类准确率；
- 对遮挡、噪声、分辨率变化的鲁棒性；
- 对抗样本防御能力（与 adversarial robustness 方向部分重叠）。

## 核心思路

分类模型通常只需要语义信息，对定位信息不敏感。然而，边界/纹理信息（相位）对细粒度分类、低分辨率输入、以及对抗扰动识别至关重要。

我们提出 **Spectral Classifier Head**：

1. **全局频谱池化（Global Spectral Pooling）**
   - 对最后一层特征图做 2D-DFT：$F = \mathcal{F}(X) \in \mathbb{C}^{H \times W \times C}$；
   - 分别对 magnitude 和 phase 做全局统计池化：
     - magnitude：mean / max / learned weighted sum；
     - phase：circular mean（用 atan2 处理周期性）。

2. **Magnitude-Phase 解耦分类头**
   - 两个并行的轻量化 MLP：
     - $g_{\mathrm{mag}}(F_{\mathrm{mag}})$ 预测粗粒度语义；
     - $g_{\mathrm{phase}}(F_{\mathrm{phase}})$ 预测结构/纹理模式；
   - 最终 logits 通过可学习融合：$y = \alpha \cdot g_{\mathrm{mag}} + (1-\alpha) \cdot g_{\mathrm{phase}}$。

3. **频域数据增强（Spectral Augmentation）**
   - 在训练时随机扰动 magnitude 的低频分量（模拟亮度变化）；
   - 随机扰动相位的高频分量（模拟纹理/边缘抖动）；
   - 增强模型对真实世界退化的鲁棒性。

## 与本项目资产的衔接

| 现有资产 | 复用方式 |
|---------|---------|
| `mfvpt/` | 已有 CIFAR-100 分类实验代码，可快速启动 |
| `spectral_detection_posttrain/spectral/fft_features.py` | FFT 特征提取函数 |
| `spectral_detection_posttrain/methods/afm/micro_afm.py` | AFM block 可改造为分类模块 |

## 实现路线图

### Phase 1：CIFAR-100 快速验证（1-2 天）
- [ ] 在 ResNet-18 / ResNet-50 最后一层后插入 Spectral Classifier Head；
- [ ] 训练 200 epoch，对比 baseline 准确率；
- [ ] 做 corruption robustness 测试（CIFAR-100-C 子集）。

### Phase 2：高维流形参数化（2-3 天）
- [ ] 将 magnitude/phase 的通道向量视为复数流形上的点；
- [ ] 学习低维流形嵌入，并在潜空间做分类；
- [ ] 引入流形上的 contrastive loss，拉近同类、推远异类。

### Phase 3：细粒度与可解释性（2-3 天）
- [ ] 在 CUB-200 / Stanford Cars 上验证细粒度分类；
- [ ] 可视化哪些频率带对哪些类别贡献最大；
- [ ] 分析 magnitude-only、phase-only、combined 的性能差异。

### Phase 4：与对抗防御结合（可选）
- [ ] 与 `mplseg_adversarial_robustness` 方向共享频域净化模块。

## 预期产出

- 技术报告：频域解耦在分类任务上的有效性分析；
- 代码：`SpectralClassifierHead` 模块；
- 指标：CIFAR-100 上准确率提升或鲁棒性明显提升。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 分类任务对全局语义更敏感，phase 信息可能噪声大于信号 | 先做 ablation：只加 magnitude head |
| FFT 全局池化丢失空间信息 | 保留空间 attention 分支，与频域分支并行 |
| CIFAR-100 图像太小，频域分辨率不足 | 同时尝试 224×224 数据集（如 Tiny-ImageNet） |
