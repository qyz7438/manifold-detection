# MPLSeg 思路迁移：对抗鲁棒性方向

## 目标

针对**物理空间补丁对抗攻击**（如在人胸口贴一张高频率乱码图片即可破坏检测/识别模型），利用 MPLSeg 的 magnitude/phase 解耦机制与**高维流形 + 复数域结构**，设计一个训练无关、可插拔的频域防御模块。

## 问题背景

物理对抗补丁通常具有以下频域特征：
- **高频能量异常集中**：乱码/棋盘格图案在幅度谱高频区产生尖峰；
- **相位结构局部相干**：真实纹理的相位在全局上更连续，而对抗补丁的相位往往呈现局部强相干、全局不一致；
- **空间局部性**：补丁只占图像一小部分，但频域影响是全局的。

MPLSeg 的启示：通过 magnitude gate 压制异常语义，通过 phase residual 修正结构。我们将这一思想用于**防御**：把对抗扰动视为“异常频域能量”，在频域上对其进行低能量、结构保持的净化。

## 核心思路

### 1. 频域异常检测门（Spectral Anomaly Gate）

对输入图像做 2D-DFT 得到 $F \in \mathbb{C}^{H \times W \times C}$。

构造一个基于统计的异常分数：
$$s_{\mathrm{anomaly}}(u,v) = \frac{|F(u,v)| - \mu_{\mathrm{local}}(u,v)}{\sigma_{\mathrm{local}}(u,v)}$$

其中 $\mu_{\mathrm{local}}, \sigma_{\mathrm{local}}$ 是在频域邻域内计算的局部均值/方差。分数超过阈值的位置被认为是异常频率。

### 2. 高维流形上的低能量投影

受 ChordEdit（arXiv:2602.19083）启发：
> 高能量扰动场在单步积分下不稳定；低能量最优传输场能更好地保持结构。

我们将复数谱系数 $F$ 视为高维流形 $\mathcal{M} \subset \mathbb{C}^{HWC}$ 上的点。防御操作不是直接裁剪异常频率（硬阈值会引入 ringing 伪影），而是沿着测地线将异常点投影回流形上的“自然图像子流形”：

$$F_{\mathrm{clean}} = \exp_{F}( -\eta \cdot \nabla_{\mathcal{M}} D(F, \mathcal{M}_{\mathrm{natural}}) )$$

其中 $D$ 是到自然图像子流形的距离，可用一个轻量 autoencoder 或高斯混合模型近似。

### 3. 幅度-相位解耦净化

- **Magnitude 净化**：对异常幅度做软阈值，降低高频尖峰；
- **Phase 净化**：对相位做局部平滑，破坏对抗补丁的相干结构，同时保留真实边界；
- **复数域约束**：净化后的 $F_{\mathrm{clean}}$ 仍满足 Hermitian 对称性，保证实数输出。

### 4. 训练无关 / 可插拔

该防御模块可插入任何预训练检测/分类模型之前：
```
input image → DFT → Manifold Projection → iDFT → clean image → pretrained model
```

## 与本项目资产的衔接

| 现有资产 | 复用方式 |
|---------|---------|
| `spectral_detection_posttrain/spectral/fft_features.py` | FFT/iFFT 与谱特征提取 |
| `spectral_detection_posttrain/methods/afm/micro_afm.py` | AFM 门控机制可直接改造为防御门 |
| `scripts/analyze_*_fft*.py` | 异常频率分析脚本 |
| `data/PennFudanPed`, `data/NWPU VHR-10` | 行人/通用目标检测数据集 |

## 实现路线图

### Phase 1：对抗补丁生成与基线攻击（2-3 天）
- [ ] 实现 `AdversarialPatchAttack`：针对 Faster R-CNN / ResNet 的物理补丁攻击（RP2 风格）；
- [ ] 在 Penn-Fudan 上生成补丁并测量 AP 下降；
- [ ] 记录干净图与对抗图的频域差异。

### Phase 2：频域防御原型（2-3 天）
- [ ] 实现 `SpectralDefense`：包含 anomaly gate + magnitude/phase purification；
- [ ] 在检测/分类任务上测试 defense-as-preprocessing；
- [ ] 测量：防御后 AP / 准确率恢复、干净样本性能损失、运行时间。

### Phase 3：高维流形投影（3-5 天）
- [ ] 训练一个小的 VAE/GMM 在频域系数上建模自然图像子流形；
- [ ] 实现测地线投影或基于 OT 的低能量传输；
- [ ] 对比硬阈值 vs 流形投影的伪影与防御效果。

### Phase 4：联合训练（可选）
- [ ] 将防御模块与检测器端到端微调；
- [ ] 探索可微防御层在 RLVR 奖励中的角色。

## 预期产出

- 技术报告：基于 MPLSeg 思想的频域对抗防御框架；
- 代码：`SpectralDefense` 模块与对抗补丁攻击实现；
- 指标：在物理补丁攻击下 AP 恢复 ≥15%，干净样本 AP 损失 ≤3%。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 频域预处理可能抹去小目标细节 | 只在异常分数高的频率做软阈值，并限制最大修改幅度 |
| 流形模型训练成本高 | 先用高斯混合或 PCA 近似，再考虑 VAE |
| 对抗样本可能自适应攻击防御模块 | 采用自适应训练（adaptive attack）评估，并不断迭代 |
