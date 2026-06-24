# Plan B：对抗防御方向的原型实现与验证

## 目标

基于 Plan A 的 `ComplexSpectralManifold` + `ChordTransport`，实现一个可插拔的频域对抗防御模块 `SpectralChordDefense`，并在目标检测任务上验证其对物理对抗补丁的防御效果。

## 问题定义

**攻击**：攻击者在图像中贴一个局部高频率补丁 $p$，使得模型预测崩溃。

**防御**：在不重新训练模型的情况下，对输入图像做频域净化：

$$x_{\mathrm{clean}} = \mathcal{F}^{-1}\left( \mathrm{ChordTransport}(\mathcal{F}(x_{\mathrm{adv}}); \mathcal{M}_{\mathrm{natural}}) \right)$$

其中 $\mathcal{M}_{\mathrm{natural}}$ 是自然图像的频域子流形。

## 模块设计

### 模块位置

```
spectral_detection_posttrain/methods/defense/
├── __init__.py
├── spectral_chord_defense.py      # 频域 Chord 防御
├── patch_attack.py                # 物理对抗补丁攻击
├── manifold_natural.py            # 自然图像频域流形（基于 Plan A）
└── eval_defense.py                # 防御评估
```

### 1. SpectralChordDefense

```python
class SpectralChordDefense(nn.Module):
    def __init__(self, manifold, transport, anomaly_gate_threshold=3.0):
        # manifold: ComplexSpectralManifold
        # transport: ChordTransport
        # anomaly_gate_threshold: 异常频率检测阈值

    def detect_anomaly(self, F):
        # 基于局部统计检测异常频率
        # s(u,v) = (|F| - local_mean) / local_std
        # mask = s > threshold
        return anomaly_mask

    def forward(self, x_adv):
        # 1. DFT
        F = torch.fft.rfft2(x_adv)
        # 2. 异常检测
        mask = self.detect_anomaly(F)
        # 3. 对异常区域做 Chord 传输，投影回自然流形
        F_clean = self.transport(F, F_natural_prototype)
        # 4. 仅修改被 mask 的频率，保留正常频率
        F_out = (1 - mask) * F + mask * F_clean
        # 5. iDFT
        x_clean = torch.fft.irfft2(F_out, s=x_adv.shape[-2:])
        return x_clean
```

### 2. AdversarialPatchAttack

实现针对检测器的物理对抗补丁攻击（RP2 风格）：

```python
class AdversarialPatchAttack:
    def __init__(self, model, patch_size, target_label=None, max_iter=1000):
        # model: Faster R-CNN 等检测器
        # target_label: None 表示无目标攻击（降低 AP）

    def attack(self, image, target_boxes=None):
        # 优化补丁像素 p，使得：
        # 1. 将 p 贴到图像上后模型检测失败；
        # 2. 补丁具有物理可行性（颜色约束、可打印性）。
        return patched_image
```

**简化版**：先做数字攻击（直接优化补丁像素），后续再扩展物理约束。

### 3. ManifoldNaturalModel

建模自然图像的频域分布。可以用以下方式之一：

- **高斯混合模型（GMM）**：在训练集频谱上拟合 GMM；
- **VAE**：训练一个频域 VAE；
- **原型集**：用训练集频谱的均值作为 $F_{\mathrm{natural}}$。

初始版本用**原型均值 + 协方差**即可。

## 实现路线图

### Phase 1：攻击实现（3 天）
- [ ] 实现简化版 `AdversarialPatchAttack`（数字攻击）；
- [ ] 在 Penn-Fudan 上生成对抗样本，测量 baseline AP 下降；
- [ ] 可视化干净图与对抗图的频谱差异。

### Phase 2：频域异常检测（1-2 天）
- [ ] 实现 `detect_anomaly`；
- [ ] 在对抗样本上验证异常 mask 能覆盖补丁引入的频率；
- [ ] 调参：阈值、局部窗口大小。

### Phase 3：Chord 防御（2-3 天）
- [ ] 接入 Plan A 的 `ComplexSpectralManifold` 和 `ChordTransport`；
- [ ] 实现 `SpectralChordDefense`；
- [ ] 测试 defense-as-preprocessing 效果。

### Phase 4：评估与消融（2 天）
- [ ] 测量指标：
  - 干净样本 AP（防御是否引入性能损失）；
  - 对抗样本 AP（防御恢复效果）；
  - 运行时间；
- [ ] 消融：
  - 硬阈值 vs Chord 传输；
  - magnitude-only vs phase-only vs combined；
  - 不同 latent_dim。

### Phase 5：文档与提交（1 天）
- [ ] 写实验报告；
- [ ] 提交代码到 git。

## 验证方式

| 验证项 | 通过标准 |
|--------|---------|
| 攻击有效 | 对抗补丁使 AP50 下降 ≥ 20% |
| 异常检测 | mask 与补丁频率位置对应（IoU > 0.5） |
| 防御恢复 | 防御后 AP50 恢复 ≥ 70% 的对抗损失 |
| 干净样本损失 | 防御后干净 AP50 下降 ≤ 3% |
| 可视化 | 净化后图像无明显 ringing 伪影 |

## 风险与依赖

| 风险 | 缓解 |
|------|------|
| 攻击实现复杂 | 先做数字攻击，再做物理约束 |
| 防御过度平滑 | 仅修改 anomaly mask 区域，加能量预算约束 |
| 运行慢 | 频域操作本身快，流形模块可用小 latent_dim |
| 自适应攻击 | 用 PGD 自适应攻击评估，迭代优化防御 |

## 预计时间

**8–10 天**（含攻击、防御、评估）。

## 与 Plan A 的依赖

必须等待 Plan A 完成，或至少完成 `ComplexSpectralManifold` 和 `ChordTransport` 的核心接口。
