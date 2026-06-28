# NWPU VHR-10 实验矩阵结果分析

## 1. 配置

- 数据集：NWPU VHR-10（10 类遥感小目标，454 训练 / 196 验证）
- 训练：12 epoch，3 seeds（42 / 2024 / 999）
- 分支 A：MobileNetV3-FPN，输入 320/480，lr 0.024，batch 16
- 分支 B：ResNet50-FPN，输入 800/1333，lr 0.005，batch 4
- fpn_sm 配置：latent=64/hidden=128，init_alpha=0.01，无 freq/level 坐标

## 2. 跨 seed 均值

| 组 | AP50 | AP75 | ECE | best_AP50 |
|---|------|------|-----|-----------|
| nwpu_mob_baseline | 0.4955 ± 0.0314 | 0.1464 ± 0.0253 | 0.0423 | 0.5335 |
| **nwpu_mob_fpn_sm** | **0.5472 ± 0.0136** | **0.1825 ± 0.0248** | 0.1103 | 0.5494 |
| nwpu_resnet_baseline | 0.9162 ± 0.0118 | **0.8069 ± 0.0189** | 0.0493 | 0.9304 |
| nwpu_resnet_fpn_sm | 0.9205 ± 0.0209 | 0.7546 ± 0.0779 | 0.0496 | 0.9340 |

## 3. 关键发现

### 3.1 MobileNetV3 分支：fpn_sm 提升 AP50/AP75，但 ECE 明显变差
- AP50 从 0.496 → 0.547（+10.4%），AP75 从 0.146 → 0.183（+24.7%）。
- 但 ECE 从 0.042 → 0.110，说明模块在困难小目标上产生了过度自信的分数。
- 可能原因：低分辨率输入下，模块学到的反相压缩使分类 logits 更分散，置信度被错误放大。

### 3.2 ResNet50 分支：输入分辨率是主导因素，fpn_sm 对 AP75 无净收益
- ResNet50 baseline 已达到 AP50 0.916、AP75 0.807，接近 NWPU 在该设置下的上限。
- 加 fpn_sm 后 AP50 微升（+0.004），AP75 均值下降（0.807 → 0.755）。
- AP75 方差大（std 0.078），其中 seed42 只有 0.667，而 seed2024 达到 0.816，说明对随机初始化敏感。
- 诊断指标显示 ResNet50 上 alpha 很小（key0 仅 ~0.019），eff_rel_delta ~0.02，压缩强度远低于 Penn-Fudan（key0 alpha ~0.5）。

### 3.3 模块强度随模型/分辨率变化
| 设置 | key0 alpha | eff_rel_delta | 压缩强度 |
|---|-----------|---------------|----------|
| Penn-Fudan MobileNetV3 | ~0.50 | ~0.48 | 强 |
| NWPU MobileNetV3 | ~0.37 | ~0.36 | 中等 |
| NWPU ResNet50 | ~0.02 | ~0.02 | 极弱 |

这说明模块学到的是一个“自适应缩放器”：当 backbone 本身很强、输入分辨率足够时，它会学到很小的残差（避免破坏已有特征）；只有在 backbone/输入较弱时，它才会学到较强的压缩来提升边界。

## 4. 对论文的影响

### 正面
- 跨数据集（NWPU）仍能看到 AP50/AP75 提升，说明模块不是只在 Penn-Fudan 上过拟合。
- ResNet50 上 AP50 不下降，说明模块对 strong backbone 是安全的。

### 负面/需注意
- **收益高度依赖输入分辨率和 backbone 强度**。在 ResNet50 高分辨率下，模块几乎失效；在 MobileNetV3 低分辨率下才有效。
- **ECE 在 MobileNetV3 上明显恶化**，需要后续加入置信度校准或调整 init_alpha/门控。
- NWPU 上结果不足以支撑“通用 FPN 增强模块”的强 claim；更像是“在低容量/低分辨率检测器上做特征正则化”。

## 5. 下一步建议

1. **VOC 仍是关键验证**：NWPU 结果混杂，VOC 20-class 能给出更清晰结论。需要补 VOC 数据并跑标准 schedule。
2. **降低 ResNet50 上的 init_alpha**：尝试 0.001 或让 alpha 随 capacity 自适应，避免在高分辨率下过扰动。
3. **加入置信度正则化**：如对 alpha 做 L2 约束，或在 loss 中加入 ECE-aware 项。
4. **可视化**：比较 ResNet50 baseline vs fpn_sm 的 FPN 特征幅度谱，确认模块是否真的学到“弱压缩”。
