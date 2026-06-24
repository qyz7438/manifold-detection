# Plans 2.21-5.2: 完整实验路线图

> 全部 deterministic (cudnn.benchmark=False)。2.x 单 seed，3-5.x 3 seed。
> 
> **核心目标**: 验证 A/C 后训练（弱门控 AFM-only / 特征约束 AFM-only）在检测和分割任务上普遍优于 baseline。

---

## 2.21 — 冻结组件消融

**问题**: 后训练时冻结 backbone/RPN/box_head，哪个对效果必要？

**设计**: 5 组 × seed42 × PF+MobV3 × A 后训练 2ep

| 组 | 冻结 | 训练 |
|---|---|---|
| freeze_all | backbone+RPN+box_head | AFM only |
| freeze_bb | backbone | AFM+RPN+box_head |
| freeze_rpn | RPN | AFM+backbone+box_head |
| freeze_box | box_head | AFM+backbone+RPN |
| freeze_none | 无 | 全模型 (负对照) |

---

## 2.22 — 数据量 Sweep

**问题**: 后训练收益是否依赖数据量？

**设计**: 4 量级 × 2 配置 (baseline+A) × seed42 × PF+MobV3 × 3ep

| 组 | 训练集 |
|---|---|
| d30_baseline / d30_A | 30 |
| d60_baseline / d60_A | 60 |
| d90_baseline / d90_A | 90 |
| d136_baseline / d136_A | 136 |

---

## 2.23 — 门控频域响应可视化

**问题**: gate conv 在压制/增强哪些频率分量？

**设计**: 纯分析，0 组训练。读 mid06_5ep checkpoint，在 val set 上分析 gate 输出的 per-frequency suppression 分布。画 gate_suppression vs input_magnitude 散点图。

---

## 2.24 — 频域变换类型消融

**问题**: FFT 特殊还是 DCT/Wavelet 也行？

**设计**: 3 变换 × 3 seed × PF+MobV3 × 3ep full fine-tune

| 组 | 变换 |
|---|---|
| fft_mid06 | FFT (正对照) |
| dct_mid06 | DCT2→门控→iDCT2 |
| dwt_mid06 | Haar wavelet→门控→逆Haar |

---

## 2.25 — 相位贡献消融

**问题**: 相位分支 (pa) 是否必要？

**设计**: 3 变体 × 3 seed × PF+MobV3 × 3ep full fine-tune

| 组 | 幅度 | 相位 |
|---|---|---|
| mag_only | mp active | pa pass-through |
| phase_only | mp pass-through | pa active |
| both | mp active | pa active (正对照) |

---

## 2.26 — 后训练配方 Sweep

**问题**: A (弱门控) vs C (特征约束) vs A+C，最优 epoch？

**设计**: 6 配方 × 3 seed × PF+MobV3, 用 mid06_5ep checkpoint 起点

| 组 | 配方 | epoch |
|---|---|---|
| A_2ep | weak gate 0.1 | 2 |
| A_5ep | weak gate 0.1 | 5 |
| C_2ep | feat constraint | 2 |
| C_5ep | feat constraint | 5 |
| AC_2ep | weak gate + constraint | 2 |
| AC_5ep | weak gate + constraint | 5 |

---

## 3.4 — VOC 20-Class Full

**问题**: A/C 后训练在多类别 (20 类) 检测上是否普遍有效？

**设计**: 2 backbone (MobV3, ResNet50) × 3 配置 (baseline, A, C) × 3 seed × VOC2012 full × 3ep

共 18 组。

---

## 3.5 — COCO Person Mini

**问题**: A/C 后训练在 COCO 域上是否有效？

**设计**: 2 backbone (MobV3, ResNet50) × 3 配置 (baseline, A, C) × 3 seed × COCO person mini (500 train/200 val) × 3ep

共 18 组。

---

## 3.6 — Backbone 矩阵

**问题**: AFM 效果是否依赖 backbone？

**设计**: 3 backbone (MobV3, ResNet50, ResNet101) × 2 数据集 (PF, VOC person 3-class) × 3 配置 (baseline, mid06, A) × 3 seed × 3ep

共 54 组。

---

## 3.7 — 检测终局报告

**问题**: 全部检测实验的 win rate 统计。

**设计**: 汇总 2.16/2.18/2.19/2.20/3.4/3.5/3.6 全部数据，输出 A/C post-training vs baseline 的 win rate，按 backbone × dataset 交叉表。

---

## 4.1 — PF 二值分割 Baseline

**问题**: 分割 pipeline 是否可行？

**设计**: FCN-ResNet18 + Penn-Fudan masks。建立最简分割基线。baseline only，3 seed，验证 mIoU 合理（>0.5）。

---

## 4.2 — MPLSeg-Style AFM on PF 分割

**问题**: mid06 AFM 在 pixel-level 预测上是否有效？

**设计**: 2 配置 (baseline, mid06) × 3 seed × PF 分割 × FCN-ResNet18 × 3ep full fine-tune

共 6 组。主指标 mIoU。

---

## 4.3 — 分割门控强度 Sweep

**问题**: 分割的最优门控强度与检测是否不同？

**设计**: 3 gate_strength (0.3, 0.6, 1.0) × 3 seed × PF 分割 × FCN-ResNet18 × 3ep

共 9 组。

---

## 4.4 — VOC Person Mask 多类分割

**问题**: 多类分割下，AFM 后训练是否仍有效？

**设计**: 2 配置 (baseline, A) × 3 seed × VOC person 3-class 分割 × FCN-ResNet18 × 3ep

共 6 组。

---

## 4.5 — 分割后训练 A/C 验证

**问题**: 分割的后训练范式与检测是否一致？

**设计**: 3 配置 (baseline, A, C) × 3 seed × PF 分割 × FCN-ResNet18 × 2ep post-training (冰冻 encoder, AFM+decoder 可训练)

共 9 组。

---

## 5.x — 已回收

5.x 计划（Cityscapes/ADE20K）已回收。先聚焦 4.x 分割出可靠结果。

---

## 总览

| Phase | Plans | 组数 | 预计时间 |
|-------|-------|------|---------|
| 2.x 消融 | 2.21-2.26 | ~49 | ~1h |
| 3.x 检测铺开 | 3.4-3.7 | ~90 | ~4h |
| 4.x 分割 | 4.1-4.5 | ~36 | ~2h |
| **合计** | **15 plans** | **~175** | **~7h** |
