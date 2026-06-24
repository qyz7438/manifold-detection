# RLIimage 实验总结与未来研究方向（2026-06-13）

## 一、项目当前状态总览

### 已完成的三条技术路径

| 路径 | 方案 | 结论 |
|------|------|------|
| **A** | 外部频谱 Verifier RLVR (Round 1-2.5) | **证伪** — 手工频谱特征压缩比 12544:1，TP/FP gap ~0.008，不能作为 RLVR reward |
| **B** | In-Network FFT/AFM (Round 2.6-2.90) | **成功验证** — AP75 +12.7%, ECE -25.8%，但效果是架构依赖的 |
| **C** | 语义分割 RLVR (Plan 4.0) | **未执行** — 仅概念阶段 |

### 核心定量结论（~140 组实验）

1. **AFM 帮助检测定位 (AP75)，但对分割 (mIoU) 无效**
   - 检测: mid06 AP75=0.690 vs baseline 0.584 (+18.2%)
   - 分割: baseline mIoU=0.521 > mid06 mIoU=0.491

2. **相位调制是 AFM 的关键机制** — phase-only AFM 即可达到完整 AFM 效果

3. **特征约束 (C) 后训练是检测最优配方** — AP75=0.674
   - C 持续优于弱门控 (A) 和 bbox 修正 (B)

4. **AFM 效果是架构依赖的** — 256ch ROI features（检测）受益，2048ch FCN features（分割）不受益

5. **外部分布式 RLVR reward 无效，in-network 梯度链路有效**
   - 同一 FFT 操作：forward 里是通路，loss 里是瓶颈

6. **后训练完美可复现** — 确定训练 + 相同 checkpoint → 相同结果

### 技术资产

- 稳定的 RLVR shell (KL-anchored signed policy objective + frozen baseline)
- MicroAFM + MultiScaleAFM (identity-preserving FFT block)
- 完整 NNI 实验框架
- Penn-Fudan Fast R-CNN MobileNetV3 基线 pipeline
- 代码迁移清理进行中 (round275-290 → shared utils)

---

## 二、相关研究全景扫描

### 2.1 RLVR/GRPO 用于视觉检测（直接竞品）

| 工作 | 年份/会议 | 核心方法 | 与本项目的关系 |
|------|-----------|---------|---------------|
| **Visual-RFT** | ICCV 2025 | GRPO + IoU/Confidence/Format reward, LVLM 检测 | 最直接对标 — 但用 MLLM 输出文本坐标，不用专用检测头 |
| **Visual-ARFT** | 2025 | Visual-RFT 的 Agent 扩展版本 | 多轮推理 |
| **Rex-Omni** | CVPR 2026 | GRPO + 几何感知 reward, 3B MLLM 检测 | MLLM 检测 SOTA |
| **Curriculum-GRPO** | 2025.10 | GRPO + 课程学习用于自动驾驶检测 | BDD-100K +7.4% IoU |
| **3D-RFT** | 2026.3 | GRPO + 3D IoU reward 用于视频 3D 感知 | 扩展到 3D |
| **CFCamo** | 2026 | 反事实配对 reward (CPR + CSPO) | 同时惩罚漏检和过检 |

**关键差异**: 这些工作都用 **MLLM/VLM 输出文本坐标**，本项目用 **专用 Faster R-CNN 检测头**。这是不同的模型范式 — 本项目更接近传统检测模型的后训练改进。

### 2.2 频域方法在视觉中的应用

| 工作 | 方法 | 关键发现 |
|------|------|---------|
| **MPLSeg** (Information Fusion 2024) | Magnitude-Aware + Phase-Sensitive 解耦学习 | 本项目的 AFM 直接灵感来源 |
| **FE-YOLO** (DSP 2025) | FFT 幅度扩展 + 相位约束 loss | 相位保结构，幅度增强亮度 |
| **FSDA-YOLO** (ICIC 2025) | FAT-Net + Spectral MHA | FFT 解耦高频边缘 vs 低频背景 |
| **FreqSal/DFENet** (TCSVT 2025) | 纯 FFT-based attention | FFT 替代 Self-Attention，O(N log N) |
| **MFD-KD** (2025) | FFT 特征图知识蒸馏 | 频率域做知识迁移 |
| **Fourier Phase Diffusion** (IJCAI 2025) | 相位谱引导扩散模型 | Phase = 结构信息 |

**关键趋势**: 2024-2025 年频域方法爆发式增长。幅度=语义/亮度，相位=结构/定位 -> 已成为共识。

### 2.3 小数据集检测后训练

| 策略 | 来源 | 效果 |
|------|------|------|
| Visual-RFT few-shot (1-100样本) | ICCV 2025 | COCO 2-shot +21.9 AP |
| Full-network fine-tuning | 2025 | 仅 fine-tune head 不够 |
| Dynamic data augmentation | 2025 | 有限样本多样化 |
| PoE 分类器 (修复背景误分类) | IEEE 2025 | 无需重训基模型 |

---

## 三、知识空白与机会

### 3.1 Visual-RFT 的 reward 设计缺陷（可直接改进）

Visual-RFT 是目前最相关的工作，但其 **reward function 从未被消融分析过**：

| 未解决的问题 | 本项目可做什么 |
|-------------|---------------|
| IoU vs Confidence vs Format reward 各贡献多少？ | 本项目的 reward 消融框架可直接回答 |
| IoU 阈值 τ 最优值？ | Penn-Fudan 上已有 IoU 阈值分析 |
| Greedy vs Hungarian 匹配的影响？ | 可用现有 box_iou 工具比较 |
| 缺少 recall penalty（漏检不惩罚） | 本项目的 geometric/energy reward 已包含此类惩罚 |
| KL 散度 β 敏感性？ | 本项目的 KL_WEIGHT/BETA 已有扫描数据 |

### 3.2 频域方法研究空白

| 空白 | 机会 |
|------|------|
| **相位-only AFM 未被充分研究** | 本项目已发现 phase-only = full AFM，但未系统消融 |
| **FFT 在检测 vs 分割上的差异未理论解释** | 256ch vs 2048ch 的维度效应需要分析 |
| **频域 loss 函数** (如 CFL, phase similarity loss) | 本项目未探索频域监督信号 |
| **自适应频段选择** (不同任务/层级用不同频段) | 当前 gate 对所有频段一视同仁 |
| **频域 + Transformer 混合架构** | 本项目纯 CNN，未试 Transformer backbone |
| **FFT 知识蒸馏** (MFD-KD 思路) | AFM 可作为蒸馏 student 的特征增强器 |

### 3.3 RLVR 方法在传统检测器上的机会

所有现有 RLVR 检测工作都用 MLLM/VLM，**没有人系统性地将 GRPO 应用到传统 Faster R-CNN/YOLO 等检测器上**。本项目是这个方向的先驱。

| 空白 | 本项目已有/需要的 |
|------|-----------------|
| GRPO 在传统检测器上的可行性 | ✅ 已建立 RLVR shell，但效果不显著 |
| 检测专用 reward 设计 | ⚠️ round275-290 的 reward 变体未在论文中发表 |
| Cross-proposal GRPO | ✅ round285 已实现 |
| Geometric + energy reward | ✅ round286 已实现 |
| In-network spectral + RLVR 组合 | ❌ 未尝试（mid06 只做了 fine-tune，没做 RLVR） |

---

## 四、候选未来研究方向

### 方向 1 (★ 高优先级): AFM + RLVR 联合训练

**idea**: 将 mid06 AFM 作为网络架构，在其上叠加 RLVR 后训练（而不是纯 fine-tune）。

**原理**: AFM 提供频域增强特征，GRPO RLVR 提供 reward 驱动的策略优化。两者互补 — AFM 改善特征质量（AP75 +12.7%），RLVR 可进一步优化 box regression 的探索行为。

**实验设计**: mid06 checkpoint → freeze backbone + AFM → GRPO with loc/geo/energy reward → 对比纯 fine-tune baseline

**风险**: RLVR 在检测上的梯度方差高（本项目的核心发现之一），AFM 特征不一定降低方差。

### 方向 2 (★ 高优先级): 频域 Reward 函数设计

**idea**: 用频域信息设计 reward 函数，而不是放在网络内部。本轮搜索发现 **FE-YOLO 的 amplitude difference loss + phase similarity loss** 以及 **FreqSal 的 Co-focus Frequency Loss (CFL)** 都直接在频域做监督。

**项目优势**: 本项目的 `extract_perchan_fft` 已有完整的频域特征提取。可设计：
- Phase similarity reward: 预测框 crop 的相位谱 vs GT 框 crop 的相位谱相似度
- Amplitude concentration reward: 幅度谱的能量集中度（低集中度 = 噪声/背景）
- Band-selective reward: 对 mid-band（边缘频段）的信号强度加权

**与现有路径 A 的区别**: 路径 A 的手工频谱特征压缩比太高（12544:1）。新设计可以在 proposal-level (N~100) 而非 pixel-level 做频域分析。

### 方向 3 (★ 中优先级): 相位解耦分析

**idea**: 系统消融 AFM 中的幅度 vs 相位贡献。本项目已偶然发现 phase-only AFM 效果与 full AFM 相当，但未系统研究。

**可探索的子方向**:
- Phase-only AFM 在不同 gate_strength 下的行为
- Phase randomization（打乱相位看效果是否消失）
- Phase 对不同层级特征的影响（浅层 vs 深层 FPN）
- Phase-conditioned attention（相位置信度作为 attention weight）

**理论价值**: 如果 phase = structure/location, magnitude = semantics 的假设成立，phase-only AFM 应该专门改善 AP75（定位），而不改善 AP50（分类）。

### 方向 4 (★ 中优先级): AFM 维度效应解释

**idea**: 解释为什么 AFM 在 256ch ROI features 上有效，在 2048ch FCN features 上无效。

**假设**:
- 过度参数化假说: 2048ch 已有足够的 spatial capacity，频域扰动无增量
- 感受野假说: FCN features 的感受野太大，频域特征噪声占比高
- 任务匹配假说: 分割需要 pixel-level 精度，FFT 的全局操作破坏了局部信息

**实验设计**: 在检测器中系统扫描 ROI feature channels (64/128/256/512) + AFM，观察效果峰值位置。

### 方向 5 (★ 中优先级): 跨数据集/跨架构验证

**idea**: 在更大规模数据集上验证 AFM 效果。

**候选**:
- VOC 20-class（已完成部分，欠训练）
- COCO mini（需要实现数据加载）
- VisDrone（小目标，频域方法可能特别有效）

**跨架构**: Swin Transformer + AFM（频域 + 自注意力的组合）

### 方向 6 (★ 探索性): 频域知识蒸馏

**idea**: 借鉴 MFD-KD 思路，将 AFM 用于知识蒸馏。Teacher 用 mid06，Student 用 baseline，在频域做特征对齐。

**优势**: 不需要 RLVR，纯监督学习，稳定性高。可在推理时不引入额外计算（仅训练时用 AFM teacher）。

### 方向 7 (★ 探索性): 检测 Reward 消融研究

**idea**: 本项目的 round275-290 包含了多种 reward 变体（loc, select, energy, geo, cross-proposal, pixel-action），但没有系统对比分析。

**价值**: 这组实验是**唯一对传统检测器 GRPO reward 设计的系统消融**。Visual-RFT 没有做 reward 消融。整理发表有学术价值。

---

## 五、建议的优先级排序

```
P0 (立即): 方向 1 (AFM+RLVR 联合) — 回答核心问题：频域特征 + RLVR 能否协同
P1 (本月): 方向 2 (频域 reward) + 方向 7 (reward 消融论文)
P2 (下月): 方向 3 (相位解耦) + 方向 4 (维度效应)
P3 (探索): 方向 5 (跨数据集) + 方向 6 (频域蒸馏)
```

---

## 六、与 Codex 讨论要点

1. **核心问题**: 本项目的 AFM 是"架构改进 + fine-tune"，不是真正的 RLVR 后训练。RLVR reward 在传统检测器上是否本质上有梯度方差上限，无法超越 fine-tune？

2. **差异化定位**: 所有现有 RLVR 检测工作都用 MLLM。本项目用传统检测器 + 频域特征 = 独特定位。但需要回答"为什么不直接用 MLLM"。

3. **论文发表策略**: 方向 7 (reward 消融) 可能是最快出论文的方向。方向 1 (AFM+RLVR) 如果成功最有影响力。

4. **资源决策**: Penn-Fudan 太小。是否需要迁移到 COCO 规模？8GB GPU 能否支持？

5. **理论深度**: 相位-幅度解耦的发现需要更深入的理论解释。是否值得做 spectral bias 分析？
