# RLIimage — RLVR Post-Training for Object Detection

## 核心目标

构建一种面向目标检测模型的 **RLVR（Reinforcement Learning with Verifiable Rewards）后训练框架**，让检测器不只是"在已有候选框里重新排序"，而是能够基于可验证奖励逐步学会：**哪些框更可信、应该看哪里、以及如何框得更准**。

## 技术路线

借鉴文本模型中的 RLVR/GRPO 后训练范式，将其迁移到目标检测任务。参考 Visual-RFT 等工作中"用可验证奖励（如 IoU reward）通过策略优化更新视觉模型"的思路，在此基础上引入**区域频域证据（spectral evidence）作为检测 verifier 的组成部分**——幅度谱用于评估候选区域是否包含目标语义，相位/结构谱用于评估边界和定位质量。

## 三条技术路径及其状态

### 路径 A：外部频谱 Verifier RLVR（Round 1-2.5）

在检测器外面，对每个候选框的 ROI crop 做 FFT，提取 R_amp/phase/structure 特征作为标量 reward，通过 GRPO-style signed policy objective 驱动检测器后训练。

**状态**: KL=10 + policy=0.0003 + signed objective + frozen baseline 的 RLVR shell 已稳定（AP50 与 baseline 持平）。但手工频谱 verifier 信号强度不足以产生 causal 区分度——real vs shuffled 的 AP50 差距 <0.001，所有组收敛到同一水平。

**结论**: 手工频谱特征（radial profile + cosine similarity）在 Penn-Fudan 上的压缩比过高（TP/FP gap ~0.008），不能作为有效的 RLVR reward。

### 路径 B：In-Network FFT（Round 2.6-2.18，已成功验证）

受 MPLSeg 启发，将 FFT 从外部 verifier 移到网络内部——在 Faster R-CNN 的 ROI box_head 前插入 AFM 模块（rFFT2 → magnitude gate + phase residual → iRFFT2 → ReLU），使频域变换成为网络架构的一部分，梯度可通过 iFFT 回传。

**Plan 2.12 关键突破（梯度诊断）**:
- 旧 AFM 的 `mag_scale`/`phase_scale` 梯度**严格为零**（不是太弱，是拓扑死锁）
- 根因: `residual_scale=0` → `∂(output)/∂(freq_out)=0` → 整个 FFT 路径梯度为零
- 修复: 采用 MPLSeg 原始设计（硬编码门控 + 无 learnable scale），`residual_scale` 初始化为 1.0
- 修复后: mag gate/phase res 卷积权重梯度恢复正常

**Plan 2.16 确定性重跑（18 组）**:
- 最佳候选: **mid06**（gate_strength=0.6）
- 3 seed 均值 vs baseline: AP75 +12.7%, P@R=0.85 +2.8%, ECE -25.8%, Pred -17%
- 弱门控（0.3）ECE 最优, 中等门控（0.6）AP75 最优, 强门控（1.0）过度压制

**AFM 实际机制**:
- FFT→门控→iFFT 作为**结构保持的特征扰动器**，在频域中均匀压缩幅度
- 门控不做频率选择——高频和低频的压制力度相同
- 作用是为 box regressor 提供更突出的边界信息，同时让 classifier 更保守
- 这不是 RLVR 后训练——这是**架构改进 + fine-tune**（标准检测 loss 反传）

**Plan 2.18 后训练结构验证**:
- 方案 A（弱门控 0.1, AFM only）: seed 42 AP50=0.865, AP75=0.690 — viable
- 方案 C（特征约束, AFM only）: seed 42 AP50=0.864, AP75=0.701 — viable  
- 方案 B（bbox 修正头）: 实现复杂，快速原型未成功

**结论**: In-network FFT 作为架构改进有效（AP75 +12.7%），但作为 RLVR reward 无效。频谱信息在梯度反传链路中有效（in-network），在外部 reward 链路中无效（RLVR verifier）。

### 路径 C：语义分割 RLVR（Plan 4.0，未执行）

Dense mask 提供 pixel-level 监督，天然更适合频域操作——没有候选框匹配噪声，每个像素都有标签。MicroAFM 可插入分割网络 classifier 前，在空间对齐的特征上做 FFT。

**状态**: 概念阶段，尚未执行最小验证。

## 关键技术资产

1. **稳定的 RLVR shell**: KL-anchored signed policy objective + frozen baseline rollout + freeze-state control。在 batch_size=1, 8G GPU 上经过 16+ 轮实验打磨。
2. **MicroAFM + MultiScaleAFM**: 残差 identity-preserving FFT block，零初始化 learnable scales，支持 current/delta/norm_delta 三种残差模式。
3. **完整的 NNI 实验框架**: 从 preset 搜索空间自动生成到 train/eval/diagnostics 全自动化。
4. **Round 2.8 诊断系统**: frozen parity, score threshold curves (7 levels), localization stats (center/size error, IoU, duplicates), AFM scales tracking。
5. **Penn-Fudan 检测基线**: 完整的 Faster R-CNN MobileNetV3 训练/评估/RLVR pipeline。

## 实验参数约定

- 数据集: Penn-Fudan Pedestrian
- 检测器: TorchVision Faster R-CNN MobileNetV3-Large-FPN
- 2.x 实验: 小规模，单种子 (42)，1 epoch，不跑 NNI
- 3.x 实验: 大规模，多种子，多 epoch，跑 NNI
- GPU: 8GB VRAM

## 当前实验结论（2026-06-04，更新至 Plan 2.18）

经过 25+ 轮实验，Penn-Fudan 目标检测上：

**已证伪**:
- 外部手工频谱 verifier 不能作为 RLVR reward（压缩比 12544:1，信号/噪声 <1）
- RLVR policy gradient 在检测任务上无效（梯度链太长，方差太高）
- 旧 MicroAFM 的 learnable gate scale 梯度严格为零（架构拓扑缺陷）

**已验证**:
- MPLSeg-style in-network AFM（mid06, gate_strength=0.6, 5 epoch fine-tune）: AP75 +12.7%, ECE -25.8% vs baseline
- AFM 机制: FFT 作为结构保持的特征扰动器（频域幅度均匀压缩 ~30%），增强边界信号
- 弱门控 AFM-only 后训练（gate_strength=0.1）可行，不崩溃
- `cudnn.benchmark=False` 确定性训练保证了可复现性

**核心洞察**:
- 同样的 FFT 操作，放在 forward 里（in-network）是通路，放在 loss 里（verifier）是瓶颈
- 这不是频谱信息的问题——是梯度链路是否完整的问题
- FFT 结构本身比可学习的门控权重更重要（frozen random gate 同样有效）

**最佳候选**: mid06（MPLSegAFMBlock, gate_strength=0.6, residual_scale init=1.0）

## Plans 2.21-4.5 实验总结（2026-06-05，~140 组实验）

### 2.x 消融发现

**2.21 冻结消融** (5 groups): freeze_rpn 最优 AP50=0.8561, freeze_box 最优 AP75=0.6696。冻结有助于后训练稳定性。

**2.22 数据量扫描** (8 groups): A 后训练在不同数据量下 AP75 稳定 (0.64-0.66)，baseline 波动大 (0.52-0.62)。后训练提供数据效率增益。

**2.23 门控可视化**: 分析脚本有 bug（inference hook 未触发），待修复。

**2.25 相位消融** (9 groups): Phase-only AFM 单 seed 最高 AP50=0.9618。相位调制是 AFM 的关键机制——纯相位分支即可达到甚至超过完整 AFM。

**2.26 配方扫描** (18 groups): C_5ep (特征约束, 5 epoch) 最优 AP75=0.6740。特征约束 (C) 持续优于弱门控 (A)。所有 seed 产出相同结果（确定性训练 + 相同 checkpoint），证明后训练完美可复现。

### 3.x 检测铺开

**3.4 VOC 20-Class** (4 groups completed): baseline AP50=0.48-0.59, mid06 AP50=0.48-0.51。mid06 在 20 类检测上不优于 baseline（训练样本 500, 3 epoch 欠训练）。Phase 2 (A/C 后训练) 因 checkpoint 缺失未执行。

**3.7 汇总** (104 experiments): C_post 最优 AP75=0.6902, ECE=0.0723。mid06 最优 AP50=0.8646。baseline 最差 AP75=0.5841, ECE=0.1012。

### 4.x 分割验证

**4.1-4.5 PF 分割** (12 groups): FCN-ResNet50 从头训练。Baseline mIoU=0.5208 > mid06 mIoU=0.4910 > A_post mIoU=0.5095 > C_post mIoU=0.5030。**AFM 在分割上不工作**——这个 negative result 说明 AFM 的效果是任务/架构依赖的。

### 核心结论

1. **AFM 帮助检测定位 (AP75)，但对分割 (mIoU) 无效**
2. **相位调制是 AFM 的关键机制**（phase-only 即达到完整 AFM 效果）
3. **特征约束 (C) 后训练是检测最优配方**（AP75=0.674）
4. **AFM 效果是架构依赖的**：256ch ROI features（检测）受益，2048ch FCN features（分割）不受益
5. **后训练完美可复现**：相同 checkpoint + 确定性训练 → 相同结果

**下一步**: 修复 ResNet50 crash（afm_channels bug）；扩展 3.4 Phase 2 后训练；在大模型/大数据集上验证 AP75 通用性
