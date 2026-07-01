# RLIimage 当前状态完整报告

**生成时间**：2026-06-22  
**分支**：`manifold-main`  
**工作目录**：`E:/CLIproject/RLimage`

---

## 1. 项目背景与目标

本项目旨在构建面向目标检测的 **RLVR（Reinforcement Learning with Verifiable Rewards）后训练框架**，并探索将频域证据（spectral evidence）引入检测器。经过 25+ 轮实验后，核心结论已发生转向：

- 外部频谱 verifier 作为 RLVR reward 已证伪（信号太弱）。
- In-network FFT（AFM 模块）作为架构改进在检测定位任务上有效（AP75 +12.7%）。
- 但 AFM 本质上是**架构改进 + fine-tune**，不是 RLVR 后训练。

因此当前阶段的目标调整为：**提出一个可独立工作的检测模块**，而不是继续把结果与 AFM/MicroAFM 等已有方法组合。

---

## 2. 本轮 redesign 工作（Round 28）

为响应“独立模块”的要求，对检测器结构进行了 redesign，引入了三个候选模块：

### 2.1 Frequency-Spatial Boundary Gate (FSBG)

- **前身**：Phase Boundary Gate (PBG)，一个简单的空间边缘注意力门控。
- **升级思路**：把 AFM 的“相位-only FFT 重建”思想与 PBG 的空间边缘门控合并，做成“频域-空间域联合边界门控”。
- **实现位置**：`spectral_detection_posttrain/methods/detection/pbg.py`
- **结构**：
  - 空间分支：Conv3x3 → ReLU → Conv3x3，输出 1 通道边缘图。
  - 相位分支：rFFT2 → 幅度置 1 → iRFFT2 → ReLU → 1x1 Conv 编码，输出 1 通道相位边界图。
  - 融合分支：将两路边界图 concat → Conv3x3 → Sigmoid 得到空间注意力门控。
  - 残差输出：`x + alpha * gate * x`，`alpha` 初始化为 0（identity start）。
- **设计意图**：既有独立创新点（频域-空间域联合），又能继承 AFM 的相位增强优势。

### 2.2 Task-Aligned Manifold (TAM)

- **实现位置**：`spectral_detection_posttrain/methods/detection/tam.py`
- **结构**：在 box_head 后的特征上接一个 encoder-decoder 残差流形。
- **设计意图**：学习任务对齐的低维流形，增强 box/regression 特征。

### 2.3 Prototype-Aware Head (PAH)

- **实现位置**：`spectral_detection_posttrain/methods/detection/pah.py`
- **结构**：用可学习 prototype 替代标准分类头，L2 归一化 + temperature 内积做分类；同时保留 bbox 回归分支。
- **设计意图**：让分类基于特征与原型相似度，提升校准和泛化。

### 2.4 训练入口

- `scripts/round28_train_eval.py` 增加了 `--use-pbg`、`--use-tam`、`--use-pah` 等 CLI 标志。
- 模块插入顺序：**PBG/FSBG → AFM → box_head → TAM**。

---

## 3. 实验结果

### 3.1 实验设置

- 数据集：Penn-Fudan Pedestrian
- 检测器：Faster R-CNN MobileNetV3-Large-FPN
- 训练：3 epoch，`box_head_only` 可训练
- 种子：42 / 123 / 2024（3 seed 平均）
- 核心指标：val AP75、AP50、ECE

### 3.2 第一轮 redesign grid（7 配置 × 3 seed）

| 配置 | AP75 | AP50 | ECE |
|------|------|------|-----|
| AFM baseline | 0.7463 | 0.9121 | 0.0657 |
| AFM + PBG (旧) | **0.7894** | 0.9115 | 0.0532 |
| AFM + TAM | 0.7331 | 0.9123 | 0.0554 |
| AFM + PAH | 0.6939 | 0.8843 | 0.1653 |
| AFM + PBG + TAM | 0.7638 | 0.9147 | 0.0610 |
| AFM + PBG + PAH | 0.6952 | 0.8841 | 0.1736 |
| AFM + PBG + TAM + PAH | 0.6926 | 0.8865 | 0.1602 |

**结论**：旧 PBG 与 AFM 协同最好（+0.0431 AP75），TAM 中性，PAH 显著损害所有指标。

### 3.3 独立模块验证（无 AFM）

| 配置 | AP75 均值 | vs Plain | AP50 均值 | ECE 均值 |
|------|----------|---------|----------|----------|
| Plain Faster R-CNN | 0.7385 | — | 0.9113 | 0.0676 |
| 旧 PBG | 0.7544 | +0.0159 | 0.9077 | 0.0743 |
| **新 FSBG** | **0.7524** | **+0.0139** | 0.9099 | 0.0648 |
| TAM | 0.7299 | -0.0086 | 0.9089 | 0.0668 |
| PAH | 0.6891 | -0.0494 | 0.8840 | 0.1777 |
| PBG + TAM + PAH | 0.6926 | -0.0459 | 0.8832 | 0.1611 |

### 3.4 与 AFM 组合验证

| 配置 | AP75 均值 | vs AFM | AP50 均值 | ECE 均值 |
|------|----------|--------|----------|----------|
| AFM baseline | 0.7463 | — | 0.9121 | 0.0657 |
| AFM + FSBG | 0.7543 | +0.0080 | 0.9120 | 0.0687 |
| AFM + 旧 PBG | 0.7894 | +0.0431 | 0.9115 | 0.0532 |
| AFM + TAM | 0.7331 | -0.0132 | 0.9123 | 0.0554 |
| AFM + PAH | 0.6939 | -0.0524 | 0.8843 | 0.1653 |

### 3.5 逐 seed 详细数据

**独立模块（无 AFM）**

| 配置 | s42 AP75 | s123 AP75 | s2024 AP75 | 均值 |
|------|---------|----------|-----------|------|
| indep_none | 0.6683 | 0.8482 | 0.6990 | 0.7385 |
| indep_pbg | 0.7285 | 0.8492 | 0.6856 | 0.7544 |
| indep_fsbg | 0.6993 | 0.8837 | 0.6741 | 0.7524 |
| indep_tam | 0.6706 | 0.8291 | 0.6900 | 0.7299 |
| indep_pah | 0.6832 | 0.8075 | 0.5765 | 0.6891 |

**AFM 组合**

| 配置 | s42 AP75 | s123 AP75 | s2024 AP75 | 均值 |
|------|---------|----------|-----------|------|
| redesign_afm | 0.6749 | 0.8411 | 0.7229 | 0.7463 |
| afm_fsbg | 0.7033 | 0.8431 | 0.7165 | 0.7543 |
| redesign_afm_pbg | 0.7090 | 0.8947 | 0.7644 | 0.7894 |
| redesign_afm_tam | 0.6729 | 0.8089 | 0.7176 | 0.7331 |
| redesign_afm_pah | 0.6638 | 0.8149 | 0.6031 | 0.6939 |

---

## 4. 关键发现

### 4.1 FSBG 不符合独立模块标准

- 独立提升仅 **+0.0139 AP75**，与旧 PBG（+0.0159）处于同一微弱量级。
- 与 AFM 组合时，FSBG 的表现（+0.0080）远弱于旧 PBG（+0.0431）。
- 跨 seed 方差极大：s123 0.8837 vs s2024 0.6741，稳定性不足。

**结论**：当前 FSBG 设计不能支撑“独立模块”的论文 claim，需要继续改进或换方向。

### 4.2 旧 PBG 仍是当前最强候选

- 独立提升 +0.0159，AFM 协同 +0.0431。
- 说明简单的空间边界门控已经有效，问题不在于“是否需要边界信息”，而在于如何把它升级为一个有独立创新性的模块。

### 4.3 PAH 持续失败

- 无论独立还是组合，PAH 都导致 AP75 下降、ECE 大幅上升（>0.14）。
- 原因可能是：prototype-based 分类在 3 epoch / 小数据上难以收敛；temperature 设计或 prototype 初始化不当；或者 ROI 特征不适合直接做原型分类。

### 4.4 TAM 基本中性

- 独立和组合均无明显增益，既不帮助也不严重损害。

### 4.5 高方差是隐忧

- 同配置不同 seed 的 AP75 差异可达 0.2（如 indep_fsbg s123 vs s2024）。
- 这意味着 3 seed 均值可能不够可靠，需要更多 seed 或交叉验证来确认结论。

---

## 5. 失败分析：为什么 FSBG 不够强？

1. **alpha=0 初始化 + 3 epoch 小数据**
   - FSBG 从 identity 开始，模块需要从零学习门控。在小数据集上 3 epoch 可能不足以让相位-空间融合分支收敛到有效状态。

2. **相位分支信息压缩过度**
   - 相位-only 重建后通过两层 1x1 conv 压缩到 1 通道，可能丢失了太多有用的频域结构信息。

3. **ROI feature 分辨率太低**
   - ROI 特征是 7×7，频域相位重建在如此低分辨率上能提取的边界信号非常有限，容易被噪声淹没。

4. **融合门控学习困难**
   - 两路 1 通道边界图直接 concat 后做 3×3 conv → sigmoid，结构简单但可能没有足够容量学习“何时信任空间、何时信任相位”。

5. **残差形式保守**
   - `x + alpha * gate * x` 的形式限制了模块的表达能力。即使 gate 学到了有用信息，alpha 太小也会让它失效。

---

## 6. 代码资产清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `spectral_detection_posttrain/methods/detection/pbg.py` | 已更新 | FSBG 实现，保留旧 PBG 别名 |
| `spectral_detection_posttrain/methods/detection/tam.py` | 已新增 | Task-Aligned Manifold |
| `spectral_detection_posttrain/methods/detection/pah.py` | 已新增 | Prototype-Aware Head |
| `spectral_detection_posttrain/core/models/build_detector.py` | 已更新 | 支持 PBG/AFM/TAM/PAH 组合插入 |
| `scripts/round28_train_eval.py` | 已更新 | 新增 CLI 标志 |
| `tests/methods/test_pbg.py` | 已更新 | FSBG 基础测试 |
| `legacy/` | 已归档 | 旧 RLVR/DPO/BEM/分割/分类等代码 |
| `docs/fsbg_independent_validation_report.md` | 已新增 | FSBG 独立验证专项报告 |

---

## 7. 下一步可选方案

### 方案 A：深度改进 FSBG（推荐尝试）

目标：让 FSBG 独立提升达到 **+0.03~0.05 AP75** 以上。

具体动作：
- 增大 FSBG 容量：spatial/phase encoder 加深，引入多尺度融合。
- 改 alpha 初始化：尝试 `alpha_init=1.0` 或 learnable 缩放。
- 让相位分支直接输出增强特征，而非只做门控。
- 加入 batchnorm 或 layer norm 稳定训练。
- 跑 5-10 seed 确认稳定性。

### 方案 B：回到旧 PBG 并升级为独立模块

目标：在旧 PBG 的基础上做一个更强的边界增强模块。

具体动作：
- 把 PBG 从“注意力门控”升级为“可学习残差边界增强器”。
- 引入多尺度边缘检测（Sobel、Laplacian、高斯差分）+ 可学习融合。
- 加入边界置信度或定位质量估计。
- 这样既有独立提升，也保留了与 AFM 的协同潜力。

### 方案 C：放弃边界门控，换全新方向

如果边界信息本身天花板低，可考虑：
- 基于 IoU/定位质量的显式 head（类似 IoU-aware）。
- 基于 contrastive learning 的检测特征增强。
- 基于 NMS-aware 或重复框抑制的模块。

### 方案 D：先解决高方差问题

- 跑更多 seed（5-10）。
- 或者做 k-fold / leave-one-out 交叉验证。
- 如果所有模块在更稳定评估下仍无提升，说明问题不在模块设计，而在 baseline 已经很强 / 数据集太小。

---

## 8. 当前阻塞点

- **FSBG 独立提升不足**，无法进入论文写作阶段。
- **PAH/TAM 无价值**，可以排除。
- **高方差** 让 3-seed 结论可信度降低。

---

## 9. 建议的下一步

最紧迫的问题是：**FSBG 还是旧 PBG 路线？**

如果坚持“频域-空间域联合”这个创新点，需要投入 1-2 轮 redesign 让 FSBG 独立提升翻倍；如果接受“空间边界增强”作为核心，旧 PBG 升级路线更稳妥。

建议下一步：
1. 选定一条路线（FSBG 改进 vs PBG 升级）。
2. 设计 2-3 个变体。
3. 每个变体跑 3 seed 快速验证。
4. 选择独立提升最大且方差最小的变体深入。
