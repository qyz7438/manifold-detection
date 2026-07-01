# FSBG 独立模块验证报告

## 结论（前置）

当前实现的 **Frequency-Spatial Boundary Gate (FSBG)** 在 3-seed Penn-Fudan 3-epoch 验证中，**独立提升不足以支撑“独立模块” claim**：

- 独立使用 FSBG：AP75 = **0.7524**，相比 plain Faster R-CNN（AP75 = 0.7385）仅提升 **+0.0139**。
- 该提升甚至略弱于旧版简单空间 PBG（AP75 = 0.7544，+0.0159）。
- 与 AFM 组合时，FSBG 也不如旧版 PBG（AFM+FSBG 0.7543 vs AFM+PBG 0.7894）。

因此，当前 FSBG 设计需要继续改进，不能直接作为独立创新点提交。

## 实验设置

- 数据集：Penn-Fudan Pedestrian
- 检测器：Faster R-CNN MobileNetV3-Large-FPN
- 训练：3 epoch，box_head_only
- 种子：42 / 123 / 2024
- 指标：val AP75（最终 epoch）

## 结果汇总

### 独立模块（无 AFM）

| 方法 | AP75 均值 | vs indep_none | AP50 均值 | ECE 均值 |
|------|----------|--------------|----------|----------|
| indep_none | 0.7385 | — | 0.9113 | 0.0676 |
| indep_pbg (旧空间门控) | 0.7544 | **+0.0159** | 0.9077 | 0.0743 |
| **indep_fsbg (新)** | **0.7524** | **+0.0139** | 0.9099 | 0.0648 |
| indep_tam | 0.7299 | -0.0086 | 0.9089 | 0.0668 |
| indep_pah | 0.6891 | -0.0494 | 0.8840 | 0.1777 |

### 与 AFM 组合

| 方法 | AP75 均值 | vs redesign_afm | AP50 均值 | ECE 均值 |
|------|----------|----------------|----------|----------|
| redesign_afm (AFM baseline) | 0.7463 | — | 0.9121 | 0.0657 |
| afm_fsbg (AFM + FSBG) | 0.7543 | **+0.0080** | 0.9120 | 0.0687 |
| redesign_afm_pbg (AFM + 旧 PBG) | 0.7894 | **+0.0431** | 0.9115 | 0.0532 |
| redesign_afm_tam | 0.7331 | -0.0132 | 0.9123 | 0.0554 |
| redesign_afm_pah | 0.6939 | -0.0524 | 0.8843 | 0.1653 |

## 关键观察

1. **FSBG 独立增益微弱**：+0.0139 AP75 处于旧 PBG 同一量级，无法与 plain Faster R-CNN 拉开显著差距。
2. **跨种子方差极大**：indep_fsbg 在 seed 123 上 AP75=0.8837，在 seed 2024 上只有 0.6741，均值被高方差支配。
3. **与 AFM 的协同性差**：AFM+FSBG 仅 +0.0080，远低于 AFM+旧 PBG 的 +0.0431。说明 FSBG 的相位-空间融合设计没有继承旧 PBG 的协同优势，反而削弱了它。
4. **PAH 持续损害所有配置**：无论独立还是组合，PAH 都导致 ECE 大幅上升、AP75 下降。
5. **TAM 基本中性**：独立和组合均无稳定增益。

## 失败假设

- 原以为把 AFM 的相位思想合并到空间门控会形成更强的独立边界模块，但实验不支持。
- 可能原因：
  - `alpha=0` 初始化 + 3 epoch 小数据，FSBG 没有足够时间学到有效门控。
  - 相位-only 重建后的 1x1 编码器过于压缩，丢失有用信息。
  - 空间分支和相位分支的 fusion conv 学习困难，导致 gate 质量不高。
  - 频域相位重建在 ROI feature（7x7）上分辨率太低，边界信号被过度平滑。

## 下一步建议

为做出真正可独立提出的模块，建议以下 redesign 方向：

1. **去掉 AFM，单独放大 FSBG**：
   - 增加 FSBG 容量（更深的 spatial/phase encoder、多尺度融合）。
   - 尝试 `alpha_init=1.0` 或 learnable 残差缩放，使其从训练开始就真正参与。
   - 引入显式监督（如 edge/boundary mask）或辅助 loss。

2. **把 FSBG 从“门控”升级为“特征增强器”**：
   - 不只做注意力门控，而是让相位重建分支直接输出增强特征并与原始特征融合。
   - 参考 AFM 的 residual_scale=1.0 设计，确保模块一开始就有影响力。

3. **解决高方差问题**：
   - 增加种子数到 5-10，或做 k-fold CV，确认当前提升不是随机波动。
   - 如果方差持续高，说明模块对初始化/数据划分过于敏感，稳定性不足。

4. **放弃 FSBG，回到旧 PBG 并做深度改进**：
   - 旧 PBG 独立提升 (+0.0159) 和与 AFM 协同 (+0.0431) 都更好。
   - 可以沿着“更强的空间边界先验 + 可学习的残差增强”路线继续升级 PBG，而不是引入显式 FFT。

## 代码状态

- `spectral_detection_posttrain/methods/detection/pbg.py`：FSBG 实现已保留。
- `scripts/round28_train_eval.py`：CLI 标志 `--use-pbg` 仍指向 FSBG。
- 缺失的 `indep_fsbg_s42` 已补跑完成。
