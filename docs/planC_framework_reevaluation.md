# Plan C 论文框架重新评估

## 新增消融实验（执行于本回合）

### A. Adapter vs backbone-only / full fine-tune（从 3-epoch 弱基线出发）

| Run | 配置 | best mIoU | final mIoU | vs baseline |
| --- | --- | --- | --- | --- |
| `planC_abl_backbone_only_3ep` | `action=none`, unfreeze backbone, **freeze classifier head**, 3ep, lr=1e-4 | **0.5985** | 0.5985 | **+4.69 pp** |
| `planC_abl_full_finetune_3ep` | `action=none`, unfreeze **全部**参数，3ep, lr=1e-4 | **0.6246** | 0.6246 | **+7.30 pp** |
| `planC_abl_feature_adapter_e2e_3ep` | `action=feature`, unfreeze backbone, freeze classifier head, 3ep, lr=1e-4 | **0.5877** | 0.5633 | +3.61 pp (best), 不稳定 |

### B. 完整训练 baseline（20 epoch from scratch）

| Run | 配置 | best mIoU | final mIoU |
| --- | --- | --- | --- |
| `seg_baseline_from_scratch_20ep` | FCN-ResNet50, AdamW lr=1e-4, 20 epoch | **0.6278** (epoch 19) | 0.6225 (epoch 20) |

关键对照：
- 3-epoch baseline mIoU = 0.5516
- 6-epoch from-scratch mIoU = 0.5973
- 3-epoch baseline + HFBA 3ep mIoU = 0.5985
- 3-epoch baseline + full fine-tune 3ep mIoU = 0.6246

## 重新评估后的关键结论

1. **原 3-epoch baseline 未收敛**
   - 完整训练 20 epoch 后 from-scratch baseline 达到 0.6278。
   - 第 6 epoch 就已达到 0.5973，与 3-epoch baseline + HFBA 的 0.5985 基本持平。
   - 因此 **HFBA 不是“post-training 提升”，只是“继续训练到第 6 epoch”**。

2. **Adapter 本身没有独立价值**
   - 在 backbone 真正冻结时，logit/feature adapter 几乎无效。
   - 在 backbone 解冻、分类头冻结时，backbone-only fine-tune（0.5985）优于 logit adapter（0.5772）和 feature adapter（0.5877 best / 0.5633 final）。

3. **有效提升的上限来自完整 fine-tune**
   - 解冻全部参数 3 epoch 达到 0.6246，已接近完整训练 20 epoch 的 0.6278。

4. **BatchNorm drift 仍是核心混淆**
   - 旧版 feature adapter 的 +3.6 pp 是 BN drift 造成的；
   - 新版 feature adapter 在受控 end-to-end 下 best 0.5877，但 final 0.5633，说明它不仅没带来稳定收益，反而让训练更不稳定。

## 对论文框架的冲击升级

原框架“冻结 backbone + 轻量 adapter 有效”已被证伪，升级到：

> **在 from-scratch 弱基线上，任何 post-training 收益都主要来自 backbone 的继续训练，adapter 动作空间本身几乎无贡献。**

因此论文不能再以 ALAPT（adapter-centric）作为核心卖点。

## 可选方向再评估

| 方向 | 内容 | 优点 | 缺点 | 数据支持 |
| --- | --- | --- | --- | --- |
| **A. 受控实证研究 + 方法论警示（推荐）** | 提出严格冻结 backbone 协议，揭示 BN drift，比较 adapter vs backbone-only vs full fine-tune | 数据充分、方法严谨、对社区有警示价值 | 不是传统“新方法 SOTA”论文；故事偏 negative | 强（消融 + 完整 baseline） |
| **B. Head-Frozen Backbone Fine-Tuning** | 把方法定义为“冻结分类头、微调 backbone” | 有明确可复现收益 | 被完整训练 baseline 击败（20ep 0.6278 > HFBA 0.5985），创新性一般 | 已被证伪 |
| **C. 继续抢救 adapter** | 尝试 ImageNet 预训练 backbone、更大 scale、更深 adapter、多数据集 | 若成功可回归原叙事 | 时间和计算成本高，成功率不确定 | 目前无支持 |

## 推荐方案

**方向 A（受控实证研究 / 方法论论文）是唯一还能立住的方向**：

- 论文标题/核心：*Revisiting Lightweight Post-Training for Semantic Segmentation: Adapter, Backbone, or BatchNorm Drift?*
- 重点不再推销某个“新方法”，而是：
  1. 指出冻结 backbone adapter 实验中 BN drift 的普遍存在与危害；
  2. 提出严格的评估协议（强制 `eval()`、记录 BN 统计量变化、区分 backbone 更新模式）；
  3. 在 from-scratch FCN-ResNet50 上给出对照数据，说明 adapter 的增益会被 backbone 继续训练解释；
  4. 呼吁领域在报告轻量后训练结果时控制 BN drift 和 baseline 训练程度。

**不再推荐 HFBA 作为独立方法**：它与“多训几个 epoch”无法区分。

## 下一步需要用户决策

1. 是否接受方向 A（方法论警示论文）？
2. 是否要继续跑 ImageNet 预训练 backbone 来验证 adapter-only 在更强基线上是否有效（方向 C）？
3. 是否要把检测侧（Faster R-CNN + AFM）的有效结果重新纳入，形成“检测 vs 分割”对比叙事？
4. 是否直接放弃 Plan C 分割论文，把资源转回检测侧？
