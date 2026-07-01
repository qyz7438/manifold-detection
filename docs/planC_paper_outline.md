# Plan C: Revisiting Lightweight Post-Training for Semantic Segmentation
## Adapter, Backbone, or BatchNorm Drift?

## 1. Abstract (English, ready-to-use)

> Post-training is a practical way to adapt a deployed semantic segmentation model to new conditions without retraining from scratch. Existing lightweight approaches often confine the update to output logits or small adapters under a "frozen backbone" assumption that we find can be confounded by BatchNorm drift. In this work, we revisit dense-prediction post-training with a strict evaluation protocol. We systematically study three action spaces---logit correction, feature residual refinement, and output calibration---and show that, when the backbone is truly frozen, all three adapters yield negligible gains on a from-scratch FCN-ResNet50 baseline. The gains previously attributed to adapters instead come from continued backbone training, and adding an adapter on top of a fine-tuned backbone does not improve over backbone-only fine-tuning. On the Penn-Fudan pedestrian segmentation dataset, a fully trained from-scratch baseline reaches 62.8\% mIoU, which exceeds a frozen-head backbone-only fine-tune (59.9\%) and all adapter-assisted variants (57.7\%). Our findings caution that BatchNorm drift and an under-trained baseline can masquerade as adapter effectiveness, and that action-space design must be evaluated together with backbone-update constraints.

---

## 2. Abstract (中文对照)

> 后训练是一种实用的模型适配方式，可以在不完全重训整个语义分割网络的情况下提升模型表现。现有轻量方法通常只在“冻结主干”的假设下更新输出 logits 或小型 adapter，而这一假设容易受到 BatchNorm 统计量漂移的干扰。本文用严格的评估协议重新审视密集预测后训练，系统比较 logit 修正、特征残差精炼和输出校准三种动作空间。我们发现，在 backbone 真正冻结时，三种 adapter 在 from-scratch FCN-ResNet50 基线上都几乎无效；先前归因于 adapter 的提升实际上来自 backbone 的继续训练，而在微调 backbone 的基础上再叠加 adapter 并不能超过单纯微调 backbone。在 Penn-Fudan 行人分割数据集上，完整训练的 from-scratch 基线即可达到 62.8% mIoU，超过冻结分类头 + 仅微调 backbone（59.9%）和所有 adapter 辅助变体（57.7%）。实验表明，密集预测后训练中的动作空间设计必须与 backbone 更新约束和基线训练程度一起评估，否则 BatchNorm 漂移和未收敛基线会制造虚假的 adapter 有效性。

---

## 3. Introduction 大纲

### 3.1 Motivation
- 语义分割模型部署后常需适配新场景、新数据或修正错误模式；
- 全网络微调计算成本高、部署流程重、易引起灾难性遗忘；
- 轻量后训练（只更新少量参数）成为实际刚需。

### 3.2 Limitations of Current Lightweight Post-Training
- 现有主流轻量方法多聚焦在：
  - output/logit calibration（温度、bias）；
  - prompt / bias tuning；
  - adapter 插入但多作用于 logits 或浅层。
- 问题：logit-space 动作容量有限，难以修正深层特征错误；feature-space 后训练在分割中尚未被系统研究。

### 3.3 Evaluation Protocol
- 把后训练评估形式化为 **“动作空间选择 + backbone 更新约束 + 基线训练程度”** 三个维度：
  - **Logit correction**：adapter 输出逐像素 logit 偏移；
  - **Feature residual refinement**：adapter 输出 penultimate feature 的残差修正；
  - **Calibration**：adapter 输出每类温度与 bias；
  - **Backbone update modes**：frozen（强制 eval）、head-frozen fine-tune、full fine-tune。
- 严格冻结 backbone 时强制 `eval()` 模式，避免 BatchNorm 统计量漂移；
- 同时报告 from-scratch baseline 在不同训练 epoch 下的表现，以区分“post-training 提升”与“基线未收敛”。

### 3.4 Key Findings
- **原 3-epoch baseline 未收敛**：完整训练 20 epoch 后 from-scratch FCN-ResNet50 达到 62.8% mIoU；第 6 epoch 的 59.7% 已与 3-epoch baseline + head-frozen fine-tune（59.9%）持平；
- **BatchNorm drift 是冻结 backbone 实验中的主要混淆因素**：旧 trainer 未强制 `base_model.eval()`，导致冻结 backbone 的 adapter 实验实际上在微调 BN 统计量，产生虚假提升；
- **真冻结 backbone 的 adapter-only post-training 几乎无效**：logit adapter 仅 +0.36 pp，feature adapter 与 baseline 持平；
- **adapter 在解冻 backbone 时也没有独立价值**：在分类头冻结、backbone 微调的设置下，backbone-only fine-tune（59.9%）优于 logit adapter（57.7%）和 feature adapter（57.8% best / 56.3% final）；
- **完整 fine-tune 接近完整训练 baseline**：3-epoch baseline + full fine-tune 3ep 达到 62.5%，与 20-epoch from-scratch 的 62.8% 相当；
- **Supervised CE 已足够强**：在当前设置下，RLVR/SPSA/DPO 并未超越直接 supervised objective；
- **动作空间的价值取决于 backbone 更新约束与基线训练程度**：脱离二者谈“logit vs feature adapter”会得出误导性结论。

### 3.5 Contributions
1. 提出一个受控的语义分割后训练评估协议，显式区分“动作空间选择”、“backbone 更新约束”与“基线训练程度”，并强制冻结 backbone 时保持 `eval()` 模式以避免 BatchNorm drift；
2. 揭示 BatchNorm drift 和未收敛基线会让“冻结 backbone + adapter”的实验产生虚假增益，并在严格协议下证明 logit/feature/calibration adapter alone 在此基线上几乎无效；
3. 提供系统消融数据，说明在 Penn-Fudan from-scratch FCN-ResNet50 上，任何 post-training 提升都可被 backbone 继续训练解释，呼吁领域在报告轻量后训练结果时控制 BN drift 和基线训练程度。

### 3.6 Paper Organization
- Sec. 2: Related Work（segmentation adapters, parameter-efficient fine-tuning, post-training）；
- Sec. 3: Method（ALAPT framework, action spaces, training objective）；
- Sec. 4: Experiments（dataset, baseline, main results, ablations）；
- Sec. 5: Discussion and Limitations；
- Sec. 6: Conclusion.

---

## 4. 与方法论述保持一致的关键措辞

- **不强调 RLVR / policy gradient / spectral verifier**：这些作为探索过但未被数据支持的路线，可放在 Related Work 或 Appendix 中简要说明；
- **核心卖点是严格的冻结 backbone 协议 + BN drift 警示 + 基线收敛性检查**，而不是“新方法”或“极少参数 adapter”；
- **基线设置透明**：明确说明 FCN-ResNet50 from-scratch、3/20 epoch、无 ImageNet 预训练，强调基线训练程度对结论的决定性影响；
- **避免使用“SOTA”和“proposed method”**：本文定位是方法论警示与受控实证研究；
- **明确区分四种协议**：frozen-backbone adapter、head-frozen backbone-only fine-tune、head-frozen adapter-assisted fine-tune、full fine-tune；
- **不再把任何提升归因于 adapter**：所有显著提升均可被 backbone 继续训练解释。
