# Plan 3.2: 阶段性总结与方向决策

> **For agentic workers:** 这是分析文档，不是实现 plan。不需要执行代码。

**Goal:** 系统评估 18+ 轮实验的全部证据，判断当前技术路线的可行性和下一步方向。

---

## 一、18 轮实验全景

### 阶段 A：外部 Verifier RLVR（Round 1 — 2.5）

目标：用 FFT 在 ROI crop 上计算 R_amp 作为标量 reward，驱动 RLVR 后训练。

| Round | 方法 | 关键结果 |
|-------|------|---------|
| 1 | softmax(adv) × model(images,targets) | reward 不参与梯度（所有 rollout 相同 loss） |
| 2 | L_det + γ × candidate_quality_loss | AP75 崩塌（0.68→0.23），post-NMS boxes 训练 bbox_head |
| 2.1 | cls-only weighted CE, box_loss=0 | AP50 崩塌到 0.62，precision 0.21 |
| 2.2 | KL-stabilized signed policy + frozen baseline | AP75 崩塌 0.23，precision 0.39 |
| 2.3 | KL=10, policy=0.0003, freeze-state, signed | **首个稳定配方**：AP50≈0.873，预测数 120-125，6/6 通过约束 |
| 2.5 | KL=10 + amp/structure/shuffled | **全部 8 组 AP50 0.884-0.886，差距<0.001**——信号完全中性 |

**阶段 A 结论**：手工频谱 verifier 在 Penn-Fudan 检测上没有因果信号。KL=10 + policy=0.0003 + signed objective + frozen baseline 是稳定的 RLVR shell，但 reward 信号太弱，无法区分 real vs shuffled。提高 policy_weight → 崩塌，降低 → 中性。

### 阶段 B：In-Network FFT（Round 2.6 — 2.7）

目标：受 MPLSeg 启发，将 FFT 从外部 verifier 移到网络内部 feature transform。

| Round | 方法 | AP50 delta | AP75 delta | precision delta |
|-------|------|-----------|-----------|----------------|
| 2.6 | AFM, sigmoid gate, zero init | -2.7% (0.853) | -36.6% (0.409) | -34% (0.439) |
| 3.1 | 同架构，cold 3-epoch | +2.0% (0.862) | +7.7% (0.613) | -13% (0.615) |
| 2.7 | **残差 identity, tanh, scale=0** | **-0.1% (0.876)** | **-17.7% (0.537)** | **-22% (0.523)** |

多尺度 FPN 注入：AP50 崩塌到 0.04（RPN 完全破坏）。hot-start：比 cold-start 更差。

**阶段 B 结论**：
- 残差 identity 设计是正确的——AP50 打平 baseline
- 但 AP75 和 precision 的持续退化表明：**FFT/iFFT 变换在预训练 2-stage detector 的 ROI 特征空间中无法在 1 epoch 内恢复到 baseline 品质**
- 3-epoch cold training 的 AP50 提升（+2.0%）可能来自额外 epoch 效应而非 AFM 贡献
- FPN 级 multi-scale 注入对 2-stage detector 致命

---

## 二、根本矛盾

```
R_amp 的 TP/FP gap = 0.008（余弦相似度 + exp 双重压缩）
policy_weight = 0.0003 → signal × weight = 2.4×10⁻⁶
KL = 10 → 任何偏离 baseline 的尝试立即被拉回

两股力量互相抵消：
- 提高 policy_weight → KL 拉不住 → 崩塌
- 降低 policy_weight → signal 被 KL 淹没 → 神经网络无变化
```

AFM 路线绕过这个矛盾——FFT 不是 reward 信号，而是架构组件。但它引入了**预训练模型的统计量错配问题**：ROI head 期望特定的特征分布，FFT→iFFT 改变了这个分布，即使 identity init 也不会自动恢复。

---

## 三、三条可选的下一步

### Option 1：写论文，不扩展实验

**适用场景**：目标是产出可发表的学术成果。

**论文框架**：
- 贡献 1：稳定的 RLVR detection shell（KL-anchored signed policy + frozen baseline）
- 贡献 2：证明手工频谱 verifier 在 Penn-Fudan 检测上没有 causal signal（系统性的 negative result）
- 贡献 3：将 MPLSeg 启发的 in-network FFT 方法迁移到检测，验证其可行性但指出架构限制

**优势**：现有数据已足够支撑一篇 paper。不需要额外 GPU 时间。
**劣势**：只跑了一个数据集（Penn-Fudan），审稿人可能质疑 generalize 能力。

### Option 2：换单阶段检测器重跑 AFM

**方向**：RetinaNet/FCOS + AFM on backbone FPN → single prediction head。没有 ROI 结构干扰，FFT 作用在 backbone feature 而非 ROI crop。

**估算**：2-3 周实现 + 实验。风险中等——如果单阶段检测器和 2 阶段一样有"预训练统计量错配"，结论不变。

### Option 3：切回分割，从 scratch 训 MPLSeg-style

**方向**：如 Plan 4.0 所述，但缩小范围——不建完整 package。在 Penn-Fudan masks 上，TorchVision FCN/DeepLabV3 + MicroAFM on classifier head，从随机初始化训 N epoch。

**优势**：和 MPLSeg 最接近的验证方式。Dense pixel labels 天然更适合频域操作。
**劣势**：完全离开了检测的主战场。之前 18 轮积累的检测实验经验无法直接复用。

---

## 四、推荐：Option 1 + 最小化 Option 2

不是"二选一"，而是**先写论文（1-2 周），写完再看 2/3**。

理由：
1. 现有 18 轮数据已经足够形成一个完整的研究叙事
2. 继续跑实验而不同时写 paper，会陷入"永远差一组实验"的循环
3. 写 paper 过程中会暴露实验缺口（比如缺少 VOC/COCO 对比），这些才是真正需要补的实验
4. 如果审稿人要求"换数据集验证"，再跑 Option 2 的单阶段检测器——只需一个数据集一个实验

**具体执行**：
- Week 1：写 Introduction + Related Work + Method（RLVR shell + spectral verifier + AFM）
- Week 2：写 Results + Analysis（基于现有 18 轮数据整理 table）
- Week 2 末尾：识别缺口，如果 < 3 天能补就跑，> 3 天 mark 为 future work

---

## Plan 位置

`E:/CLIproject/RLimage/docs/superpowers/plans/2026-06-04-plan32-analysis-and-direction.md`
