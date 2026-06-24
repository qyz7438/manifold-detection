# Plan 3.6: 真正 RLVR 后训练（3 方向）

> **核心转变：** loss reweighting → policy gradient。RLVR 的定义——模型采样 → 频谱 reward 打分 → policy gradient 更新。

---

## 方向 1：RPN 采样 + REINFORCE

**目标：** 用频谱 reward 教会 RPN 在频域"热闹"的区域多提 proposal。

**改动：** 不改模型结构。RPN 按 objectness 概率分布采样 proposal（非确定性的 top-K），频谱 quality 打分，REINFORCE 更新 RPN。

```
设计:
  每个 FPN level
    RPN 输出 objectness logits (H, W, A)
    → softmax → 概率分布
    → 按概率采样 K 个位置（探索模式，不取 top-K）
    → 每个采样位置生成 proposal box
    → ROI Align → 频谱 quality(reward)
    → REINFORCE: ∇J = reward × ∇log P(采样位置)
    → RPN 参数更新
```

**冻结策略：** backbone + box_head 冻结，只更新 RPN。

**超参：** K（每层采样数），reward 基线（baseline subtraction 降方差）

**验证方式：** 从 V1 收敛 checkpoint 出发，5 epoch RL，对比冻结基线 AP。

**推理难度：** 中。torchvision RPN 内部是确定性 top-K，需要替换为随机采样。

---

## 方向 2：BBox 探索 + 频谱 Reward

**目标：** 在方向 1 追加——RPN 给好 proposal 后，box_head 用频谱 reward 学会调得更准。

**改动：** box_head 输出 bbox delta 的均值 μ 和对数方差 logσ²（Gaussian policy）。每个 proposal 采样 M 个候选 delta，频谱 quality 打分，REINFORCE 更新 box_head。

```
设计:
  对每个正样本 proposal:
    RPN 给出 proposal box b₀
    box_head → (μ_x, μ_y, μ_w, μ_h, logσ²)   # mean + variance
    采样 M 个 delta ~ N(μ, σ²)                # 探索
    → M 个候选框 b₁...bₘ
    → 每个候选框 ROI → 频谱 quality → reward
    → REINFORCE: ∇J = mean(r × ∇log P(delta|μ,σ))
    → box_head 参数更新
```

**冻结策略：** backbone + RPN 冻结，只更新 box_head + box_predictor。

**超参：** M（每 proposal 采样数），σ 初始化

**推理难度：** 高。torchvision box_head 是确定性 MLP，需替换输出层为 parameterized distribution。

---

## 方向 3：端到端 RLVR

**目标：** RPN 采样 + bbox 探索 + 多组件频谱 reward，全模型 RL 更新。

**改动：** 方向 1+2 的联合，加 PPO 稳定训练（clip ratio + advantage normalization）。

```
设计:
  单张图 forward:
    1. backbone → FPN features（冻结）
    2. RPN 采样 proposal 集合 P（方向 1）
    3. 每个 proposal p ∈ P:
         box_head → Gaussian policy → 采样 delta → 候选框 b'
         候选框 b' → ROI FFT → spectrum_quality(b')
    4. 每个 proposal 的 reward:
         r = 0.3×HF_energy(b') + 0.4×(1-entropy(b')) + 0.3×phase_coherence(b')
         + 0.1×IoU(b', GT)（可选，有 GT 时）
    5. PPO update:
         旧 policy 和新 policy 输出概率比 → clip → advantage 加权
         联合更新 RPN + box_head

  advantage = r - V(state)（学一个 value 函数做 baseline）
```

**冻结策略：** 只冻结 backbone。RPN + box_head + box_predictor + value_head 都训练。

**超参：** clip ϵ=0.2, γ=0.99, value_loss_coef=0.5

**推理难度：** 最高。需要实现 PPO buffer、advantage 计算、多 epoch 训练。

---

## 执行顺序与预期

| 方向 | 难度 | 先决条件 | 预期效果 |
|------|------|---------|---------|
| 1 RPN REINFORCE | 中 | 无 | AP75 提升（RPN学会看频域活跃区域） |
| 2 +BBox 探索 | 高 | 方向1成功 | AP75 进一步提升（bbox调节更精准） |
| 3 端到端 PPO | 最高 | 方向1+2成功 | 最佳效果，但可能不稳定 |

**从方向 1 开始**——改动最小，最接近已验证的信号（频谱 quality 有效），先确认 RLVR 在检测上是否通则。
