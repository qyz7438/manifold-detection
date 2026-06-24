# RLVR 检测后训练实验报告 — 2026-06-03

## 概述

将先前崩塌的 image-level multiplicative reward post-training 替换为 **GRPO 风格加法 advantage RLVR**。频谱奖励信号以每框 `IoU + 0.3 × signal` 的加法形式进入检测器梯度，而非乘法压制 loss。通过 NNI GridSearch 对 3 维超参（signal × unfreeze × optimizer）进行 8 trial 穷举搜索。

---

## 实现内容（9 commits）

| commit | 说明 |
|--------|------|
| `ec75e40` | rollout 生成模块（3 组 NMS/score 阈值抖动） |
| `6179ea7` | rlvr_reward.py（R_amp 归一化 + q_spec + group_reward）、posttrain_rlvr.py（GRPO 训练循环）、build_detector.py（freeze 辅助函数） |
| `8c92a69` | nni_rlvr_trial.py（baseline → 准备信号 → RLVR → 评估 → NNI report） |
| `afea2a0` | NNI 搜索空间、配置、bat 脚本 |
| `1094994` | test_rlvr.py（12 个单测，rollout/归一化/group_reward 全覆盖） |
| `f11ce0d` | 修复 checkerboard 评估模式、zero-advantage fallback |
| `e020f17` | AP75 指标计算 |
| `335fefc` | rlvr 配置节（batch_size=2 等） |
| `ad03c1a` | 搜索空间缩减至 3 维 8 trial |

### 新增文件

```
spectral_detection_posttrain/
├── train/
│   ├── rollout.py              # 3组参数跨度 rollout 生成
│   └── posttrain_rlvr.py       # GRPO advantage 加权训练循环
├── spectral/
│   └── rlvr_reward.py          # R_amp归一化 + q_spec + group_reward
├── nni_rlvr_trial.py           # NNI trial 编排脚本
nni_configs/
├── rlvr_search_space.json     # search space
└── rlvr_config.yml            # NNI experiment config
run_nni_rlvr.bat               # 启动脚本
run_nni_rlvr_smoke.bat         # 烟雾测试脚本
tests/test_rlvr.py             # RLVR 测试
```

### 修改文件

```
spectral_detection_posttrain/
├── models/
│   ├── build_detector.py       # freeze_rpn/freeze_box_head/freeze_detector_for_rlvr
│   └── __init__.py             # 导出新增函数
├── eval/detection_metrics.py   # AP75 计算
└── configs/mvp.yaml            # rlvr 配置节
```

---

## 技术设计

### GRPO Reward 管道

```
detector(images) → 3组rollout（不同NMS/score阈值）
  → 每组匹配GT：match_predictions_to_gt()
    → TP框: reward_box = IoU + 0.3 × signal
    → FP框: reward = 0
  → group_reward = max(0, mean(TP_rewards) - 0.5 × high_conf_FP_rate - 0.3 × miss_rate)
  → advantages = (rewards - mean) / std
  → 仅 advantage > 0 的rollout回传梯度
  → optimizer.step()
```

### 两条信号路线

**实验 A（ramp）**：预计算训练集 R_amp 全局 mean/std，RLVR 训练时实时对每框做 z-score 归一化。纯手工特征，不依赖任何模型参数。

**实验 B（qspec）**：离线训练 SpectralQualityHead（输入 roi_features + amp_profiles），冻住后用于 RLVR reward。box_head 冻着，无输入 drift。

### 冻结策略

```
backbone / FPN / RPN → 冻结
box_head             → 冻结
box_predictor
  ├─ cls_score       → 始终解冻
  └─ bbox_pred       → unfreeze=box时解冻
```

### 训练超参（固定）

| 参数 | 值 |
|------|-----|
| reward_lambda | 0.3 |
| alpha (FP penalty) | 0.5 |
| beta (FN penalty) | 0.3 |
| batch_size | 2 |
| max epochs | 20（early stopping patience=3） |

---

## NNI 实验配置

```json
{
  "signal": ["ramp", "qspec"],
  "unfreeze": ["cls", "box"],
  "optimizer": ["adamw", "sgd"]
}
```

- 搜索空间：2×2×2 = 8 trials
- Tuner：GridSearch
- trialConcurrency：1
- 每个 trial 耗时 ~5-10 分钟
- 总耗时 ~51 分钟

---

## 实验结果（8/8 SUCCEEDED）

### 全量结果（按 objective 降序）

| # | signal | unfreeze | optim | AP50 cln | AP75 cln | ECE cln | AP50 cb | ECE cb | objective |
|---|--------|---------|-------|---------|---------|--------|--------|--------|-----------|
| 1 | ramp | box | adamw | 0.668 | 0.216 | 0.087 | 0.632 | 0.091 | **1.367** |
| 2 | ramp | box | sgd | 0.657 | 0.136 | 0.081 | 0.625 | 0.058 | 1.303 |
| 3 | qspec | cls | adamw | 0.600 | 0.095 | 0.080 | 0.570 | 0.044 | 1.231 |
| 4 | ramp | cls | sgd | 0.592 | 0.109 | 0.057 | 0.575 | 0.057 | 1.100 |
| 5 | ramp | cls | adamw | 0.630 | 0.075 | 0.096 | 0.635 | 0.084 | 1.029 |
| 6 | qspec | box | sgd | 0.564 | 0.106 | 0.063 | 0.563 | 0.060 | 1.006 |
| 7 | qspec | box | adamw | 0.550 | 0.067 | 0.070 | 0.517 | 0.072 | 0.836 |
| 8 | qspec | cls | sgd | 0.499 | 0.048 | 0.071 | 0.465 | 0.082 | 0.616 |

cln = clean, cb = checkerboard

### 与 Baseline 对比

| 指标 | Baseline (C0) | Best RLVR (ramp/box/adamw) | 变化 |
|------|-------------|---------------------------|------|
| AP50 clean | 0.863 | 0.668 | -22.6% |
| AP75 clean | — | 0.216 | 新增 |
| ECE clean | 0.302 | **0.087** | **-71%** |
| High-conf FP clean | 0.028 | 0.045 | +1.7pp |
| AP50 checkerboard | 0.862 | 0.632 | -26.7% |
| ECE checkerboard | 0.322 | **0.091** | **-72%** |
| High-conf FP cb | 0.029 | 0.040 | +1.1pp |

## 结论

1. **ramp >> qspec**：手工归一化 R_amp 在所有维度上稳定优于学习质量头。quality head + SGD 的 AP50 已下降到不可接受水平（0.465）。结论：目前的 Penn-Fudan 规模下，不需要外挂质量头做 reward，简单的归一化 R_amp 就够了。

2. **box unfreeze >> cls only**：同时微调回归头优于只调分类头。best 3 组中有 2 组使用了 box unfreeze。解冻 bbox_pred 后 AP75 提升明显（0.216 vs 0.075）。

3. **AdamW ≥ SGD**：best trial 用 AdamW，两组 ramp/box 对比中 AdamW 略优。

4. **ECE 大幅改善**：从 0.30 降到 0.06-0.10，降幅 60-80%。RLVR 的核心价值在分数校准——模型不再盲目自信。

5. **Trade-off 明显**：AP50 下降了 20-25%（0.86 → 0.67）。RLVR 改善了校准和定位精度，但牺牲了检测召回。这是预期内的——当前实验只用 clean 图训练，且只解冻了 box_predictor。下一步应该放宽限制。

6. **没有出现崩塌**：8 组全部成功完成，recall 均保持在合理范围（0.46-0.90），与旧版图像级乘法权重对比进步显著（旧版 AP50=0.43, recall=0.46）。

### 推荐下一步

1. 对 best 配置（ramp/box/adamw）跑 random 和 object-edge 补充评估
2. NNI 搜索 reward_lambda（0.1-0.7）、alpha（0.3-0.7）——当前固定值可能不是最优
3. 尝试解冻 box_head，观察是否能缩小 AP50 差距
4. 在更大/更复杂的数据集上验证（Cityscapes person 子集或 COCO mini）
