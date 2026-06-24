# Round 2.1: Stability-first RLVR

## 背景

Round 2 的 8 个 trial 全因 `ap50_clean` 约束失败（best=0.673，要求>=0.810）。AP75 从 0.679 崩塌到 0.234，high-conf FP 反而从 2 增至 5-19。五个实现 bug 导致结果不可用。

---

## Round 2 已确认的 Bug + 修复

### 1. box regression 用 post-NMS final boxes 训练 bbox_head :: train/inference 分布错位

fix:

```python
if box_loss_weight <= 0:
    regression_targets = None  # 完全跳过 box_coder.encode
    loss_policy_box = 0.0       # 不构造 bbox loss
else:
    regression_targets = per_image_encode(...)
```

不是把 loss 乘 0，而是**彻底不构造 bbox target**，避免坐标分布错位编码继续进入计算图。

### 2. shuffled_ramp 未被实际调用 :: 频域因果对照无效

fix: 在 `build_rewarded_roi_actions()` 内部，matched 已确定后 shuffle:

```python
if cfg.signal == "shuffled_ramp":
    amp = shuffle_tp_ramp(amp, matched)
amp = amp * matched.float()  # FP 频域项=0
```

### 3. temperature 未传入 DetectionVerifierConfig :: 所有温度实验无效

fix: `DetectionVerifierConfig` 增加 `temperature: float = 1.0` 字段；`build_rewarded_roi_actions()` 内 `normalize_group_advantages(rewards, temperature=cfg.temperature)`。

### 4. R_amp z-score 把 FP(0) 拉成 ~-650 :: reward 极端负值污染梯度

fix: `compute_r_amp_stats_from_loader()` 输出 `{"p05": ..., "p95": ..., ...}`；`normalize_ramp` 改为 `clamp((R_amp - p05) / (p95 - p05 + 1e-6), 0, 1)`；FP 频域项固定为 0。

### 5. 训练候选框阈值和评估阈值复用同一个 :: 口径混淆

fix: 新增 `reward_score_threshold=0.2`，只在 `build_rewarded_roi_actions()` 内过滤 RLVR 训练候选框；`eval_detector.py` 仍用 `score_threshold=0.05`。

### 6. s_amp 过滤与 boxes 过滤不同步 :: R_amp 和过滤后 box 错位

fix: 训练循环中，`reward_score_threshold` 过滤先用 `keep = scores >= reward_score_threshold` 统一过滤 boxes/labels/scores/s_amp，再取 top-k。保证所有特征使用同一个 keep mask 和 order。

---

## 保守参数

| 参数 | 值 | 说明 |
|------|-----|------|
| unfreeze | cls | 只更新 cls_score |
| box_loss_weight | 0 | 完全跳过 bbox target 编码 |
| optimizer | adamw | 固定 |
| temperature | 1.0 | 固定 |
| max_candidates | 40 | 从 80 降到 40 |
| reward_score_threshold | 0.2 | 和 eval 的 0.05 分离 |
| reward_lambda | 0.1 | R_amp 在 verifier reward 中的权重 |

---

## 搜索矩阵（6 trials + 完整 preset）

```json
[
  {"name": "iou_cls_005",    "signal": "none",           "reward_lambda": 0.0,  "policy_loss_weight": 0.005, "box_loss_weight": 0.0, "unfreeze": "cls", "optimizer": "adamw", "temperature": 1.0, "max_candidates": 40, "reward_score_threshold": 0.2},
  {"name": "iou_cls_01",     "signal": "none",           "reward_lambda": 0.0,  "policy_loss_weight": 0.01,  "box_loss_weight": 0.0, "unfreeze": "cls", "optimizer": "adamw", "temperature": 1.0, "max_candidates": 40, "reward_score_threshold": 0.2},
  {"name": "ramp_cls_005",   "signal": "ramp",           "reward_lambda": 0.1,  "policy_loss_weight": 0.005, "box_loss_weight": 0.0, "unfreeze": "cls", "optimizer": "adamw", "temperature": 1.0, "max_candidates": 40, "reward_score_threshold": 0.2},
  {"name": "ramp_cls_01",    "signal": "ramp",           "reward_lambda": 0.1,  "policy_loss_weight": 0.01,  "box_loss_weight": 0.0, "unfreeze": "cls", "optimizer": "adamw", "temperature": 1.0, "max_candidates": 40, "reward_score_threshold": 0.2},
  {"name": "shuffled_cls_01", "signal": "shuffled_ramp",  "reward_lambda": 0.1,  "policy_loss_weight": 0.01,  "box_loss_weight": 0.0, "unfreeze": "cls", "optimizer": "adamw", "temperature": 1.0, "max_candidates": 40, "reward_score_threshold": 0.2},
  {"name": "ramp_cls_03",    "signal": "ramp",           "reward_lambda": 0.1,  "policy_loss_weight": 0.03,  "box_loss_weight": 0.0, "unfreeze": "cls", "optimizer": "adamw", "temperature": 1.0, "max_candidates": 40, "reward_score_threshold": 0.2}
]
```

---

## 训练诊断日志

每 epoch 记录以下字段到 `metrics_train.jsonl`：

| 字段 | 含义 |
|------|------|
| candidate_count | 平均每张图进入 RLVR 的候选框数 |
| matched_tp_count | 平均匹配 TP 数 |
| fp_count | 平均 FP 数 |
| amp_norm_mean | 归一化后 R_amp 的 batch 均值 |
| amp_norm_std | 归一化后 R_amp 的 batch 标准差 |
| policy_weight_mean | softmax 权重的均值 |
| policy_weight_max | softmax 权重的最大值 |
| loss_det | L_det |
| loss_policy_cls | L_candidate_quality |
| shuffle_effective | shuffled_ramp 是否实际打乱（bool） |

目的：如果 ramp 和 shuffled 没差异，可以快速判断是频域信号方差太小还是实现没生效。

---

## Hard Constraints（两级成功标准）

### 一级成功（必须全部通过）

```
AP50_clean          >= baseline - 0.05  (~0.83)
Recall_clean        >= baseline - 0.05  (~0.85)
AP75_clean          >= baseline - 0.10  (~0.58)
high_conf_FP_clean  <= baseline + 3    (~5)
```

### 二级成功（一级通过后才判断）

```
AP50_object_edge            >= baseline_edge - 0.08
Recall_object_edge          >= baseline_edge - 0.07
high_conf_FP_object_edge    <= baseline_edge + 5
ramp_cls_01 > iou_cls_01    (ECE 或 AP75_edge)
ramp_cls_01 > shuffled_cls_01  (频域因果性)
```

如果一级都没过 → RLVR 还未稳定，不能讨论 R_amp 是否有效。

---

## 实现步骤

### Commit 1: `fix: percentile R_amp norm, shuffled_ramp in verifier, temperature in config`

- `spectral/rlvr_reward.py`：compute_r_amp_stats_from_loader 输出 p05/p95，normalize_ramp 改为 percentile clamp
- `rlvr/detection_verifier.py`：DetectionVerifierConfig 加 temperature；build_rewarded_roi_actions 内 shuffled_ramp 逻辑
- `train/posttrain_rlvr.py`：去掉训练循环里的 shuffle；统一 keep mask 过滤 boxes/labels/scores/s_amp

### Commit 2: `fix: skip bbox target encode when box_loss_weight=0, add diagnostic logging`

- `train/posttrain_rlvr.py`：box_loss_weight<=0 时完全跳过 box_coder.encode；每 epoch 记录候选框统计
- `nni_rlvr_trial.py`：默认 unfreeze=cls，接受 reward_score_threshold 和 max_candidates 传参

### Commit 3: `test: add Round 2.1 verifier + NNI tests`

- `tests/test_rlvr_verifier.py`：temperature changes weights, shuffled only TP, FP amp=0, percentile clamp [0,1]
- `tests/test_nni_rlvr_round21.py`：preset has unfreeze=cls, box_loss_weight=0, reward_score_threshold, amp_mask_sync

### Commit 4: `feat: Round 2.1 NNI config — 6 trials stability-first`

- `nni_configs/rlvr_round21_search_space.json`：6 preset（全部展开完整字段）
- `nni_configs/rlvr_round21_config.yml`：GridSearch

---

## 验证

1. `pytest` 全部通过（包括新增 test_rlvr_verifier + test_nni_rlvr_round21）
2. IoU-only cls RLVR（policy_loss_weight=0.005）通过一级所有 hard constraints
3. ramp 组通过一级约束，且在 ECE/AP75_edge 上 > iou + > shuffled
4. baseline_metrics.json 在 NNI 前生成（包含 clean + object_edge）
5. 训练日志确认 shuffle_effective=True 且 amp_norm_std > 0

---

## Plan 位置

`E:/CLIproject/RLimage/docs/superpowers/plans/2026-06-04-round21-stability-first-rlvr.md`
