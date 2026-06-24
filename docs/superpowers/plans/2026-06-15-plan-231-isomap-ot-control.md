# Plan: Isomap(6) + Sinkhorn OT 位移场控制实验

## 目标
将 Isomap(6) 流形距离通过 Sinkhorn 最优传输（OT）位移场传递给分类器，实现大步低能量更新。区别于 DPO pair 框架和 PG 独立采样，每个 batch 计算一次 OT plan，做一次聚合更新。

## 核心洞察（基于预计算数据验证）

| 指标 | 数值 | 含义 |
|------|------|------|
| Isomap pair 一致率 | **60.6%** | 同标签样本比跨标签样本更接近的概率 |
| f(d) vs IoU Spearman | **0.51** | 流形距离映射到目标置信度后与 IoU 中度相关 |
| 全 Sinkhorn OT 位移方差 | **0.018** | 比 PG 梯度方差（0.052）低 **2.8x** |
| OT 位移 pair 一致率 | **52.3%** | 位移方向与 IoU 排序的一致性 |
| 最优 temperature | **1.0** | 在一致性和方差之间平衡最佳 |
| 最优 Sinkhorn reg | **0.1** | 收敛稳定且方差足够低 |

**关键发现**：
- 全 Sinkhorn OT（reg=0.1, temp=1.0）方差比 PG 低 2.8x，但 pair 一致性从 f(d) 的 63% 降至 52%——OT 平滑操作会损失部分排序信号
- reg=0.01 时一致性回升到 60.7%（接近原始 f(d)），但方差比仅 2.1x——存在 reg 与一致性的 trade-off
- 这是 **batch-level 聚合更新**，不是 per-proposal 独立梯度——天然方差更低

## 1. 数据流设计

### 1.1 训练时数据流（per batch）

```
输入: images, targets (batch_size=2, list-of-images格式)

Step 1: 前向传播
  model(images, targets) → loss_dict (detection losses)

  通过 forward hooks 捕获:
    - box_roi_pool pre-hook → sampled_props (proposal boxes, list of tensors)
    - box_head pre-hook → roi_features (N, 256, 7, 7)

Step 2: 提取 per-proposal 信息
  bf = model.roi_heads.box_head(roi_features)        # (N, 1024)
  cls_logits = model.roi_heads.box_predictor.cls_score(bf)  # (N, 2)
  person_logit = cls_logits[:, 1]                     # (N,)
  current_conf = sigmoid(person_logit)                # (N,)

  通过 baseline model (frozen) 计算:
    baseline_bf = baseline_model.roi_heads.box_head(roi_features)
    baseline_logits = baseline_bp.cls_score(baseline_bf)
    baseline_conf = sigmoid(baseline_logits[:, 1])    # (N,)

Step 3: 计算 Isomap 流形距离
  方案 A（实时计算）:
    - 从 roi_features 提取 amp_lo per-channel stats (N, 768)
    - StandardScaler + PCA(50, whiten=True) 投影
    - 计算到 TP median 的 whitened Euclidean 距离 d_i

  方案 B（预查表）:
    - 在 runner 初始化时加载 embeddings.npz
    - 通过 proposal 索引或特征 hash 映射到预计算 Isomap 嵌入
    - 计算 d_i = ||Isomap(x_i) - TP_median||_2

  推荐: 方案 A（实时），因为训练过程中 roi_features 会漂移，预计算嵌入可能过时。
  但为减少开销，可先验证方案 B 是否足够稳定。

Step 4: 构建目标分布
  d_median = median(d_i)  # 按 image 分组计算 median
  f(d_i) = sigmoid(-(d_i - d_median) / temperature)  # 离 TP 越近，目标置信度越高

  目标分布 ν = {f(d_i)}，源分布 μ = {current_conf_i}

Step 5: Sinkhorn OT Plan
  代价矩阵 M_ij = (current_conf_i - f(d_j))^2
  a = b = ones(N) / N  (均匀边际)
  plan = ot.sinkhorn(a, b, M, reg=sinkhorn_reg, numItermax=1000, stopThr=1e-6)

  每个 proposal 的 OT barycenter:
    bary_i = sum_j(plan[i,j] * f(d_j)) / sum_j(plan[i,j])
    delta_i = bary_i - current_conf_i  # OT displacement

Step 6: Loss 构建
  ot_loss = mean(delta_i^2)  # 位移场能量
  kl_loss = KL_WEIGHT * mean((current_conf - baseline_conf)^2)  # 锚定 baseline

  total_loss = det_loss + OT_WEIGHT * ot_loss + kl_loss

Step 7: 反向传播
  通过 person_logit → cls_score → box_head → 梯度回传
  注意: ot_loss 的梯度只通过 current_conf（即 person_logit）传播，
        不通过 f(d_i) 传播（f(d_i) 是目标，detach）
```

### 1.2 验证时数据流（frozen baseline）

复用 `prototype_ot_control.py` 模式：
1. 加载 frozen baseline checkpoint
2. 在 val 集上跑 inference，收集 per-image proposals
3. 对每个 image 独立计算 OT plan 和 displacement
4. 可视化 displacement 方向 vs IoU 排序
5. 报告 per-image variance ratio 和 pair consistency

## 2. 超参设计

### 2.1 固定超参（基于预计算数据）

| 参数 | 值 | 理由 |
|------|-----|------|
| Isomap 维度 | 6 | 已验证最优（diagnostic_6d.py） |
| PCA 维度 | 50 | 与 round2112 一致 |
| temperature | 1.0 | 预计算数据最优（一致性 63%，方差比 4.3x） |
| Sinkhorn reg | 0.1 | 收敛稳定，方差足够低 |
| Sinkhorn max_iter | 1000 | 标准值 |
| Sinkhorn stopThr | 1e-6 | 标准值 |

### 2.2 待调超参

| 参数 | 搜索范围 | 初始值 | 调参逻辑 |
|------|----------|--------|----------|
| OT_WEIGHT | [0.01, 0.1, 0.5, 1.0] | 0.1 | 控制 OT 位移场强度。太小则信号弱，太大则震荡。参考 DPO_WEIGHT=0.1 |
| KL_WEIGHT | [0.001, 0.01, 0.1] | 0.01 | 锚定 baseline，防止漂移。与 round2108 一致 |
| temperature | [0.5, 1.0, 2.0] | 1.0 | 影响 f(d) 的锐度。低温 = 硬阈值，高温 = 软过渡 |
| Sinkhorn reg | [0.05, 0.1, 0.5] | 0.1 | 正则化强度。小 reg = 更接近硬 OT，但可能数值不稳定 |
| 优化器 head_lr | [0.001, 0.0005] | 0.001 | 标准检测头学习率 |
| 优化器 body_lr | [0.0001, 0.00005] | 0.0001 | 标准 body 学习率 |
| 梯度裁剪 | [1.0, 2.0] | 2.0 | 与现有 runner 一致 |

### 2.3 消融对照组

1. **det_only_unf**: 纯检测 loss，无 OT（baseline）
2. **ot_iou_target**: OT 目标分布用 IoU 代替 f(d)——验证"流形距离 vs IoU 直接监督"哪个更好
3. **ot_manifold**: 完整方案（Isomap + OT）
4. **pg_baseline**: 复现 round2102 PG 方案，直接对比方差和收敛稳定性

## 3. 验证计划（Phase 0：不训练）

在编写训练 runner 之前，先跑验证脚本确认 displacement 方向正确。

### 3.1 验证脚本：`scripts/validate_ot_manifold.py`

功能：
1. 加载 frozen baseline checkpoint（`runs/round227_v1_baseline_20ep/checkpoint_best.pth`）
2. 在 val 集上收集 proposals（batch_size=2, score_threshold=0.05, max_candidates=40）
3. 对每个 image：
   - 提取 roi_features → amp_lo stats → StandardScaler+PCA → Isomap 距离 d_i
   - 计算 f(d_i) = sigmoid(-(d_i - d_median)/temperature)
   - 运行 Sinkhorn OT 得到 displacement delta_i
   - 计算 displacement 与 IoU 的 Spearman 相关
   - 计算 pair consistency（displacement 方向 vs IoU 排序）
4. 聚合报告：
   - 全局 OT displacement variance vs PG variance
   - 全局 pair consistency
   - per-image 分布（histogram）
   - 可视化：scatter plot（displacement vs IoU）

### 3.2 验证通过标准

- [ ] OT displacement 与 IoU 的 Spearman > 0.3（正相关，即 displacement 方向正确）
- [ ] per-image pair consistency > 0.55（高于随机 0.5）
- [ ] variance ratio PG/OT > 2.0（OT 方差显著低于 PG）
- [ ] Sinkhorn 在所有 image 上收敛（无 NaN/inf）
- [ ] 单个 image 的 OT 计算时间 < 100ms（N~40 时）

### 3.3 预计算 Isomap 校准

与 round2112 相同的校准流程：
1. 在训练集上跑 frozen baseline，收集所有 TP 的 amp_lo features
2. fit StandardScaler + PCA(50, whiten=True)
3. 在 whitened space 中 fit Isomap(6)
4. 计算 TP cluster median
5. 保存 calib.pkl（scaler, pca, isomap, tp_median）

**注意**：训练过程中 roi_features 会漂移，实时 Isomap 距离可能不完全准确。但 OT 的聚合性质对单点误差有鲁棒性。

## 4. 实验设计

### 4.1 实验配置

| 配置 | 说明 |
|------|------|
| 数据集 | Penn-Fudan Pedestrian |
| 检测器 | Faster R-CNN MobileNetV3-Large-320-FPN |
| Checkpoint | `runs/round227_v1_baseline_20ep/checkpoint_best.pth` |
| Batch size | 2（与所有 2.x 实验一致） |
| Epochs | 8（与 round2108/2112 一致） |
| Seed | 42（单种子快速验证） |
| 训练模式 | unfreeze_rlvr（body_lr=0.0001, head_lr=0.001） |
| 设备 | CUDA（8GB VRAM） |

### 4.2 实验组

| 组名 | 模式 | 说明 |
|------|------|------|
| A | det_only_unf | 纯检测 loss baseline |
| B | ot_manifold | Isomap + Sinkhorn OT（完整方案） |
| C | ot_iou_target | OT 目标用 IoU 代替 f(d)（对照） |
| D | pg_baseline | 复现 round2102 PG（方差对比） |

### 4.3 评估指标

| 指标 | 来源 |
|------|------|
| AP50 / AP75 | `evaluate_model()` |
| ECE | `evaluate_model()` |
| Precision@R=0.85 | `evaluate_model()` |
| OT displacement variance | 训练日志 |
| PG gradient variance | 训练日志（对照组） |
| per-image pair consistency | 训练日志 |
| 训练时间 / epoch | 计时 |

### 4.4 预期结果

| 假设 | 预期 |
|------|------|
| 稳定性 | OT 组比 PG 组收敛更平滑（loss curve 方差低 1.3x-2.8x） |
| AP75 | OT 组 >= det_only 组（不劣化） |
| ECE | OT 组可能降低（置信度校准改善） |
| 速度 | 每 batch 增加 ~50-100ms（N~40 的 Sinkhorn） |
| 可扩展性 | N>200 时 Sinkhorn O(N^2) 可能成为瓶颈 |

## 5. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| Sinkhorn O(N^2) 太慢 | 中 | 训练时间增加 2-3x | 1. 限制 max_candidates=40；2. 使用 batch-level 而非 image-level OT；3. 考虑 POT 的 GPU 版本 |
| Isomap 距离在训练中漂移 | 高 | 目标分布不准确 | 1. 每 epoch 重新校准；2. 使用在线 PCA；3. 回退到 IoU-based target |
| OT displacement 方向错误 | 中 | 置信度更新方向与 IoU 相反 | Phase 0 验证脚本必须先通过 pair consistency > 0.55 |
| Sinkhorn 数值不稳定 | 低 | NaN/inf | 1. 增加 reg；2. 添加 clip；3. 检查 cost matrix 尺度 |
| 梯度链断裂 | 低 | OT loss 不传播 | 确认 current_conf 通过 sigmoid(person_logit) 计算，person_logit 有 grad |
| 内存泄漏 | 低 | OOM | 每 batch 释放 OT plan 矩阵（del plan） |

## 6. 实现步骤（按优先级）

### Phase 0: 验证（1-2 天）
1. 编写 `scripts/validate_ot_manifold.py`（frozen baseline OT 可视化）
2. 跑通验证，确认 pair consistency > 0.55 且 variance ratio > 2.0
3. 调 temperature 和 Sinkhorn reg，记录最佳组合
4. 测量 per-image OT 计算时间（N=20,40,80,160）

### Phase 1: 最小训练 runner（2-3 天）
1. 复制 `round2108_dpo.py` → `round2115_ot_manifold.py`
2. 替换 DPO loss 为 OT loss：
   - 添加 `extract_amp_lo_perchan_stats()`（从 round2112 复制）
   - 添加 `compute_isomap_distance()`（实时 PCA+Isomap）
   - 添加 `sinkhorn_ot_plan()`（从 prototype_ot_control.py 复制）
   - 添加 `compute_ot_loss()`（displacement energy + KL）
3. 跑单 seed 42，8 epoch，对比 det_only
4. 检查 loss curve 和梯度稳定性

### Phase 2: 消融与调参（2-3 天）
1. 跑 ot_iou_target 对照组（验证 Isomap 是否优于直接 IoU）
2. 调 OT_WEIGHT（0.01, 0.1, 0.5）
3. 调 temperature（0.5, 1.0, 2.0）
4. 调 Sinkhorn reg（0.05, 0.1, 0.5）
5. 记录最佳组合

### Phase 3: 扩展验证（可选，2-3 天）
1. 多 seed 验证（42, 123, 456）
2. 与 PG baseline 直接对比方差和收敛曲线
3. 尝试 batch-level OT（跨 image 聚合，而非 per-image）
4. 尝试预计算 Isomap 查表（验证实时计算开销是否必要）

## 7. 代码结构

```
scripts/
  validate_ot_manifold.py      # Phase 0: frozen baseline validation
  round2115_ot_manifold.py      # Phase 1: training runner

spectral_detection_posttrain/
  experiments/
    runner_utils.py             # 已有: build_opt, evaluate_model, etc.
  ot/
    __init__.py
    sinkhorn.py               # sinkhorn_ot_plan(), compute_ot_loss()
    isomap_target.py          # extract_features(), compute_isomap_distance(), build_target_distribution()
```

## 8. 关键代码片段（伪代码）

### 8.1 OT Loss 计算

```python
def compute_ot_loss(person_logit, baseline_person_logit, roi_features, scaler, pca, isomap, tp_median,
                    temperature=1.0, sinkhorn_reg=0.1, ot_weight=0.1, kl_weight=0.01):
    """
    person_logit: (N,) current model's person class logit
    baseline_person_logit: (N,) frozen baseline's person class logit
    roi_features: (N, 256, 7, 7) ROI features before box_head
    """
    # 1. Current confidence
    current_conf = torch.sigmoid(person_logit)  # (N,)

    # 2. Isomap distance (detach, no grad through feature extraction)
    with torch.no_grad():
        amp_lo_feats = extract_amp_lo_perchan_stats(roi_features)  # numpy (N, 768)
        whitened = pca.transform(scaler.transform(amp_lo_feats))   # (N, 50)
        isomap_emb = isomap.transform(whitened)                  # (N, 6)
        dists = np.linalg.norm(isomap_emb - tp_median, axis=1)  # (N,)

        # 3. Target distribution
        d_median = np.median(dists)
        z = np.clip((dists - d_median) / temperature, -50, 50)
        f_d = 1.0 / (1.0 + np.exp(z))  # (N,)
        f_d_tensor = torch.from_numpy(f_d).to(person_logit.device).float()

    # 4. Sinkhorn OT
    N = len(current_conf)
    M = ot.dist(current_conf.detach().cpu().numpy().reshape(-1, 1),
                f_d.reshape(-1, 1), metric='sqeuclidean')  # (N, N)
    a = np.ones(N) / N
    b = np.ones(N) / N
    plan = ot.sinkhorn(a, b, M, reg=sinkhorn_reg, numItermax=1000, stopThr=1e-6, verbose=False)

    # 5. Barycentric displacement
    row_sums = plan.sum(axis=1, keepdims=True).clip(min=1e-9)
    barycenter = (plan @ f_d.reshape(-1, 1)) / row_sums  # (N, 1)
    bary_tensor = torch.from_numpy(barycenter.squeeze(-1)).to(person_logit.device).float()

    delta = bary_tensor - current_conf  # (N,)

    # 6. Loss
    ot_loss = delta.pow(2).mean()
    baseline_conf = torch.sigmoid(baseline_person_logit)
    kl_loss = (current_conf - baseline_conf).pow(2).mean()

    return ot_weight * ot_loss + kl_weight * kl_loss, {
        'ot_loss': ot_loss.item(),
        'kl_loss': kl_loss.item(),
        'delta_mean': delta.mean().item(),
        'delta_std': delta.std().item(),
    }
```

### 8.2 训练循环集成

```python
# 在 round2108_dpo.py 的 training loop 中替换 DPO 部分:
if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
    bf = model.roi_heads.box_head(rf)
    cls_logits = model.roi_heads.box_predictor.cls_score(bf)
    person_logit = cls_logits[:, 1]  # (N,)

    with torch.no_grad():
        baseline_bf = baseline_model.roi_heads.box_head(rf)
        baseline_logits = baseline_bp.cls_score(baseline_bf)
        baseline_person = baseline_logits[:, 1]

    # 替换这里: DPO → OT
    ot_loss, ot_diag = compute_ot_loss(
        person_logit, baseline_person, rf,
        scaler, pca, isomap, tp_median,
        temperature=TEMPERATURE,
        sinkhorn_reg=SINKHORN_REG,
        ot_weight=OT_WEIGHT,
        kl_weight=KL_WEIGHT,
    )

    diag['ot_loss'].append(ot_diag['ot_loss'])
    diag['delta_std'].append(ot_diag['delta_std'])

    loss = det + ot_loss
```

## 9. 与现有工作的关系

| 工作 | 方法 | 区别 |
|------|------|------|
| round2102 (PG) | 独立采样 + GRPO advantage | 方差高，per-proposal 独立梯度 |
| round2108 (DPO) | Pairwise best-vs-worst | 需要 pair 构造，只更新 pair 差异 |
| round2112 (Manifold DPO) | Isomap + DPO pair | 用流形距离选 pair，但仍是 pair 框架 |
| **本计划 (OT)** | Isomap + Sinkhorn batch OT | 全 batch 聚合，一次大步更新，方差更低 |

## 10. 成功标准

| 级别 | 标准 |
|------|------|
| 最低 | Phase 0 验证通过（consistency > 0.55, variance ratio > 2.0） |
| 期望 | 单 seed OT 组 AP75 >= det_only 组，且 loss curve 更平滑 |
| 优秀 | OT 组 AP75 > 0.70（超越 round2108 DPO 的 0.724），且训练时间 < 2x det_only |
| 突破 | 多 seed 验证稳定，方差比 PG 低 2x 以上，证明 batch-OT 是检测 RLVR 的有效范式 |

## 11. 时间线

| 阶段 | 任务 | 预估时间 |
|------|------|----------|
| Phase 0 | 验证脚本 + 跑通验证 | 1-2 天 |
| Phase 1 | 最小 runner + 单 seed 训练 | 2-3 天 |
| Phase 2 | 消融调参 | 2-3 天 |
| Phase 3 | 多种子 + 对比分析 | 2-3 天 |
| 总计 | | **7-11 天** |

## 12. 附录：预计算数据关键统计

```
embeddings.npz:
  Isomap: (3224, 6)   — 3224 proposals from 34 images
  ious: (3224,)        — 955 TP, 2269 FP
  confs: (3224,)       — current confidence
  img_ids: (3224,)     — image grouping

raw_features.npz:
  X_raw: (3224, 768)   — amp_lo per-channel stats (mean, std, max over 28 freq bins × 256 channels)

关键发现:
  - Isomap pair agreement: 60.6% (vs random 50%)
  - Distance to TP median vs IoU: Spearman = -0.51
  - Full Sinkhorn OT (reg=0.1, temp=1.0): variance = 0.018, PG variance = 0.052, ratio = 2.8x
  - OT displacement consistency: 52.3% (reg=0.1) vs 60.7% (reg=0.01)
```
