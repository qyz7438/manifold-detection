# RLVR / DPO 方向审计与优化方案

日期: 2026-06-16

范围:
- 包内 RLVR 主路径: spectral_detection_posttrain/train/posttrain_rlvr.py, spectral_detection_posttrain/rlvr/*, spectral_detection_posttrain/spectral/rlvr_reward.py
- 代表性 RLVR runner: scripts/round2101_nwpu_rlvr.py, round2103_runner.py, round2105_runner.py, round2107_nwpu_discrete.py, round2116_edge_rlvr.py, round2117_tpnn_rlvr.py
- 代表性 DPO runner: scripts/round250_runner.py, round251_runner.py, round252_runner.py, round255_runner.py, round258_runner.py, round2108_dpo.py, round2112_manifold_dpo.py, round2113_*.py, round2115_edge_dpo.py, round2119_tpnn_dpo.py, round297b_dpo_repro.py

验证:
- 命令: python -m pytest tests/test_rlvr_verifier.py tests/test_rlvr_policy_objective.py tests/test_roi_policy_loss.py tests/test_round30_training_diagnostics.py -q
- 结果: 20 passed

## 总体判断

RLVR 当前包内实现已经有一个可复用 shell: frozen baseline rollout, signed policy objective, KL anchor, freeze-state control。但它的核心动作仍是“在固定 rollout boxes 上调整分类 logit”，不是完整的 bbox action policy。也就是说，它更像 confidence calibration / ROI classification policy gradient，而不是能直接学习“框往哪里移动”的检测 RLVR。

DPO 分成两类:
- 2.50-2.58: 高斯采样 bbox delta 的 DPO。形式接近动作偏好学习，但实现质量参差不齐，2.52 有明确无效 pair bug，2.55/2.58 才让质量依赖 decoded box。
- 2.108 以后: GT group 内候选 proposal 的 pairwise logit DPO。数据流更短，能直接改 classifier，结果更容易稳定；但它本质还是 proposal reranking / confidence calibration，bbox 回归主要仍靠 detection loss。

优先建议: 短期优先把 DPO 收敛成 canonical candidate-preference runner；RLVR 暂停继续堆脚本，先修 action/logprob/reward 闭环。若要证明 spectral verifier 有用，必须让 spectral quality 对 action 后的 decoded box 产生可测差异，并用 shuffled preference / IoU oracle / det-only 同范围对照隔离效果。

## RLVR 主要漏洞

### P0: no-candidate 分支返回字段不完整

build_rewarded_roi_actions 在没有候选框时只返回 boxes/labels/matched_gt_boxes/weights，缺少 policy_labels/advantages/matched/scores/rewards 等字段。主训练循环随后无条件拼接 policy_labels/advantages，并统计 matched。

证据:
- spectral_detection_posttrain/rlvr/detection_verifier.py:129-135
- spectral_detection_posttrain/train/posttrain_rlvr.py:280-284
- spectral_detection_posttrain/train/posttrain_rlvr.py:324-326

影响:
- 某张图没有高于 reward_score_threshold 的预测框时会 KeyError。
- 当前 Penn-Fudan baseline 可能很少触发，所以单测和实验没有暴露。
- 注释说 recovery boxes 不进入 policy，但 no-candidate 分支实际把 GT boxes 当作 boxes 返回，语义自相矛盾。

修复:
- no-candidate 返回“空 action”且包含完整字段，所有长度为 0。
- 如果要 recovery loss，单独返回 recovery_boxes/recovery_labels，不要混进 policy action。
- 加单测: GT 存在但 prediction 为空、prediction 全被 threshold 过滤。

### P0: GT 为空时 verifier 会崩

当有预测框但 gt_boxes 为空时，ious.max(dim=1) 对 (N, 0) 张量求 max 会报错。

证据:
- spectral_detection_posttrain/rlvr/detection_verifier.py:137-139
- spectral_detection_posttrain/rlvr/detection_verifier.py:148

影响:
- Penn-Fudan 每图有行人，所以没暴露。
- 迁移到 VOC/NWPU/VisDrone 或负样本数据时会直接崩。

修复:
- gt_boxes.numel() == 0 时所有候选设为 unmatched/background，reward 只允许 FP penalty 或 0。
- 加单测覆盖空 GT。

### P1: signed RLVR 目标没有 bbox action 闭环

当前 signed objective 选择的是 model 自己预测的类别 label，loss 是 advantage * log p(class)。rollout box 是固定候选框，reward 评估这些固定框；bbox regression head 不通过 signed policy loss 得到“哪个 delta 更好”的梯度。

证据:
- policy label 来自预测类别: spectral_detection_posttrain/rlvr/detection_verifier.py:145-146
- signed loss 只取 class logprob: spectral_detection_posttrain/rlvr/roi_policy_loss.py:80-82
- 主 loss 组合: spectral_detection_posttrain/train/posttrain_rlvr.py:296-312

影响:
- 如果实验说“RLVR 学会框得更准”，当前实现无法支撑这个解释。
- unfreeze=box/roi 时 bbox 改善主要来自 supervised detection loss、KL/共享特征副作用，或 legacy CE path，而不是 signed RLVR 本身。

修复:
- 明确拆成三种 action head: keep/reject action、class action、bbox-delta action。
- 若目标是定位，必须采样 bbox deltas，decode 后计算 reward，再用对应 delta 的 logprob 训练 bbox policy。
- 记录每个 objective 能影响哪些参数，禁止把 classifier-only RLVR 解释成 localization RLVR。

### P1: legacy weighted CE 的 bbox target 是零

非 signed 分支里 bbox regression target 被写死为全 0，然后交给 weighted_fastrcnn_policy_loss 做 smooth L1。

证据:
- spectral_detection_posttrain/train/posttrain_rlvr.py:299-304
- spectral_detection_posttrain/rlvr/roi_policy_loss.py:37-46

影响:
- 如果启用 --policy-objective weighted_ce 且 box_loss_weight > 0，模型会被训练成对正样本输出 0 delta，即“proposal 不需要修正”，这不是 matched GT 的真实 regression target。

修复:
- 使用 TorchVision BoxCoder.encode(matched_gt_boxes, proposal_boxes) 生成 regression targets。
- 或在 canonical runner 中禁用该分支，直到 target 计算正确。

### P1: verifier reward 对 unmatched FP 仍可能给正 IoU reward

compute_box_rewards 直接加 w_iou * ious，只有高置信 unmatched 框被扣分。低置信但 IoU < 0.5 的 FP 仍可得到正 reward。

证据:
- spectral_detection_posttrain/rlvr/detection_verifier.py:46-48
- matched hard threshold: spectral_detection_posttrain/rlvr/detection_verifier.py:139

影响:
- 近阈值 FP 可能被鼓励，尤其在 high_conf_threshold 以下。
- 这会削弱 verifier 对 TP/FP 的 causal 区分。

修复:
- 明确 reward 定义: unmatched 的 IoU reward 是否允许存在。
- 更稳妥默认: reward = matched * (w_iou*iou + w_cls*class + spectral) - fp_penalty。
- 加单测: IoU=0.49, score=0.2 的 unmatched 框 reward 不应为正。

### P1: amp signal 缺 stats 时被静默置零

选择 ramp/shuffled_amp/amp_structure 等信号但未提供 --r-amp-stats 时，训练继续跑，s_amp 全零。

证据:
- stats 只在有路径时加载: spectral_detection_posttrain/train/posttrain_rlvr.py:139-141
- 缺 stats 时置零: spectral_detection_posttrain/train/posttrain_rlvr.py:260-265

影响:
- run 名字/配置显示使用 amp reward，但实际 reward 没有 amp 成分。
- 会污染 ablation 结论。

修复:
- amp signal 必须要求 stats 文件存在且 count > 0。
- 写入 metadata: stats path/hash/count/p05/p95。

### P2: matching 阈值和日志口径不统一

R_amp 匹配用 config 的 matching.iou_threshold/score_threshold；action/verifier 匹配 hard-code best_iou >= 0.5；候选过滤又用 reward_score_threshold。

证据:
- R_amp 匹配: spectral_detection_posttrain/train/posttrain_rlvr.py:256
- action 匹配: spectral_detection_posttrain/rlvr/detection_verifier.py:139
- 候选过滤: spectral_detection_posttrain/rlvr/detection_verifier.py:110-127

影响:
- 同一候选在 R_amp 里可能是 unmatched，在 verifier 里可能是 matched，反之亦然。
- 不记录这些阈值时，不同 run 很难比较。

修复:
- 把 action IoU threshold 放入 DetectionVerifierConfig。
- 每个 run 记录 matching_thresholds 和 reward_thresholds。

### P2: epoch diagnostics 只记录最后一个 batch

candidate_count/matched_count/advantage_mean/std/person_rate 在 batch 循环内更新，epoch row 只写最后一个 batch 的值。

证据:
- batch 内计算: spectral_detection_posttrain/train/posttrain_rlvr.py:324-350
- epoch row 写入: spectral_detection_posttrain/train/posttrain_rlvr.py:338-351

影响:
- metrics_train.jsonl 的 reward/advantage 统计可能误导排查。

修复:
- 使用 accumulator 记录 epoch mean/std/count。
- 复用已有 build_reward_component_summary 汇总 reward components。

### P2: checkpoint selection 只看 AP50

主 runner 用 val_ap50 选 best checkpoint。

证据:
- spectral_detection_posttrain/train/posttrain_rlvr.py:334-361

影响:
- 项目当前主要收益在 AP75/ECE，AP50 选 checkpoint 可能选错 epoch。

修复:
- 配置化 selection_metric，默认检测后训练用 AP75，校准实验可用 ECE 或 composite。

## RLVR runner 方向问题

### 2.101/2.103 classifier RLVR 是 confidence calibration，不是定位 RLVR

这些 runner 从 baseline logits 采样分类扰动，在 current logits 上算 logprob；reward 通常由 baseline bbox/IoU 或几何分数给出。对于同一 proposal 的多个 samples，box 几何基本不变，梯度主要调整分类置信度。

证据:
- 2.101: scripts/round2101_nwpu_rlvr.py:256-294
- 2.103: scripts/round2103_runner.py:197-235

结论:
- 可以作为 confidence/NMS 排序实验。
- 不应作为 bbox RLVR 证据。

### 2.105/2.107 discrete RLVR 更接近检测 RLVR，但推理闭环仍需定义

keep/reject Bernoulli action + NMS hit reward 是更像 RLVR 的动作设计。

证据:
- scripts/round2107_nwpu_discrete.py:160-177

问题:
- 训练时采样 keep/reject，推理时仍是常规 detector 分数/NMS；action policy 没有作为独立模块输出。
- 需要证明更新后的 confidence 分布确实改变 NMS 排序，而不是 detection loss 主导。

修复:
- 记录 keep rate, hit reward, NMS survivor changes。
- 对照 det-only 同 unfreeze 范围和 shuffled reward。

### 2.116/2.117 spectral bonus 是 proposal-level，不是 decoded-action-level

edge/TPNN bonus 都主要从 raw proposals crop 计算，再加到分类扰动 reward 上。

证据:
- edge bonus: scripts/round2116_edge_rlvr.py:111-117
- TPNN bonus: scripts/round2117_tpnn_rlvr.py:139-158

影响:
- 它可以帮助 proposal scoring，但不能证明 spectral reward 对 bbox 修正有作用。

### P1: 2.117 TP manifold 校准存在 batch offset bug

build_tp_manifold 中对每张图计算 IoU 时使用 decoded[:len(p_img)]，没有按 batch offset 切片。第二张及之后的图会拿错 decoded boxes。

证据:
- scripts/round2117_tpnn_rlvr.py:61-65

对比:
- 2.119 的 DPO 版本已正确使用 offset: scripts/round2119_tpnn_dpo.py:51-64

修复:
- 用 decoded[off:off+n_p]，并在循环末尾更新 off。
- 对任意 batch_size=2 加单测/脚本断言: per-image proposal count 与 decoded slice 对齐。

## DPO 主要漏洞

### P0: 2.52 DPO pair 实际是恒定偏好

2.52 声称使用 ROI FFT quality，但同一个 proposal 的两个 sampled deltas 被赋相同 q_val。随后 chosen = q_quality[:, 0] >= q_quality[:, 1]，所以 sample 0 永远 chosen。

证据:
- scripts/round252_runner.py:178-203

影响:
- 该实验的 DPO loss 不是由 verifier quality 产生，而是固定偏好 sample 0。
- 2.52 的结论应标记为无效或仅作失败案例。

修复:
- quality 必须在 decoded box 或 action 后 ROI 上计算。
- pair 必须要求 abs(q0 - q1) >= margin。
- 加单测: 对同一 proposal 的两个不同 delta，pair label 必须随 quality 变化；平局必须 invalid。

### P1: 2.58 没有 q_diff margin，平局也参与训练

2.58 的 edge_truncation_quality 是 delta-dependent，但所有 pair 都 valid；如果 crop 失败、超过 256 上限、或 quality 相等，>= 会把 sample 0 设为 chosen。

证据:
- scripts/round258_runner.py:203-233

对比:
- 2.55 有 q_diff > 0.02: scripts/round255_runner.py:216-226

修复:
- 所有 sampled-action DPO 统一使用 min_quality_margin。
- 记录 invalid/tie/valid pair 数量。

### P1: 多个 sampled-action DPO 只处理前 256 个 patches

2.50/2.51/2.55/2.58 都限制 min(N*K, 256) 个 crop。后续 action 的 quality 为 0 或 tie，可能被误当成有效 pair。

证据:
- 2.50: scripts/round250_runner.py:185-220
- 2.51: scripts/round251_runner.py:185-225
- 2.55: scripts/round255_runner.py:194-218
- 2.58: scripts/round258_runner.py:203-226

修复:
- 要么同步截断 N = min(N, max_pairs // K)，要么为未计算 quality 的 pair 标记 invalid。
- metrics 记录 quality_computed_pairs / total_pairs。

### P1: reference logprob 泄漏当前版本已修，但实现分散风险高

早期 audit 指出 reference logprob 可能对 current sampled deltas/sigma 反传。当前检查到 2.50/2.51/2.55/2.58 都已有 ref_deltas = deltas.detach()，ref_sigma 也在 no_grad 里生成。

证据:
- 2.50: scripts/round250_runner.py:164-170
- 2.51: scripts/round251_runner.py:164-170
- 2.55: scripts/round255_runner.py:173-179
- 2.58: scripts/round258_runner.py:182-188

问题:
- 每个 runner 自己实现 gaussian_log_prob 和 DPO ratio，容易再次回归。

修复:
- 抽到 spectral_detection_posttrain/dpo/losses.py。
- 单测断言 reference branch 不给 current action sample 或 ref params 产生梯度。

### P1: custom BoxCoder decode 与 TorchVision 推理链路不完全一致

公共 decode_boxes 手写 BoxCoder 解码，只 clamp min=0，没有按 image size clip max，也没有走 TorchVision postprocess/NMS。

证据:
- spectral_detection_posttrain/experiments/runner_utils.py:156-184

影响:
- DPO/RLVR reward 看到的 decoded boxes 可能和真实 detector eval 的 boxes 不一致。
- pixel crop quality 尤其受越界/尺寸裁剪影响。

修复:
- 使用 model.roi_heads.box_coder.decode 或 TorchVision BoxCoder。
- 对每张图按 image_shape 做 clip_boxes_to_image。
- 把 decode/postprocess 统一成公共 helper，runner 不再各写一份。

### P1: 2.108+ proposal-pair DPO 没有 IoU eligibility floor

2.108 用每个 GT group 内最高 IoU vs 最低 IoU 做 DPO；2.112/2.113/2.119 用 manifold/edge 分数选 pair。但 group 是按 max-IoU 的 GT index 分组，不要求候选本身达到 TP 阈值。

证据:
- 2.108 IoU group: scripts/round2108_dpo.py:98-116
- 2.112 manifold group: scripts/round2112_manifold_dpo.py:217-250
- 2.119 TPNN group: scripts/round2119_tpnn_dpo.py:128-140

影响:
- 可能在同一 GT 附近的两个低质量 FP 中学习偏好。
- manifold 分数可能把“像 TP 的背景纹理”推高。

修复:
- pair eligibility: max(iou_pair) >= 0.5 或至少 chosen IoU/score 满足阈值。
- 记录 pair 的 IoU gap、chosen IoU、rejected IoU、quality gap。
- 加 oracle/shuffled ablation。

### P1: 2.108 是 IoU oracle DPO，不是 spectral DPO

2.108 的 chosen/rejected 由 GT IoU 直接决定。

证据:
- scripts/round2108_dpo.py:102-116

解释:
- 它可作为上界/正控，证明 pairwise logit DPO 能改排序。
- 不能用来证明 spectral verifier 有效。

### P1: 2.112/2.113/2.115/2.119 本质是 classifier reranking DPO

这些脚本都对 person_logit 做 pairwise contrastive，pair 来自 proposal-level spectral/manifold/edge 分数。它们没有对 bbox delta action 建模。

证据:
- 2.112: scripts/round2112_manifold_dpo.py:240-250
- 2.113 disagreement: scripts/round2113_manifold_confidence_disagreement_dpo.py:376-392
- 2.115 edge: scripts/round2115_edge_dpo.py:98-113
- 2.119 TPNN: scripts/round2119_tpnn_dpo.py:133-140

影响:
- 这条路线值得保留，但应命名为 candidate preference / reranking DPO。
- 不应解释为 bbox refinement DPO。

### P2: 2.113 disagreement pair selector 使用当前 confidence，偏好目标会移动

2.113 先用当前 person_logit 得到 confidence，再和 manifold score 计算 disagreement，最后用同一个 person_logit 做 DPO 训练。

证据:
- confidence 进入 pair selector: scripts/round2113_manifold_confidence_disagreement_dpo.py:344-350
- pair 选择和 loss: scripts/round2113_manifold_confidence_disagreement_dpo.py:353-392

影响:
- pair label 随模型更新移动，可能形成自反馈。
- 不是不能做，但需要记录 pair turnover 和冻结/滞后 selector 对照。

修复:
- 使用 frozen baseline confidence 或 EMA selector。
- 每 epoch 记录 pair overlap/turnover。

## 优化路线

### P0: 先修可信度基础设施

1. 建 canonical runner，并把旧 scripts/round*.py 作为归档。
2. 加 config schema:
   - 未知 model name 直接报错。
   - 正式实验禁止 random-init fallback。
   - amp signal 缺 stats 直接报错。
   - AFM channels 自动推断或强断言。
3. 每个 run 写 metadata:
   - git commit、dirty diff 摘要、config hash、checkpoint hash、torch/torchvision/cuda 版本。
   - runner snapshot、selection metric、thresholds、seed、determinism flags。
4. 所有 action/pair builder 加 invariant:
   - 字段完整。
   - tensor length 对齐。
   - no-candidate/empty-GT 安全。
   - valid pair 数量不足时 loss 为 0 且记录原因。

### P1: 把 DPO 收敛成两个清晰产品

Track A: candidate reranker DPO
- 以 2.108/2.119/2.97b 为基础。
- 训练显式 scorer 或校准 head，对候选 proposal/action 排序。
- 推理时必须接入 rerank/NMS 前处理，否则只算离线分数不算 detector 改进。
- 指标: top1/top3 IoU, AP75, ECE, precision/recall, NMS survivor change。

Track B: detector-head DPO
- 对同一批 frozen proposals 的 person_logit 做 pairwise DPO。
- pair builder 支持 iou_oracle, spectral, manifold, edge, shuffled。
- 强制 min_iou_floor 和 min_quality_margin。
- det-only 同 unfreeze 范围是必须对照。

短期优先 Track A，因为数据流最短、pair consistency 可离线诊断、失败原因最容易定位。

### P2: RLVR 只保留两种严格定义的实验

1. Keep/reject RLVR:
   - action: per proposal Bernoulli keep/reject。
   - reward: NMS 后 AP75 hit / FP penalty。
   - policy: current confidence 的 Bernoulli logprob。
   - 验证: keep rate、NMS survivor、hit reward、shuffled reward。

2. Bbox-delta RLVR:
   - action: 从 bbox delta distribution 采样。
   - reward: decoded box 的 IoU / spectral quality。
   - policy: bbox delta logprob。
   - 必须用 TorchVision BoxCoder + image clip。

当前包内 signed ROI policy 可作为 classifier RLVR 保留，但命名应改为 roi_class_policy_pg，避免误解为 localization RLVR。

### P3: 评价与结论口径

每个实验至少记录:
- AP50/AP75/ECE/precision/recall/pred count。
- reward mean/std、advantage std、valid pair ratio。
- chosen/rejected IoU gap、quality gap。
- shuffled verifier/pair 结果。
- det-only 同 epoch/同 unfreeze/同 checkpoint 对照。

结论模板:
- “IoU oracle DPO 有效”只能说明 pairwise logit preference 可优化排序。
- “spectral DPO 有效”必须满足: spectral pair consistency > shuffled、DPO > det-only、shuffled DPO 不提升、推理链路实际使用该信号。
- “RLVR 改善定位”必须满足: bbox action logprob 参与训练，reward 来自 action 后 decoded boxes，AP75 提升超过 det-only。

## 推荐下一步

1. 修 P0/P1 测试和代码保护: no-candidate、empty-GT、2.52 tie、2.117 offset、amp stats required。
2. 新建 spectral_detection_posttrain/dpo/，集中 DPO loss、pair builders、quality builders。
3. 先重跑一个最小矩阵:
   - det-only
   - IoU oracle DPO
   - spectral/manifold DPO
   - shuffled spectral/manifold DPO
4. 如果 spectral DPO 仍不能超过 det-only + shuffled，停止 detector-head DPO，转向 explicit reranker 或回到 in-network AFM。
