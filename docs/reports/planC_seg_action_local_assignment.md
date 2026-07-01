# Plan C：语义分割 Action-Local RLVR 实验分配

> 来源：`session_e2b780e4-7251-45d2-bdaa-3bddb609e869` 中的 Plan C 文件 `plans/wonder-man-jean-grey-punisher.md`。  
> 方向：从检测 RLVR 失败转向语义分割的 action-local 可验证奖励，冻结 baseline FCN-ResNet50，只训练小 action head。

---

## 目标

验证 **action-reward 对齐原则**：在语义分割中，只有 reward 直接衡量的变量与 policy action 实际改变的变量对齐时，RLVR/DPO 才能产生稳定的因果改进。

成功标准：至少一种 action-reward 组合在 3 seeds 上稳定提升 mIoU/boundary IoU，而 random reward、supervised fine-tune、end-to-end full-network RLVR 等对照不提升或提升更少。

---

## 前提：必须先实现的代码模块

按 Plan C，跑实验前需完成：

| 步骤 | 模块 | 文件 | 状态 |
|---|---|---|---|
| 1 | 可复用的 Penn-Fudan 分割 loader | `spectral_detection_posttrain/datasets/penn_fudan_seg.py` | 待实现 |
| 2 | Action adapters（logit / feature / calibration） | `spectral_detection_posttrain/core/models/seg_adapters.py` | 待实现 |
| 3 | Reward 函数（含 NC alignment） | `spectral_detection_posttrain/methods/segmentation/action_rewards.py` | 待实现 |
| 4 | NC / flatness 诊断指标 | `spectral_detection_posttrain/methods/segmentation/nc_diagnostics.py` | 待实现 |
| 5 | Action-local RLVR/DPO trainer | `spectral_detection_posttrain/trainers/segmentation/action_local_seg_rlvr.py` | 待实现 |
| 6 | Smoke config + analysis script | `spectral_detection_posttrain/configs/seg_action_local_smoke.yaml`、`scripts/analyze_planC_results.py` | 待实现 |

> 在以上模块可用之前，本分配表处于“待运行”状态。

---

## 核心实验分配（Step 7）

Seeds：`42, 123, 2024`（Plan C 最低证据要求）。  
Dataset：Penn-Fudan 分割（完整训练集）。  
Baseline：`FCN-ResNet50`，训练完成后冻结。

| # | run_name | action | reward | objective | 目的 | 状态 |
|---|---|---|---|---|---|---|
| 1 | `planC_baseline_frozen_s42` | none | none | none | 参照：冻结 baseline | 待跑 |
| 2 | `planC_baseline_frozen_s123` | none | none | none | 参照 | 待跑 |
| 3 | `planC_baseline_frozen_s2024` | none | none | none | 参照 | 待跑 |
| 4 | `planC_logit_ce_rlvr_s42` | logit | ce_delta | rlvr | 主实验：action-local CE reward | 待跑 |
| 5 | `planC_logit_ce_rlvr_s123` | logit | ce_delta | rlvr | 主实验 | 待跑 |
| 6 | `planC_logit_ce_rlvr_s2024` | logit | ce_delta | rlvr | 主实验 | 待跑 |
| 7 | `planC_logit_iou_rlvr_s42` | logit | iou_delta | rlvr | IoU reward 是否比 CE 更好 | 待跑 |
| 8 | `planC_logit_iou_rlvr_s123` | logit | iou_delta | rlvr | | 待跑 |
| 9 | `planC_logit_iou_rlvr_s2024` | logit | iou_delta | rlvr | | 待跑 |
| 10 | `planC_logit_boundary_rlvr_s42` | logit | boundary_delta | rlvr | 边界质量 reward | 待跑 |
| 11 | `planC_logit_boundary_rlvr_s123` | logit | boundary_delta | rlvr | | 待跑 |
| 12 | `planC_logit_boundary_rlvr_s2024` | logit | boundary_delta | rlvr | | 待跑 |
| 13 | `planC_logit_nc_rlvr_s42` | logit | nc_alignment_delta | rlvr | NC 几何对齐 reward | 待跑 |
| 14 | `planC_logit_nc_rlvr_s123` | logit | nc_alignment_delta | rlvr | | 待跑 |
| 15 | `planC_logit_nc_rlvr_s2024` | logit | nc_alignment_delta | rlvr | | 待跑 |
| 16 | `planC_logit_ce_nc_rlvr_s42` | logit | ce_delta + nc_alignment_delta | rlvr | 组合 reward | 待跑 |
| 17 | `planC_logit_ce_nc_rlvr_s123` | logit | ce_delta + nc_alignment_delta | rlvr | | 待跑 |
| 18 | `planC_logit_ce_nc_rlvr_s2024` | logit | ce_delta + nc_alignment_delta | rlvr | | 待跑 |
| 19 | `planC_logit_ce_dpo_s42` | logit | ce_delta | dpo | DPO 变体 | 待跑 |
| 20 | `planC_logit_ce_dpo_s123` | logit | ce_delta | dpo | | 待跑 |
| 21 | `planC_logit_ce_dpo_s2024` | logit | ce_delta | dpo | | 待跑 |
| 22 | `planC_feature_ce_rlvr_s42` | feature | ce_delta | rlvr | Feature-level action | 待跑 |
| 23 | `planC_feature_ce_rlvr_s123` | feature | ce_delta | rlvr | | 待跑 |
| 24 | `planC_feature_ce_rlvr_s2024` | feature | ce_delta | rlvr | | 待跑 |
| 25 | `planC_calib_ece_dpo_s42` | calibration | ece_delta | dpo | Calibration action | 待跑 |
| 26 | `planC_calib_ece_dpo_s123` | calibration | ece_delta | dpo | | 待跑 |
| 27 | `planC_calib_ece_dpo_s2024` | calibration | ece_delta | dpo | | 待跑 |
| 28 | `planC_logit_random_rlvr_s42` | logit | random | rlvr | 负对照：random reward | 待跑 |
| 29 | `planC_logit_random_rlvr_s123` | logit | random | rlvr | | 待跑 |
| 30 | `planC_logit_random_rlvr_s2024` | logit | random | rlvr | | 待跑 |
| 31 | `planC_logit_supervised_s42` | logit | CE loss | SGD | 容量对照：监督微调同一 head | 待跑 |
| 32 | `planC_logit_supervised_s123` | logit | CE loss | SGD | | 待跑 |
| 33 | `planC_logit_supervised_s2024` | logit | CE loss | SGD | | 待跑 |
| 34 | `planC_fullnet_rlvr_s42` | full network | ce_delta | rlvr | 负对照：端到端 RLVR | 待跑 |
| 35 | `planC_fullnet_rlvr_s123` | full network | ce_delta | rlvr | | 待跑 |
| 36 | `planC_fullnet_rlvr_s2024` | full network | ce_delta | rlvr | | 待跑 |

---

## 诊断指标（必须记录）

| 指标 | 说明 |
|---|---|
| mIoU / boundary IoU / pixel acc | 核心性能 |
| per-class IoU | 看哪类受益 |
| ECE | calibration action 必须看 |
| reward_delta | policy_reward - base_reward |
| policy/base KL | 防止偏离 baseline 太远 |
| NC1 | 类内方差 |
| NC2 | ETF 角度偏差 |
| NC3 | classifier weights vs class means 对齐 |
| NC4 | nearest-class-center accuracy |
| flatness_proxy | adapter 参数扰动后的 val loss 增量 |

---

## 实验优先级与阶段

### 第一阶段：基础设施 + smoke（~2 天）

1. 实现 Step 1-5 代码模块。
2. Smoke test（1 epoch，50 train images）：
   ```bash
   python -m spectral_detection_posttrain.trainers.segmentation.action_local_seg_rlvr \
     --config spectral_detection_posttrain/configs/seg_action_local_smoke.yaml \
     --checkpoint runs/round41_baseline_s42/checkpoint_last.pth \
     --run-name planC_logit_ce_smoke \
     --action logit --reward ce_delta --objective rlvr \
     --epochs 1 --limit-train 50
   ```
3. 通过 smoke 标准：不崩溃、train reward_delta > 0、val mIoU 在 baseline ±2% 内。

### 第二阶段：核心对照矩阵（~3-4 天，GPU2 空闲后）

按上表跑 36 个 run。为节约时间可先跑单 seed（s42）快速筛选，再对 promising 组合补 123/2024。

### 第三阶段：分析与下一步（~1 天）

1. 用 `scripts/analyze_planC_results.py` 汇总。
2. 判断退出条件：
   - 若 `planC_logit_ce_rlvr` 等主实验 mean mIoU ≥ baseline + 1% absolute → 进入论文重定位。
   - 若 action-local 也失败 → 项目变为纯负结果，需重新评估价值。

---

## 与本机其他方向的区分

- **M1 VOC 检测矩阵**：在远程 GPU2 进行中，不干扰。
- **LSG / FPN-SM 检测模块**：本 session 已将其判定为 Plan B / 历史分支；当前主线是 Plan C 语义分割 action-local RLVR。
- **AFM / in-network FFT**：按 AGENTS.md 视为历史分支，Plan C 不依赖这些模块。

---

## 备注

- 本文件依据 session `e2b780e4-7251-45d2-bdaa-3bddb609e869` 的 Plan C 生成。
- 此前生成的 `docs/reports/lsg_phase1_assignment.md` 是基于未找到 session 时的假设，已过时。
