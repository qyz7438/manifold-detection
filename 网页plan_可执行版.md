# 区域频域证据检测后训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个可运行、可验证的目标检测后训练实验工程，验证“预测框 ROI 与人工标注框 ROI 的频域幅度谱相似度”能否作为区域证据 verifier，并用于 reward-weighted post-training 提升局部 Patch 干扰下的检测可靠性。

**Architecture:** 第一版使用 Penn-Fudan Pedestrian + TorchVision Faster R-CNN。训练一个 baseline detector 后，生成预测框并与 GT 框按 class + IoU 匹配；对预测 ROI 和 GT ROI 裁剪、resize、FFT、提取 radial amplitude profile，计算 `R_amp`。先做 reward 判别力诊断，再做 image-level reward-weighted fine-tuning，最后评估 clean/patch 的 AP50、Recall、High-confidence FP 和 R_amp 分布。

**Tech Stack:** Python 3.10+, PyTorch, torchvision detection, matplotlib, PyYAML, pytest, tqdm, pandas.

---

## 0. 纠偏结论

原始 `网页plan.txt` 已经从“ViT 分类后训练”改成了“目标检测模型的区域频域证据可验证奖励后训练”。因此当前实现方向必须切换：

- 不再把 CIFAR-100 分类作为主线。
- 不再把 FFT 当成分类增强或多视图一致性主角。
- FFT 只出现在 **检测 ROI 证据验证阶段**。
- 项目第一道门不是 mAP 提升，而是：`R_amp` 能否区分 TP 和 FP。

## 1. 第一版成功标准

Smoke 必须完成：

- 下载或读取 Penn-Fudan Pedestrian。
- 使用 TorchVision Faster R-CNN 构建 person detector。
- 小样本训练 baseline 1 epoch，保存 checkpoint。
- 在验证集上匹配预测框和 GT 框，计算 ROI 频域 `R_amp`。
- 输出 TP/FP 的 `R_amp` 均值和 AUC。
- 加载 baseline 做 1 epoch reward-weighted post-training。
- 分别在 clean 和 patch 验证集上输出 AP50、Precision、Recall、High-confidence FP。
- 生成至少一张 ROI FFT 可视化或 R_amp 分布图。

## 2. 第一版文件结构

```text
spectral_detection_posttrain/
  __init__.py
  configs/
    smoke.yaml
    baseline.yaml
    posttrain.yaml
    eval_patch.yaml
  datasets/
    __init__.py
    penn_fudan.py
    patch_transform.py
  models/
    __init__.py
    build_detector.py
  matching/
    __init__.py
    box_iou.py
    pred_gt_matcher.py
  spectral/
    __init__.py
    roi_crop.py
    fft_features.py
    radial_profile.py
    spectral_reward.py
  train/
    __init__.py
    train_baseline.py
    posttrain_reward_weighted.py
  eval/
    __init__.py
    detection_metrics.py
    eval_detector.py
    eval_spectral_reward.py
  visualization/
    __init__.py
    visualize_roi_fft.py
  utils/
    __init__.py
    config.py
    io.py
    seed.py
tests/
  test_detection_matching.py
  test_spectral_reward.py
  test_detection_patch.py
```

## 3. Core Tasks

### Task 1: 数据集和 Patch 测试集

- [ ] 实现 `PennFudanDetectionDataset`，从 `PNGImages` 和 `PedMasks` 读取图像、mask、boxes、labels。
- [ ] 如果数据不存在，自动下载 `PennFudanPed.zip` 并解压。
- [ ] 实现 `build_penn_fudan_loaders(config, limit_train, limit_val)`。
- [ ] 实现 `add_detection_patch(image, target, placement, patch_type)`，支持 `background`、`object`、`edge`、`random`。

### Task 2: 检测模型

- [ ] 实现 `build_detector(config)`。
- [ ] 默认使用 `fasterrcnn_mobilenet_v3_large_320_fpn`。
- [ ] 加载 COCO 预训练权重，并将 ROI predictor 改为 2 类：background/person。
- [ ] 支持 smoke fallback：如果预训练权重下载失败，可通过配置切到 random init，但完整实验应使用 pretrained。

### Task 3: 匹配和检测指标

- [ ] 实现 `box_iou(boxes1, boxes2)`。
- [ ] 实现 `match_predictions_to_gt(prediction, target, iou_threshold, score_threshold)`。
- [ ] 实现 AP50、Precision、Recall、False Positive Rate、High-confidence FP、Miss Rate。

### Task 4: ROI 频域证据奖励

- [ ] 实现 `crop_and_resize_roi(image, box, size=128)`。
- [ ] 实现 `compute_fft_amplitude(roi)`：灰度化、Hann window、`torch.fft.fft2`、`fftshift`、`abs`、`log1p`、normalize。
- [ ] 实现 `radial_profile(amplitude, num_bins=32)`。
- [ ] 实现 `spectral_reward(roi_pred, roi_gt)`：`R_amp = exp(-cosine_distance(profile_pred, profile_gt))`。
- [ ] 实现 `compute_prediction_rewards(image, prediction, target)`，输出 TP/FP 的 `R_amp`。

### Task 5: Baseline 训练

- [ ] 实现 `python -m spectral_detection_posttrain.train.train_baseline`。
- [ ] 训练 Faster R-CNN baseline。
- [ ] 保存 `runs/<run_name>/checkpoint_last.pth`、`metrics_train.jsonl`、`config.yaml`。

### Task 6: Reward 判别力诊断

- [ ] 实现 `python -m spectral_detection_posttrain.eval.eval_spectral_reward`。
- [ ] 对 baseline predictions 做 TP/FP 匹配。
- [ ] 输出 `mean_r_amp_tp`、`mean_r_amp_fp`、`auc_tp_vs_fp`、样本数量。
- [ ] 生成 `r_amp_distribution.png`。

### Task 7: Reward-weighted 后训练

- [ ] 实现 `python -m spectral_detection_posttrain.train.posttrain_reward_weighted`。
- [ ] 加载 baseline。
- [ ] 先用当前模型预测框，计算 image-level spectral reward。
- [ ] 正常计算 Faster R-CNN detection loss。
- [ ] 使用 `L_post = image_weight * L_det`，其中 `image_weight = 1 + lambda * (1 - image_reward)`。
- [ ] 第一版支持冻结 backbone，仅更新 detection head/box head。

### Task 8: Clean/Patch 评估

- [ ] 实现 `python -m spectral_detection_posttrain.eval.eval_detector`。
- [ ] 支持 `--patch-mode none/background/object/edge/random`。
- [ ] 输出 `ap50`、`precision`、`recall`、`false_positive_rate`、`high_conf_fp_rate`、`miss_rate`。

### Task 9: 可视化

- [ ] 实现 `python -m spectral_detection_posttrain.visualization.visualize_roi_fft`。
- [ ] 展示 image、pred ROI、GT ROI、pred FFT amplitude、GT FFT amplitude。
- [ ] 如果没有匹配框，输出说明文件而不是崩溃。

## 4. Smoke 命令

`run_smoke.bat` 应执行：

```bat
python -m pytest tests -q
python -m spectral_detection_posttrain.train.train_baseline --config spectral_detection_posttrain/configs/smoke.yaml --run-name det_smoke_baseline --limit-train 8 --limit-val 4 --epochs 1
python -m spectral_detection_posttrain.eval.eval_spectral_reward --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_baseline/checkpoint_last.pth --run-name det_smoke_reward --limit-val 4
python -m spectral_detection_posttrain.train.posttrain_reward_weighted --config spectral_detection_posttrain/configs/smoke.yaml --baseline runs/det_smoke_baseline/checkpoint_last.pth --run-name det_smoke_posttrain --limit-train 4 --limit-val 4 --epochs 1
python -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_posttrain/checkpoint_last.pth --run-name det_smoke_eval_clean --limit-val 4 --patch-mode none
python -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_posttrain/checkpoint_last.pth --run-name det_smoke_eval_patch --limit-val 4 --patch-mode random
python -m spectral_detection_posttrain.visualization.visualize_roi_fft --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_posttrain/checkpoint_last.pth --run-name det_smoke_roi_fft --limit-val 4
```

## 5. 第一版解释口径

第一版可以声称：

- 已实现目标检测 ROI 频域证据 verifier。
- 已能诊断 `R_amp` 对 TP/FP 的判别力。
- 已实现 image-level reward-weighted post-training。
- 已能比较 clean 与 patch 场景下的检测指标。

第一版不能声称：

- 已证明 mAP 大幅提升。
- 已实现 ROI-level loss weighting。
- 已完成 DETR/Hungarian matching。
- 已证明相位谱有效。
- 已在 COCO/VOC 上验证泛化。

## 6. 后续扩展

只有当 `R_amp(TP) > R_amp(FP)` 或 AUC 明显高于 0.5 时，再进入：

1. ROI-level reward-weighted loss。
2. Reward head：`ROI feature + spectral feature -> quality score`。
3. VOC/COCO 子集。
4. 相位/边缘结构谱。
5. DETR + Hungarian matching。
