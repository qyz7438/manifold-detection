# Plan C Cleanup Note

## 中心思想对齐

Plan C 的论文方法已确定为 **Action-Local Adapter Post-Training (ALAPT)**：
- 在冻结的分割基线上插入轻量 adapter；
- 通过选择动作空间（logit / feature / calibration）进行局部修正；
- 使用标准 supervised CE 训练；
- 传统 RLVR / SPSA / DPO 路线不再作为核心方法。

## 清理动作

为保持代码与方法论述一致，已将以下 RLVR/SPSA/DPO 相关文件移入 `legacy/` 目录：

### Segmentation trainer
- `spectral_detection_posttrain/trainers/segmentation/action_local_seg_rlvr.py`
  → `spectral_detection_posttrain/trainers/segmentation/legacy/action_local_seg_rlvr.py`

### Reward functions (only used by RLVR/DPO/SPSA)
- `spectral_detection_posttrain/methods/segmentation/action_rewards.py`
  → `spectral_detection_posttrain/methods/segmentation/legacy/action_rewards.py`

### Grid scripts (contained RLVR/DPO/SPSA experiments)
- `scripts/run_planC_grid.py` → `scripts/legacy/run_planC_grid.py`
- `scripts/run_planC_followup_grid.py` → `scripts/legacy/run_planC_followup_grid.py`

## 新增/保留文件

- `spectral_detection_posttrain/trainers/segmentation/action_local_adapter.py`
  - 新的主 trainer，仅支持 supervised CE；
  - 保留 `--action {logit,feature,calibration}` 和 `--unfreeze-baseline` 用于消融；
  - 可选 NC/flatness 诊断。

- `scripts/run_planC_adapter_grid.py`
  - 新的 supervised-only grid，覆盖 logit/calibration/feature-scale-sweep/e2e 变体。

- `spectral_detection_posttrain/core/models/seg_adapters.py`
  - 保留 `LogitCorrectionAdapter`、`FeatureRefinementAdapter`、`CalibrationAdapter`。

## 未清理部分

项目其它模块（如 `spectral_detection_posttrain/trainers/detection/`、`scripts/round21xx_*.py` 等）仍包含历史 RLVR/DPO 检测实验代码。这些属于项目早期资产，不在 Plan C 论文主线内，因此未做迁移。如果需要进一步统一，可另行处理。
