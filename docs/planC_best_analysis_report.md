# Plan C 最优结果分析（修正版）

## 背景

此前在 Penn-Fudan 行人分割上的 Plan C 后续实验里，两个候选最优结果被认为来自“冻结 backbone + 轻量 adapter”：

| Run | 配置 | 报告 best mIoU | vs baseline |
| --- | --- | --- | --- |
| `planC_followup_e2e_sup_logit_3ep_ref` | `action=logit`, `unfreeze_baseline=True` | 0.5877 | +0.0361 |
| `planC_followup_sup_feature_scale10_3ep` | `action=feature`, `unfreeze_baseline=False`, `scale=1.0` | 0.5874 | +0.0358 |

baseline mIoU：0.5516（`runs/round41_baseline_s42`，FCN-ResNet50 from-scratch 3 epoch）。

## 发现的问题：冻结 backbone 时 BatchNorm 统计量发生漂移

旧版 `action_local_adapter.py` 在训练循环中只调用 `policy.train()`，**没有强制让冻结的 backbone 保持 `eval()` 模式**。虽然 backbone 的卷积权重被 `requires_grad=False` 冻结，但其 BatchNorm 层的 running mean / running var 仍在 `train()` 模式下随 forward 更新。这意味着旧版所谓“冻结 backbone + adapter”实际上在悄悄微调 backbone 的 BN 统计量，从而产生了虚假的 mIoU 提升。

修复方式：当 `unfreeze_baseline=False` 时，在每个 epoch 开始显式调用 `policy.base_model.eval()`。

```python
policy.train()
if not args.unfreeze_baseline:
    policy.base_model.eval()
```

## 修正后的重跑结果

| Run | 配置 | 修复后的 best mIoU | vs baseline | 备注 |
| --- | --- | --- | --- | --- |
| `planC_adapter_logit_truefrozen` | `action=logit`, frozen backbone（真冻结） | 0.5552 | **+0.0036** | 稳定，但提升极小 |
| `planC_adapter_feature_scale10_truefrozen` | `action=feature`, frozen backbone（真冻结） | 0.5516 | **0** | 与 baseline 持平，epoch 3 甚至跌至 0.5286 |
| `planC_adapter_e2e_sup_logit_repro` | `action=logit`, `unfreeze_baseline=True` | 0.5772 | **+0.0256** | 端到端微调 backbone + logit adapter，可复现 |
| `planC_abl_backbone_only_3ep` | `action=none`, unfreeze backbone, freeze head | **0.5985** | **+0.0469** | 仅微调 backbone，无 adapter |
| `planC_abl_full_finetune_3ep` | `action=none`, unfreeze 全部参数 | **0.6246** | **+0.0730** | 完整 fine-tune |
| `planC_abl_feature_adapter_e2e_3ep` | `action=feature`, unfreeze backbone, freeze head | 0.5877 (best) / 0.5633 (final) | +0.0361 / +0.0117 | 不稳定 |

关键结论：
1. **真正“冻结 backbone + 轻量 adapter”的 post-training 在 Penn-Fudan 上几乎无效。**
2. **原先 feature adapter scale=1.0 的 +3.6 pp 提升完全是 BN drift 造成的 artifacts。**
3. **adapter 在解冻 backbone 时也没有独立价值**：backbone-only fine-tune（0.5985）明显优于 logit adapter（0.5772）和 feature adapter（0.5877 best / 0.5633 final）。
4. **有效提升主要来自 backbone 的继续训练**：完整 fine-tune 达到 0.6246，是当前设置下的上限。
5. 由于 `action=logit` 的实现中 classifier head 仍通过 `no_grad()` 冻结，因此 `e2e_sup_logit` 的实际训练对象是 **整个 backbone + 一个很小的 logit adapter**，而不是 classifier head。

## 对论文框架的冲击

原 `docs/planC_paper_outline.md` 的核心卖点是：

> “在冻结的分割主干上插入一个零初始化的小 adapter……feature adapter 将 mIoU 从 55.2% 提升至 58.7%，只引入极少可训练参数、无灾难性遗忘。”

这一说法**不再成立**。修正后的数据表明：
- 冻结 backbone 的 feature adapter 无法超过 baseline；
- 唯一显著的改进来自 backbone fine-tuning，已经不再是“parameter-efficient post-training”。

因此论文框架需要调整。可行的方向：

### 方向 A：转向“受控实证研究 + Head-Frozen Backbone Adaptation”
将论文重新定位为 **“重新审视分割后训练：adapter、backbone 还是 BatchNorm drift？”**。提出严格的冻结 backbone 协议，揭示 BN drift，并将 **Head-Frozen Backbone Adaptation (HFBA)**——即冻结分类头、仅微调 backbone——作为简单且强的后训练基线。这是目前数据支持最强的方向。

### 方向 B：接受为 negative result，强调 BN drift 的教训
论文主题改为 **“语义分割轻量后训练：adapter 真的够吗？”**。贡献包括：
- 系统比较 logit / feature / calibration 三种动作空间；
- 揭示 BN drift 会让“冻结 backbone”实验产生虚假增益；
- 证明在弱基线（from-scratch FCN-ResNet50）上，有意义的提升仍需 backbone 更新。

### 方向 C：提升基线后再验证
使用 ImageNet 预训练 backbone 或更充分训练的 baseline，重新检验 adapter-only post-training 是否有效。如果有效，论文可以回归“冻结 backbone + adapter”叙事，但需明确 baseline 强度。

## 代码改动

- `spectral_detection_posttrain/trainers/segmentation/action_local_adapter.py`：
  - 冻结 backbone 时强制 `base_model.eval()`；
  - `unfreeze_baseline=True` 时保存完整 `policy.state_dict()`，否则只保存 adapter。
- `scripts/run_planC_adapter_grid.py`：
  - 增加 true-frozen 对照组与 e2e 复现实验。

## 建议下一步

1. 与用户确认论文方向（A/B/C）。
2. 若选 A 或 B，立即更新 `docs/planC_paper_outline.md` 的 Abstract、Key Findings 与 Contributions。
3. 若选 C，先训练一个更强的 baseline（ImageNet pretrain 或更长 epoch），再跑冻结 adapter 实验。
