# M3 / M4 实验矩阵总结

## M3：COCO 17 smoke run（500 train / 200 val，1 epoch，seed 42）

> 注：smoke run 仅用于快速验证 pipeline，1 epoch 结果不代表真实性能。

| Run | AP50 | AP75 | 备注 |
|-----|------|------|------|
| `coco_mob_baseline_smoke` | 0.1408 | 0.0405 | lr 从 0.024 调至 0.004 后无 NaN |
| `coco_mob_fpn_sm_smoke` | 0.0079 | 0.0001 | 1 epoch 下 FPN-SM 未学到有效信号 |

结论：pipeline 可跑通；1 epoch smoke 不足以判断 FPN-SM 在 COCO 上的效果，如需结论需正式训练。

## M4：NWPU VHR-10 baseline + FPN-SM（3 seeds，12 epochs）

| 模型 | 方法 | AP50（mean ± std） | AP75（mean ± std） | ΔAP50 vs baseline | ΔAP75 vs baseline |
|------|------|---------------------|---------------------|--------------------|--------------------|
| MobileNet | baseline | 0.5165 ± 0.0164 | 0.1495 ± 0.0201 | — | — |
| MobileNet | FPN-SM | 0.5521 ± 0.0174 | 0.1852 ± 0.0236 | **+0.0356** | **+0.0357** |
| ResNet50 | baseline | 0.9204 ± 0.0060 | 0.8167 ± 0.0077 | — | — |
| ResNet50 | FPN-SM | 0.9274 ± 0.0089 | 0.7555 ± 0.0791 | +0.0069 | **−0.0612** |

各 seed 原始值：

| 模型 | 方法 | s42 | s2024 | s999 |
|------|------|------|-------|------|
| MobileNet | baseline | 0.4975 | 0.5259 | 0.5260 |
| MobileNet | FPN-SM | 0.5589 | 0.5323 | 0.5650 |
| ResNet50 | baseline | 0.9176 | 0.9273 | 0.9163 |
| ResNet50 | FPN-SM | 0.9324 | 0.9326 | 0.9170 |

## 结论

1. **MobileNet + FPN-SM 在 NWPU 上稳定提升 baseline**，AP50 约 +0.036，AP75 约 +0.036，三种子一致。
2. **ResNet50 + FPN-SM 在 NWPU 上基本无提升**：AP50 仅 +0.0069，AP75 因 seed 2024 异常低（0.6671）而平均下降 0.0612。ResNet50 baseline 已接近 0.92 AP50，提升空间小。
3. **COCO smoke 仅验证 pipeline**：lr 0.024 导致 NaN，调至 0.004 后正常完成，但 1 epoch 结果无意义。
4. 所有 6 个 M3/M4 run 均正常完成，无 NaN / OOM / failed_nan。

## 下一步建议

- NWPU ResNet50 FPN-SM seed 2024 的 AP75（0.6671）明显低于其他两种子（~0.78/0.82），可单独复现该 seed 排查是否为随机波动。
- 如需 COCO 结论，应进行完整训练（而非 smoke run）。
