# NWPU Coordinated Low-Energy Action Report

日期：2026-07-10

分支：`codex/energy-guided-roi-transport`

研究问题：在类别条件终点未知时，能否从 detector 可见的 ROI 信息中，学习一组经过原生阈值、匹配和 NMS 后仍有净收益的协调低能量动作？

## 1. 结论摘要

这轮实验把两个问题分开了：

1. **动作解是否存在？存在。** M0 GT-oracle 在同一冻结 detector 和同一 196 图 full-val 上，将 AP75 从 `0.313280` 提升到 `0.479998`，且 AP50、FPR、预测数都通过预注册安全门。
2. **当前网络能否从 detector-only 输入学到这些动作？基本不能。** M1 只把 AP75 提升 `0.000200`，未达到 `+0.002` 门；取消 move gate 后 AP75 反而下降 `0.003149`。

因此：

- 保留“原生后处理后的协调动作空间确有可达收益”这一结论。
- 不恢复“当前 M1 已形成有效检测方法”的主张。
- 放弃继续微调当前 `73-way local direction + independent move gate + fixed L2 energy bias` 结构。
- 固定 L2 action energy 在本轮几乎没有独立贡献，不能继续称为已验证的低能量机制。
- 下一结构应直接预测候选动作在集合上下文和原生 NMS 后的**联合边际效用**，而不是分别学习局部方向与 move/stop 后再取交集。

## 2. 分级 Profile 分工

本轮没有把全部任务交给 5.6-sol：

- `gpt-5.6-luna`：远程/仓库审计、runner 和诊断脚本实现。
- `gpt-5.6-terra`：M0 门设计、纯搜索/策略模块、阻断式代码与科学审查。
- `deepseek-v4-pro`：M0、M1 和 bypass 结果的只读外部复核。
- `gpt-5.6-sol`：研究边界、集成审查、TDD 安全修复、远程实验裁决和最终综合。

## 3. 锁定输入与复现边界

| 项目 | 值 |
|---|---|
| Detector | `fasterrcnn_mobilenet_v3_large_320_fpn` |
| Detector checkpoint SHA256 | `de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027` |
| Annotation SHA256 | `dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e` |
| Train split | 454 images, `7abe3c...cd9bd` |
| Validation split | 196 images, `49f05c...9684` |
| Native postprocess | score `0.05`, NMS `0.5`, top `100` |
| Candidate actions | 73 box deltas, steps `0.05/0.10/0.20` |
| Action budget | 4 per image |
| GPU | remote physical GPU2 only |

所有 full-val 结果均为 batch size 1 的 native-parity 路径。它们应与同路径 identity 比较，不应混用历史 batch size 16 指标。

## 4. M0: GT-Pruned Set Oracle

版本：`det.energy.set_oracle.m0.001`

结果 SHA256：`372d872ccaebea8418c2ff4428e56a666d4e4440a77b16acbd127f4999869cda`

### 4.1 Detector 指标

| Mode | AP50 | AP75 | Precision | Recall | FPR | ECE | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity | 0.663925 | 0.313280 | 0.586153 | 0.715658 | 0.413847 | 0.110722 | 1271 |
| local | 0.672068 | 0.397680 | 0.585404 | 0.724304 | 0.414596 | 0.110678 | 1288 |
| set_greedy | 0.673504 | 0.478531 | 0.588785 | 0.726225 | 0.411215 | 0.110568 | 1284 |
| set_beam | 0.673674 | 0.479998 | 0.588785 | 0.726225 | 0.411215 | 0.110600 | 1284 |
| delta_permuted_beam | 0.665096 | 0.386440 | 0.586207 | 0.718540 | 0.413793 | 0.113131 | 1276 |

`set_beam - identity`：

- AP75 `+0.166718`
- AP50 `+0.009749`
- FPR `-0.002632`
- predictions `+1.023%`

### 4.2 预注册门

| Gate | 结果 | 关键值 |
|---|---|---|
| G0 strict native parity | PASS | 0 mismatched images; box/score error 0 |
| G1 set problem exists | PASS | 174/196 images have >=2 candidates |
| G2 coordination | PASS | beam-local sum 115.68; 180 improved images; bootstrap CI `[0.473, 0.714]` |
| G3 NMS-specific | PASS | native/NMS-off uplift ratio 6.488 |
| G4 location alignment | PASS | beam-permuted bootstrap CI `[0.406, 0.648]` |
| G5 detector headroom | PASS | all AP/FPR/count constraints passed |

### 4.3 M0 能说什么

M0 证明的是：在 `GT-pruned K12 + B4 beam + 固定 utility` 条件下，位置正确的动作集合在原生 NMS 后存在较大的可达收益。

M0 不能证明：

- detector-only 特征包含足够信息来预测这些动作；
- beam 找到了完整 73-action 空间的全局最优；
- 该 oracle 数字是可部署模型性能；
- 固定 L2 action energy 是收益来源。

## 5. M1: GT-Free Validation Set Policy

版本：`det.energy.set_policy.m1.001`

代码提交：`bf8574970412ed111341ebe8d7532d75d7a740df`

结果 SHA256：`9b089b6625d0dfc9d9fe6a1553be6d5646b78a8179c7de15e506b2e0215a88e5`

Policy checkpoint SHA256：`54a34c463ea5ed2fea58e7fa9534fc059867d2b7cf1444b32255aadb9055d9cd`

### 5.1 训练监督

- Train-only oracle cache：454 images，28,405 proposals。
- `256x7x7` spatial ROI features 以 fp16 保存，约 712.6 MB。
- 3,155 个 local action direction labels。
- 335 个 beam-selected move labels。
- Validation extractor 明确使用 `targets=None`；GT 只用于最终计分。

### 5.2 训练曲线

| Epoch | Total | Action | Move | Action Acc | Move Acc | Selected Action Acc |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.4173 | 2.9728 | 0.4444 | 0.7190 | 0.9279 | 0.6485 |
| 2 | 2.9316 | 2.5500 | 0.3816 | 0.6816 | 0.9247 | 0.6606 |
| 3 | 2.7438 | 2.3886 | 0.3551 | 0.6438 | 0.9090 | 0.6817 |
| 4 | 2.5460 | 2.2289 | 0.3172 | 0.6300 | 0.9173 | 0.6795 |

总体 action accuracy 下降而 selected-action accuracy 上升，说明模型开始偏向少量正动作，但约 32% 的 selected direction 在训练集上仍不正确。

### 5.3 Full-Val 指标

| Mode | AP50 | AP75 | Precision | Recall | FPR | ECE | Predictions | Actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 0.663925 | 0.313280 | 0.586153 | 0.715658 | 0.413847 | 0.110722 | 1271 | 0 |
| learned_set | 0.662862 | 0.313480 | 0.585557 | 0.716619 | 0.414443 | 0.113813 | 1274 | 88 |
| energy_zero | 0.661073 | 0.313476 | 0.583987 | 0.714697 | 0.416013 | 0.115587 | 1274 | 89 |
| shuffled_spatial | 0.661754 | 0.305350 | 0.582229 | 0.717579 | 0.417771 | 0.116388 | 1283 | 226 |

`learned_set - identity`：

- AP75 `+0.000200`
- AP50 `-0.001063`
- FPR `+0.000595`
- predictions `+0.236%`

核心 G2 失败，因为 AP75 未达到 `+0.002`。其余安全量没有失控，因此这不是灾难性结构崩坏，也不是有意义的安全-收益交换；它是几乎没有取得 oracle headroom。

空间特征置乱后 AP75 比 learned 低 `0.008130`，表明对应空间信息确实影响选择；但 learned 自身没有形成足够净收益。

`learned_set - energy_zero` 的 AP75 仅 `+0.000003`，固定 L2 energy 项在当前结构中近似无效。

## 6. 单臂 Move-Gate Bypass 诊断

版本：`det.energy.set_policy.movegate_diag.001`

结果 SHA256：`83aeb33864b7fa918d2f5703fd4ad599f57d00800fd3c1590381b5ca62bed9fb`

该实验冻结原 policy，只移除 `move_logit > 0` 硬门；action margin、`move_logit + margin` 排序、top-4、energy、score/NMS 全部不变。

| Mode | AP50 | AP75 | Precision | Recall | FPR | ECE | Predictions | Actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 0.663925 | 0.313280 | 0.586153 | 0.715658 | 0.413847 | 0.110722 | 1271 | 0 |
| learned_set | 0.662862 | 0.313480 | 0.585557 | 0.716619 | 0.414443 | 0.113813 | 1274 | 88 |
| move_gate_bypass | 0.665130 | 0.310131 | 0.584567 | 0.720461 | 0.415433 | 0.116656 | 1283 | 310 |

`bypass - identity`：

- AP50 `+0.001205`
- recall `+0.004803`
- AP75 `-0.003149`
- FPR `+0.001585`
- predictions `+0.944%`

动作数增加 3.5 倍后，AP50/recall 小幅上升而 AP75 明显下降。新增动作不是单纯制造大量预测或 FPR 爆炸，而是偏向粗定位命中，不能提供严格 IoU 定位收益。

因此 move gate 不是主要瓶颈；它原本在抑制方向/联合排序产生的 AP75 伤害。

## 7. Codex 判断

1. **M0 成功，M1 失败。** “解存在”和“当前网络能学到解”必须分开报告。
2. **不是只优化了某指标导致其他指标轻微下降的成功模型。** M1 AP75 只有噪声级正增益；bypass 虽改善 AP50/recall，却损害目标 AP75。
3. **当前 energy 不是有效信息源。** 它只是候选 delta 的固定 L2 logit bias，消融影响约为零。
4. **当前两头分解不再继续。** local action CE 与独立 move BCE 的交集没有对齐 post-NMS marginal utility。
5. **空间 ROI 信息不是纯噪声。** shuffled control 明显更差，因此不能简单断言 `256x7x7` 没有信息；问题是现有监督与排序没有把信息变成正确 AP75 动作。

## 8. DeepSeek 复核

DeepSeek 同意以下窄结论：

- move gate 不是主要瓶颈；取消它后 310 个动作仍未恢复 AP75。
- 当前 action ranking 更相关于 AP50/recall，而不是严格定位精度。
- 固定 L2 energy 项是近似空因子。
- 不应继续通过阈值或预算 sweep 宣称改进。

需要限定 DeepSeek 的一个过强说法：M0 并未证明动作对 detector-only 特征“原则上可学习”；它只证明 GT 条件下存在动作。shuffled 对照说明空间对应信息被模型使用，但还不能证明该信息足以恢复 oracle 方向。

## 9. 综合结论

当前研究状态不是“流形低能量动作已经失败”，而是：

```text
post-NMS action reachability: supported
current detector-only policy distillation: not supported
current fixed L2 energy mechanism: not supported
current independent direction + move-gate factorization: retired
```

下一版若继续，只保留一个核心目标：

```text
predict joint marginal utility DeltaU(proposal, action, set context)
```

训练目标必须直接对应原生 threshold/NMS 后的 AP75 边界、FP 代价、动作能量和 abstention；推理时用同一个标量完成方向、move/stop 和集合排序，避免当前两个 head 的目标错配。

在实现下一版前，先在 train split 内做 image-group-held-out candidate-utility probe，验证 detector-only 输入对真实 marginal utility 是否具有跨图泛化；该 probe 不使用 validation 阈值 sweep，也不产生 detector gain 主张。
