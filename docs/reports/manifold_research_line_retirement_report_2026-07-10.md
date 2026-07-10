# Manifold Detection 研究路线取舍完整报告

日期：2026-07-10
审计分支：`codex/energy-guided-roi-transport`
审计提交：`1ba4d70ba5633a99966b27d3c2bab44216168355`
项目范围：NWPU/VOC/Penn-Fudan 目标检测中的谱域、流形、RLVR 与 ROI transport 研究

## 1. 执行结论

我们没有放弃“低能量流形校正”这个母问题，但已经放弃了以下等价假设：

```text
更低的本征维度
  = 更好的检测特征
  = 更接近类别原型
  = 更低的手工能量
  = 更好的局部 IoU
  = 更高的最终 AP
```

实验表明，这些等号都不能直接成立。检测器真正的终点不是一个已知类别原型，也不是单个 proposal 的局部最优点，而是经过分类、bbox 回归、阈值、匹配、NMS 和 top-K 后仍然成立的集合级检测状态。

当前总决策是：

1. 停止 external FFT reward、旧 RLVR/GRPO/DPO、直接 prototype attraction、HWM teacher、连续 residual direction field 和 73-way hard one-hot ranking。
2. 撤回所有依赖弱 12-epoch baseline 或 legacy custom postprocessor 的 action-local 增益解释。
3. 停止扩展当前 proposal-independent dense candidate head，包括 seeds 2024/999、网络加深和在同一 full-val 上继续调 margin/epoch。
4. AFM、FPN-SM、PrototypeBank、Sinkhorn、dual-energy 和 verifier 特征不删除，但降级为架构基线、诊断或可复用原语。
5. 若继续研究，问题边界必须改成 set-level、NMS-aware 的协调动作选择，或使用 detector-native regression suggestions 作为候选。

一句话概括：**被放弃的是“用局部代理量直接定义已知终点”的路线；尚未被否证的是“在未知终点下，以低能量约束集合级动作”的路线。**

## 2. 项目边界

本报告只审计 `manifold-detection` 的目标检测路线，不把 `RLimage` 的语义分割 Plan C、分类 MFVPT 或其他任务的结果混入因果结论。

仓库中部分历史文档和代码来自更早的 `RLimage` 工作区。它们可以说明设计来源和失败模式，但不能自动作为当前 manifold 项目的验证结果。尤其是：

- `docs/current_status_full_report.md` 明确记录的是旧 `E:/CLIproject/RLimage` 工作目录；
- `docs/planC_*` 属于分割路线；
- `mfvpt/` 属于历史分类路线；
- 对抗补丁防御是独立任务线，目前未成功，但不属于本次 ROI transport 停线判断。

## 3. 状态定义与证据等级

### 3.1 状态定义

| 状态 | 含义 |
|---|---|
| **硬停止** | 不再对同一假设、目标和评价路径继续调参或扩模型。重新启动必须引入新的因果变量。 |
| **证据作废** | 历史数值受到 bug、弱基线、评估污染或 parity 失败影响，不能再用于方法主张。 |
| **降级保留** | 不再作为论文主线，但代码和结果保留为架构基线、诊断、对照或组件。 |
| **边界停止** | 当前 formulation 停止；更广泛问题仍开放，但必须改变输入、动作、目标或集合结构。 |

### 3.2 证据等级

本报告按以下顺序裁决冲突：

1. 强基线、native postprocess、full-val、严格 parity、对照完整的实验；
2. full-val 多种子 matched-budget attribution control；
3. full-val 单种子并含 shuffle/context/oracle 控制；
4. smoke、离线 cache、synthetic 或 oracle 诊断；
5. 后来被 bug audit、弱基线或评估污染推翻的历史结果。

因此，2026-07-10 的 strong-native 结果优先于 2026-07-09 的 HWM 报告，也优先于更早的 smoke、旧后处理和 12-epoch baseline 结果。

## 4. 路线总表

| 路线 | 原始主张 | 最关键证据 | 当前状态 | 保留内容 |
|---|---|---|---|---|
| External FFT reward | 手工幅度/相位/结构分数能提供因果 reward | real 与 shuffled 多次不可分，VOC 上 shuffled 略高 | **硬停止** | 离线特征、shuffle protocol |
| Score rescue / RLVR / GRPO / DPO | reward/preference 能独立提升 detector AP | 同预算 box-head/full-model fine-tune 显著更强；DPO 可高 pair accuracy 但伤 AP | **硬停止为主线** | 稳定训练壳、KL、pair validity、预算约束 |
| 旧 AFM | ROI 内 FFT 可直接带来 AP75 增益 | 幅相 scale 为零，收益来自 box-head 继续训练 | **证据作废/硬停止旧实现** | 失败诊断 |
| MPLSeg AFM / FPN-SM | in-network 频域适配可改善定位 | PF、VOC ResNet、NWPU MobileNet 有正证据；跨骨干/数据集不稳定 | **降级保留** | 架构基线、频域组件 |
| Prototype/ETF/PAH endpoint | 类别原型或 simplex 是检测终点 | prototype-local 信号不增加逐候选信息；PAH 当前形式伤 AP/ECE | **硬停止为已知终点** | PrototypeBank、anchor、诊断 |
| PG-AW / Plan A / Plan B | 特征几何约束可保护或提升 AP75 | 几何指标改善但 AP75 下降；Plan A/B 未恢复 | **硬停止** | error-mode taxonomy、bbox preserve 思想 |
| Intrinsic-dimension optimization | 更低 ID 的 ROI 表示更适合校正 | 无提交的 layerwise 数值可复核；低 ID 未转化为动作/AP 信息 | **降级为诊断** | ID/effective-rank/NC1 日志 |
| Dual-energy / cone endpoint | 类内紧凑与类间有序可直接形成 action signal | group 结构可分，但逐候选 AUC/AP 不超过 detector score 或 shuffle | **硬停止为 scorer/loss** | group-level structure diagnostics |
| HWM teacher + energy | 历史最优 epoch 能指导低能量动作 | box-only 解释约 85% 增益，HWM loss 近似死项 | **硬停止** | high-IoU preservation |
| Continuous residual field | 小 bbox/score residual 可形成有效方向场 | strong-native 下所有 scale 低于 identity | **硬停止** | bounded residual API、identity init |
| 73-way hard ranking | 把局部动作当分类可学出下一步 | loss 约等于 `ln(73)`，大多数动作有害 | **硬停止该 objective** | 73 候选动作空间 |
| Dense candidate gain | 连续相对增益可学出低能量动作 | 学到局部 IoU sign，但 AP75 增益未过 gate，context 解释大部分收益 | **边界停止** | dense relative gain、abstention、候选 reachability |
| Legacy action postprocess | 历史 precision/FPR/ECE 改善来自 learned action | 零动作本身就改变预测数与指标 | **证据作废** | native parity harness |

## 5. 详细路线审计

### 5.1 External FFT reward：停止把频谱摘要当外部监督

原始思路是从 ROI crop 或预测框/GT 的幅度、相位、结构差异中构造 `R_amp` 等 verifier reward，再用 RLVR 更新检测器。

关键结果：

- 最初 `R_amp` 可以离线区分 TP/FP，但粗粒度训练把 AP50 从约 `0.8630` 降到 `0.4349`，Recall 从 `0.8791` 降到 `0.4615`；
- 修复 signed objective、冻结 baseline、KL 和训练状态后，RLVR shell 可以保持稳定；
- 但 Penn-Fudan 的 real amp、shuffled amp 和 structure controls 相近；
- VOC 上 spatial+spectral 为 AP50/AP75 `0.7724/0.3751`，shuffled spectral 反而为 `0.7742/0.3765`；
- 后续 raw-iFFT/high-dimensional fusion 离线排名更好，但在线训练只产生约 `+0.0029` AP75，未超过更简单的 raw-iFFT control。

裁决：

- **硬停止**“手工 external FFT 是因果 reward”的主张；
- 不再继续扩大频谱特征、band、phase、radius 或融合维度；
- 保留 raw-iFFT/FFT 特征作为离线 ranker、diagnostic 和 shuffle control。

主要证据：`docs/round2x_thinking_and_results.md`、`docs/round2200_2203_large_scale_manifold_fusion_summary.md`。

### 5.2 RLVR、GRPO、DPO 与 score rescue：停止作为独立 AP 主线

这条线并非完全没有信号，而是信息密度明显低于普通监督训练。

历史信号：

- Oracle additive rescue 曾把旧 NWPU baseline AP75 从约 `0.2939` 提到 `0.3459`，即 `+0.0520`；
- 但预测数从 `1425` 增至 `19811`，FPR 从 `0.478` 增至 `0.960`，AP50 下降约 `0.024`；
- GRPO smoke 只移动约 `+0.0007` AP75；
- DPO 可以达到很高 pair accuracy，但 AP75 下降，预测数和 FPR 暴涨；
- 5-epoch DPO 单种子最好约 `+0.005` AP75，但整体 LC-HI pool 没有同步改善。

matched-budget 归因控制进一步关闭了主线：

| 方法 | 三种子平均 AP75 delta |
|---|---:|
| `rlvr_random_fulltrain_10ep` | `+0.0350` |
| `box_head_only_10ep` | `+0.0792` |
| `full_model_10ep` | `+0.1058` |

因此，旧 RLVR/DPO 增益主要不能超过 detector 自身继续训练。spectral context 也不优于 random/zero context。

裁决：

- **硬停止** RLVR/GRPO/DPO 作为当前独立检测提升方法；
- **降级保留**它们作为未来 structured outcome 的训练机制；
- 保留 signed objective、KL/reference anchor、valid pair mask、absolute threshold constraints、rescue budget 和 false-positive guardrails。

主要证据：`docs/reports/boxrlvr_direct_matrix_review_2026-07-09.md`、`docs/reports/boxhead_finetune_10ep_control_2026-07-09.md`、`docs/reports/full_model_finetune_10ep_control_2026-07-09.md`。

### 5.3 AFM 与 FPN-SM：退出主线，不是全家族否证

必须区分三个结论：

1. 旧 scale-gated AFM 的正结果作废。其幅相 scale 保持零，表面 AP75 增益来自残差或 box-head 继续训练。
2. 修复梯度拓扑后的 MPLSeg-style AFM 在 Penn-Fudan 上确有定位收益，但属于架构 fine-tuning，不是低能量 transport 或 RLVR 结果。
3. FPN-SM 有混合但真实的正负证据，不能简单判死。

FPN-SM 关键结果：

- VOC ResNet50 三种子：AP50 `+0.0492`、AP75 `+0.0707`，对应 `p=0.0219/0.0073`；
- NWPU MobileNet：约 AP50 `+0.0356`、AP75 `+0.0357`；
- NWPU ResNet50：AP50 仅 `+0.0069`，AP75 `-0.0612`，且方差较大；
- NWPU MobileNet 早期配置中 ECE 从约 `0.042` 恶化到 `0.110`；
- 480x800 设置存在 seed 发散。

裁决：

- AFM/FPN-SM **降级为历史架构基线**，不再作为本项目默认论文主线；
- 不应声称频域家族整体无效；
- 只有在用户明确要求架构路线，且使用强基线、matched budget 和独立复现时才可重启。

主要证据：`docs/round2x_thinking_and_results.md`、`docs/reports/nwpu_matrix_analysis.md`、`docs/reports/m3_m4_matrix_summary.md`、`docs/reports/voc_m1_m2_statistical_significance.md`。

### 5.4 Prototype attraction、ETF 与 PAH：停止把类别几何当已知终点

原始 manifold 路线使用：

- `PrototypeBank` 表示每类多个原型；
- `SinkhornAssigner` 做平衡软分配；
- `TransportHead` 学习小残差，将 ROI 拉向原型投影点；
- ETF/PAH 尝试把类别关系固定为 simplex 或 prototype classifier。

问题不在于这些组件无法工作，而在于终点定义不充分。类别原型只描述类别归属，不能同时决定 bbox 精度、score calibration、阈值穿越和 NMS 生存。

实验证据：

- proposal-aligned PrototypeBank/TransportHead 只有 Penn-Fudan positive smoke，没有提交的 clean full-val/NWPU 正结果；
- 1024-D box-head 的 AP75 正/负样本 DualEnergy 为 `0.3225/0.3209`，几乎无区分；
- 75-D final-head 在 group level 有分离，但逐候选结构特征不增加 detector score 之外的信息；
- PAH 的历史 PF 实验中 AP75 与 ECE 明显变差；
- ETF/remote-sensing prototype 分支没有进入当前主线，也没有当前提交内可审计的 validated full-val gain。

裁决：

- **硬停止** prototype/ETF/PAH 作为“真实最优终点”的主张；
- **降级保留** PrototypeBank、Sinkhorn、class anchor、relation matrix 和 prototype diagnostics；
- 不再维护两个概念不同的 transport head 作为统一论文故事。

主要证据：`docs/energy_guided_roi_transport.md`、`docs/reports/roi_dual_energy_nwpu_cache_analysis.md`、`docs/reports/roi_structure_information_gain_round2148.md`。

### 5.5 PG-AW、Plan A、Plan B：停止直接修整 ROI feature geometry

这条线试图把 NC1、类内紧凑、FG/BG 分离、LOC_ERR、bbox-aware 等几何指标变成 loss，或在梯度层面投影冲突方向。

关键结果：

- baseline AP50/AP75 约 `0.6452/0.2948`；
- 几何 loss 可以让 NC1 下降、separability 上升；
- 最佳 PG-AW 约 `0.64779/0.29346`，AP50 只增 `+0.0026`，AP75 约降 `-0.0013`；
- `cos(G_geo, G_box) ≈ -0.011`，接近正交；投影后几何信号几乎全部保留，但 AP75 没恢复；
- Plan A loc-error 为 `0.64779/0.29345`；
- Plan B bbox-aware 为 `0.64779/0.29348`，其 raw loss 约为 `intra_tp` 的 `1/4000`。

裁决：

- **硬停止**通过 raw ROI feature geometry 直接改善 AP75 的路线；
- 保留 TP/CLS_ERR/LOC_ERR/BG taxonomy；
- 保留 high-IoU preservation 和 bbox-aware safety 的思想，但它们只能约束动作，不能充当目标。

证据说明：当前最完整数值记录在未跟踪的 `docs/energy_transport_vs_fpn_sm_analysis.md`，所引用远程 run report 未随当前提交保存，因此证据等级低于正式 strong-native matrix，但方向与后续实验一致。

### 5.6 Intrinsic dimension：停止把降维本身当优化目标

仓库实现可以在同一批 RPN proposals 上捕获 `(N,256,7,7)` ROI、fc1 和 1024-D box-head 特征，并计算 ID、effective rank、spectral decay、NC1 与 separability。

但当前仓库没有提交可复核的 `layerwise_geometry.json` 或完整原始结果，所以“256x7x7 的本征维度低于 1x1024”只能视为观察性结论。更重要的是，即使该观察成立，也不能推出 256x7x7 是更好的 correction state：

- spatial 256x7x7 的 73-way hard ranking AP75 为 `0.025362`，比 1024-D 的 `0.051452` 更差；
- dense spatial gain 只有非常弱的 detector-level 边际信息；
- context-only 已解释大部分 AP75 变化。

裁决：

- **停止**“逐层强制降维或最低 ID 就会提高检测”的训练目标；
- **保留** layerwise ID/effective-rank/NC1 作为表示诊断；
- 只有在结构指标能够增加 detector score/context 之外的信息，并通过 shuffle control 后，才考虑重新转为 loss。

### 5.7 Dual-energy 与 cone endpoint：停止作为逐候选 scorer

从 `basic` 项目吸收的正确内容是同时约束类内稳定与类间关系：

```text
E_intra = compactness + basin stability
E_inter = relation alignment + class anchor + separation
```

synthetic 五种子实验中，dual variant 能同时把两部分降到约 `0.0075 +/- 0.0019`，说明公式本身无退化。但真实 NWPU cache 表明：

- detector `label_prob` 的 AP75 AUC 为 `0.8293`；
- 1024-D `-E_intra` AUC 约 `0.4017`；
- 75-D `-E_intra` AUC 约 `0.4411`；
- 75-D `-E_basin` AUC 约 `0.5248`；
- 简单融合没有超过 `label_prob`；
- 较大 residual capacity 同时提高 rescue band 和 risky low-IoU band，缺乏选择性；
- global/class-conditional shuffle 也不支持逐 ROI 增量信息。

裁决：

- **硬停止** dual-energy/cone-DPOG 作为 standalone score residual、reranker 或主损失；
- **降级保留**为 batch/group-level structure diagnostic；
- cone projection 只回答局部残差稳定性，不再代表完整 manifold endpoint。

主要证据：`docs/basic_to_roi_cone_projection.md`、`docs/reports/roi_dual_energy_controlled_experiment.md`、`docs/reports/roi_dual_energy_nwpu_cache_analysis.md`、`docs/reports/roi_structure_information_gain_round2148.md`。

### 5.8 HWM teacher 与 energy term：停止因果主张

早期三种子 HWM full-val 报告曾显示平均 AP75 `+0.0540`，同时 precision/FPR/ECE 变好。但随后两个控制推翻了这一解释：

1. loss-causal ablation 中 `boxonly` 解释约 85% 的 AP75 增益；
2. `box+preserve` 的平均 AP75 delta `+0.0505`，高于带 HWM/energy 的 `current +0.0490`；
3. HWM loss 只有约 `2e-5` 到 `3e-5`，energy loss 也远小于 box loss；
4. 后续 zero-action parity 表明旧 action postprocessor 本身改变 prediction count、precision、FPR 和 ECE。

裁决：

- **硬停止** HWM 作为低能量 teacher 的主张；
- high-IoU preservation 作为普通 consistency regularizer 保留；
- 旧 HWM full-val 报告保留为“先有正结果、后被因果消融推翻”的研究记录。

主要证据：`docs/reports/action_local_hwm_nwpu_fullval_2026-07-09.md`、`docs/reports/hwm_loss_causal_ablation_2026-07-09.md`。

### 5.9 弱基线与 legacy postprocessor：历史 action 证据撤回

这是本项目最重要的实验卫生修复。

#### 弱基线

12-epoch NWPU detector 尚未充分适配。三种子 full-model continuation 的平均 AP75 delta 为 `+0.1058`，远高于旧 action、RLVR 和 box-only action gains。seed42 的 C0 18-epoch low-LR continuation也把 AP50/AP75 从 `0.518845/0.192277` 提到约 `0.607227/0.281808`。

因此，所有以旧 12-epoch checkpoint 为对照、又没有 matched-budget detector fine-tune 的增益，不能归因于新模块。

#### Legacy postprocessor

零 score/box action 仍然改变输出：

| checkpoint | Native AP75 | Zero-action AP75 | Prediction delta |
|---|---:|---:|---:|
| 12-epoch best | `0.191656` | `0.193084` | `-166` (`-8.56%`) |
| C0 AP75 best | `0.281522` | `0.283097` | `-58` (`-3.54%`) |

根因包括只取每 proposal 的 argmax 类、缺少 native small-box filter、bbox decode 不同、未恢复原图坐标，以及 threshold/top-K 不一致。

native repair 后，两 checkpoint 均满足：

- aggregate parity passed；
- strict parity passed；
- 196 张图 `mismatched_images=0`；
- actions exact zero；
- box/score/count/label error 全零。

裁决：

- **证据作废**：历史 action-local precision、FPR、ECE、prediction-count 改善；
- **证据作废**：将旧后处理的 prediction suppression 解释为 learned conservative action；
- 保留 native parity harness，并将其作为所有未来 action 实验的前置门。

主要证据：`docs/reports/nwpu_c0_native_parity_review_2026-07-10.md`、`docs/reports/nwpu_strong_native_candidate_energy_review_2026-07-10.md`。

### 5.10 Continuous residual field：明确 no-go

在 strong native identity AP75 `0.317619` 下：

| policy | Best AP75 | 相对 identity |
|---|---:|---:|
| Direct box-only | `0.167666` | `-0.149953` |
| Preserve-2 | `0.290477` | `-0.027142` |
| Preserve-2 最小 scale | `0.314247` | `-0.003372` |

所有测试 scale 都未超过 identity，learned direction 也没有稳定超过 within-image permutation control。即使使用 perfect GT acceptance gate，旧动作也只有 AP75 `0.346452` 或 `0.328327`，没有达到预设的 `0.36` 继续门槛。

裁决：

- **硬停止**旧 continuous direction field；
- 不再为该方向追加 stop gate、benefit head、energy smoothing 或更多 scale sweep；
- 保留 bounded residual、identity initialization、threshold preservation 和 rescue budget API。

### 5.11 73-way hard one-hot：objective 结构错误

动作空间包含 identity，加上 24 个平移/缩放/联合方向与 `0.05/0.10/0.20` 三个步长，共 73 个候选。

动作空间本身有明显 reachability：

| Greedy GT policy | AP50 | AP75 |
|---|---:|---:|
| 每图最多 32 动作 | `0.730278` | `0.519619` |
| 不限动作 | `0.732473` | `0.553664` |

但 hard classification 失败：

| feature | loss | AP75 |
|---|---:|---:|
| 1024-D box-head | `4.2364` | `0.051452` |
| 256x7x7 spatial ROI | `4.2866` | `0.025362` |

而 `ln(73)=4.2905`。box run 的 3662 个动作中仅 12.15% 有益，83.94% 有害。完整 budget/margin sweep 仍低于 identity。

裁决：

- **硬停止**把 73 个平滑相邻候选压成一个 one-hot class 的 objective；
- 不否证 soft ranking、continuous utility 或 structured candidate selection；
- 73 动作生成器继续作为 reachability 和候选空间工具。

### 5.12 Dense relative gain：停止当前 head，不否证局部信号

Dense supervision 学习：

```text
E(identity) - E(action) ~= quality(action) - quality(identity)
```

它比 hard one-hot 更合理，因为保留了所有候选的连续质量。结果也发生质变：box/spatial candidate-sign accuracy 达到约 `0.9047/0.8769`，说明局部几何顺序可学习。

strong-native full-val 的最佳 detector 结果：

| feature/target | AP75 | identity delta | moves |
|---|---:|---:|---:|
| Spatial / raw IoU gain | `0.318879` | `+0.001260` | 122 |
| Context-only / raw IoU gain | `0.318447` | `+0.000828` | 136 |
| Spatial feature shuffle | `0.317981` | `+0.000362` | - |

最佳 spatial 策略中 57.38% 动作有益、25.41% 有害，平均 proposal IoU gain 为 `+0.005414`。这证明局部 sign learning 真实存在。

但 detector-level promotion 失败：

- 预注册门槛为 AP75 `+0.0015`，实际只有 `+0.001260`；
- spatial 相对 context-only 只多 `+0.000432`；
- spatial 相对 feature shuffle 只多 `+0.000898`；
- 单种子、经过 epoch 选择；
- energy-drop `0.001` 时 AP75 下降 `-0.009230`，`>=0.003` 基本退回 identity，只有狭窄 `0.002` 窗口有效。

裁决：

- **边界停止**当前 proposal-independent candidate-energy head；
- 不扩 seeds 2024/999，不加深网络，不继续在同一 full-val 调 margin；
- 保留 dense relative gain、abstention、identity comparison 和 candidate reachability；
- 不能声称 spatial ROI information 为零，只能说其边际信息弱且未过 gate。

主要证据：`docs/reports/nwpu_strong_native_candidate_energy_review_2026-07-10.md`。

## 6. 被撤回的具体论文叙事

以下表达今后不应再出现：

1. “降低 ROI 本征维度会自然提升检测性能。”
2. “类别原型就是流形运输的真实终点。”
3. “HWM 找到了低能量轨迹的历史最优老师。”
4. “旧 action head 同时改善 AP75、precision、FPR 和 ECE。”
5. “频谱 verifier 的离线 AUC 可以直接转化为在线 AP。”
6. “DPO pair accuracy 高说明检测排序一定改善。”
7. “Greedy GT candidate policy 是数学意义上的 AP upper bound。”
8. “256x7x7 的低 ID 自动意味着它比 1024-D 更适合预测动作。”
9. “当前 `+0.001260` 已经证明了 spatial manifold detector gain。”
10. “多训练几轮 detector 得到的收益可以归因于 transport module。”

## 7. 仍然成立的结果与可复用资产

### 7.1 科学结果

- 局部候选动作空间有显著 reachability，但 local oracle 不是全局 AP upper bound；
- hard one-hot 与平滑候选质量结构不匹配；
- dense identity-relative supervision 能学习局部 gain sign；
- ROI spatial feature 有弱正边际信息，但 context/delta prior 解释大部分效果；
- local IoU gain 与最终 detector AP 不是同一个目标；
- threshold、matching 和 NMS 必须进入终点定义；
- 几何指标改善不能替代 full-val detection improvement；
- baseline adequacy 与 zero-action parity 是任何 post-training claim 的前提。

### 7.2 工程资产

- native BoxCoder、全前景类展开、small-box filter、class-wise NMS、top-K 与 transform postprocess；
- strict zero-action parity harness；
- bounded score/box residual、identity initialization；
- threshold preservation、rescue budget、high-IoU preservation；
- candidate action generator 与 dense gain targets；
- feature shuffle、context-only、identity 和 oracle controls；
- PrototypeBank、Sinkhorn、dual-energy、ID/effective-rank 诊断；
- AFM/FPN-SM 作为非默认架构 baseline；
- canonical metadata、checkpoint/split hash 与 full-val/smoke 标记。

## 8. 根因归纳

### 8.1 终点错位

多数失败路线优化的是 feature/prototype/local-IoU，而评估的是 post-NMS detector AP。代理量和终点之间缺少可验证的等价关系。

### 8.2 局部动作与集合决策错位

单 proposal 变好不代表最终预测集合变好。动作会改变 NMS 排名、重复框、类别竞争和全局 matching。

### 8.3 信息增益不足

很多 verifier、prototype 或 spatial feature 可以在离线分组上工作，却没有在 detector score/context 之上提供稳定增量信息。

### 8.4 相对目标缺少绝对安全约束

DPO 和 hard ranking 可以学会谁更好，却不能保证较差候选留在阈值以下，也不能控制预测数量和 FPR。

### 8.5 弱基线制造虚假模块增益

普通 full-model continuation 的提升远大于旧模块，说明 baseline 未收敛时，任何额外训练容量都可能看起来有效。

### 8.6 评估路径改变制造虚假决策增益

legacy postprocessor 在零动作时就改变输出。没有 exact identity parity，任何 precision/FPR/ECE 叙事都不可信。

## 9. DeepSeek 复核与 Codex 仲裁

DeepSeek V4 Pro 同意：

- external FFT reward、HWM、旧 continuous residual、hard 73-way ranking 和 prototype-local scorer 应停止；
- full-model fine-tuning 与 native parity 推翻了大量历史 action 解释；
- spatial/context controls 只支持弱局部信息，不支持 detector gain；
- 下一步应转向 set-level、NMS-aware selection。

Codex 对 DeepSeek 的三点过强结论作了修正：

1. 不能说所有频域方法已失败。FPN-SM 在 VOC ResNet50 和 NWPU MobileNet 上仍有正证据，因此应降级而非彻底否证。
2. 不能说低能量 thesis “从未被公平测试”。proposal-local dense candidate 已在 strong baseline 与 native parity 下得到公平测试并失败；尚未测试的是 set-level reformulation。
3. 不能用 RLimage 分割结果否定 manifold detection 的 AFM/transport 路线，跨项目证据已从最终裁决中排除。

## 10. 下一步唯一合理边界

若恢复研究，建议只推进以下问题：

```text
proposal set + detector-native candidates
  -> set-level action scores
  -> coordinated selection
  -> native threshold / matching / NMS
  -> post-NMS structured utility
  -> low-energy and safety constraints
```

### 10.1 候选空间

- 保留 identity；
- 保留 73 个 bounded geometric candidates 作为保守补充；
- 加入 detector-native bbox regression suggestions；
- 不再把候选编码成单一 hard class。

### 10.2 目标

- 主目标必须是 post-NMS、set-level structured utility；
- dense local gain 只能作为辅助目标；
- 必须显式考虑 threshold crossing、duplicate suppression、class competition 和 matching；
- low energy 是动作成本，不是终点定义。

### 10.3 强制对照

- native identity；
- context-only；
- feature shuffle；
- local greedy GT reachability；
- matched-budget box-head/full-model continuation；
- 相同 checkpoint、split hash、postprocess 与 prediction budget。

### 10.4 推进门槛

1. exact native zero-action parity；
2. seed42 development split 先超过预注册 AP75 gate；
3. 不在 full-val 上反复选择 epoch/margin；
4. 通过后才扩 seeds 2024/999；
5. 报告 AP50、AP75、precision、recall、FPR、ECE、prediction count 和 action diagnostics；
6. 必须优于 matched-budget ordinary fine-tuning，而不只是优于冻结 identity。

## 11. 最终回答

我们已经放弃：

- 把 external FFT 当 reward；
- 把 RLVR/GRPO/DPO 当当前独立 AP 提升主线；
- 把原型、ETF、低 ID 或 dual-energy 当已知最优终点；
- 直接在 ROI feature 上做几何压缩/投影以换取 AP75；
- HWM teacher 与旧 energy term；
- 旧 continuous residual direction field；
- 73-way hard one-hot ranking；
- 继续扩展当前 proposal-independent dense candidate head；
- 所有由弱 baseline 或 legacy postprocessor 支撑的历史 action 增益叙事。

我们没有放弃：

- 低能量作为局部动作成本与安全约束；
- bounded candidate action space；
- dense identity-relative gain；
- class prototype、dual-energy、ID 与频域特征作为诊断或先验；
- AFM/FPN-SM 作为单独架构 baseline；
- 在 set-level、NMS-aware 终点上重新定义 transport 的可能性。

因此，下一篇方法如果仍叫 manifold transport，它的核心不应再是“让特征更紧凑”，而应是：

> **在未知类别条件检测终点下，学习一组经过原生后处理仍有净收益的协调低能量动作。**

## 12. 主要证据索引

- `docs/reports/nwpu_strong_native_candidate_energy_review_2026-07-10.md`
- `docs/reports/nwpu_c0_native_parity_review_2026-07-10.md`
- `docs/reports/full_model_finetune_10ep_control_2026-07-09.md`
- `docs/reports/boxhead_finetune_10ep_control_2026-07-09.md`
- `docs/reports/hwm_loss_causal_ablation_2026-07-09.md`
- `docs/reports/boxrlvr_direct_matrix_review_2026-07-09.md`
- `docs/reports/roi_dual_energy_nwpu_cache_analysis.md`
- `docs/reports/roi_structure_information_gain_round2148.md`
- `docs/round2x_thinking_and_results.md`
- `docs/energy_guided_roi_transport.md`
- `AGENTS.md`
