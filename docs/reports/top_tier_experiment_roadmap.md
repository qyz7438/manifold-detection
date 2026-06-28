# 顶刊/顶会投稿实验路线图（RLimage / FPN Spectral Manifold）

> 目标：把当前项目从“实验探索”升级为可投稿 CVPR/ICCV/ECCV/TPAMI 的完整证据链。
> 本文件按“已完成 / 进行中 / 未开始”维护，建议每月回顾一次。

---

## 1. 论文核心贡献定位（建议）

**主标题方向**：面向两阶段目标检测的 FPN 级频谱残差适配器（FPN Spectral Manifold / fpn_sm）。

**一句话卖点**：
> 在 Faster R-CNN 的 FPN 与 RPN/ROI head 之间插入一个轻量、可学习的复频域残差适配器，通过低秩复数谱变换增强多尺度边界与定位信号，在多个数据集上稳定提升 AP75，且不会显著增加参数量或推理延迟。

**关键证据需要覆盖**：
1. 多数据集、多 backbone、多输入分辨率上的性能提升。
2. 严格的消融：插入位置、模块设计、频率坐标、层级坐标、初始化、相位/幅度分支。
3. 与相关频率模块/注意力机制的公平对比。
4. 统计显著性（多 seed 均值/方差，配对 t 检验）。
5. 效率分析（参数量、FLOPs、FPS、显存）。
6. 可视化与失败案例分析（频谱残差、门控响应、TP/FP 分布）。
7. 可复现性（确定性训练、开源配置、完整日志）。

---

## 2. 已完成的核心资产（截至 2026-06-22）

| 资产 | 状态 | 关键结果 |
|---|---|---|
| 稳定的 Faster R-CNN + MobileNetV3/ResNet50 训练/评估流程 | 完成 | Penn-Fudan / NWPU / VOC loader 都已支持 |
| FPN-level Spectral Manifold 模块 (`fpn_sm`) | 完成 | 复频域低秩残差 + 频率/层级坐标门控 |
| MPLSeg-style MicroAFM / ROI-level AFM | 完成 | 梯度诊断、相位消融 |
| Penn-Fudan 多组消融（gate、phase-only、freeze、data efficiency） | 完成 | 最佳 mid06 AP75 +12.7% |
| NWPU VHR-10 多分辨率矩阵 | 完成 | 480×800 baseline AP75 翻倍；fpn_sm 在 MobileNetV3 上有效 |
| VOC 20-class 数据与 loader | 完成 | 07+12 trainval / 12 val |
| VOC 20-class 实验矩阵 | 进行中 | MobileNetV3 bs32 + ResNet50 bs4，3 seeds，12 epochs |
| 确定性训练、多 seed 复现 | 完成 | `cudnn.benchmark=False` |
| 诊断指标（ECE、high-conf FP、score-IoU、localization stats） | 完成 | 自动输出 |

---

## 3. 顶刊/顶会还缺的实验清单

### 3.1 主实验：跨数据集 / 跨架构 / 跨骨干（高优先级）

| # | 实验 | 目的 | 状态 | 备注 |
|---|---|---|---|---|
| M1 | **VOC 20-class 完整矩阵**（MobileNetV3 + ResNet50，baseline/fpn_sm，3 seeds，12 ep） | 验证 fpn_sm 在标准 20 类检测上的泛化性 | 进行中 | 当前 MobileNetV3 seed42 已跑到 epoch 9，AP50≈0.70 |
| M2 | **VOC 20-class 统计汇总**（mean±std，配对 t 检验） | 给出可发表论文的表格 | 未开始 | 等 M1 完成 |
| M3 | **COCO 2017 验证**（至少 Faster R-CNN MobileNetV3 + ResNet50，1× schedule） | 顶会常见数据集，强泛化信号 | 未开始 | 数据量大，需要 1-2 周 |
| M4 | **NWPU VHR-10 多 seed 最终矩阵**（320×480 / 480×800，3 seeds） | 补足统计显著性 | 部分完成 | 已有单/双 seed 结果，需补齐第三 seed |
| M5 | **Penn-Fudan 最终 3-seed 汇总**（含 ECE、AP75、localization stats） | 小数据集上的完整基线 | 部分完成 | 已有大量实验，需整理统一配置 |
| M6 | **不同输入分辨率扫描**（320/480/640/800 for MobileNetV3；800/1000/1333 for ResNet50） | 证明收益与分辨率的关系 | 部分完成 | NWPU 已做 320 vs 480；VOC 只有 320/800 |

### 3.2 消融实验：模块设计（高优先级）

| # | 实验 | 目的 | 状态 | 备注 |
|---|---|---|---|---|
| A1 | **插入位置消融**：只在 FPN / 只在 ROI box_head / FPN+ROI / 无 | 定位最佳插入点 | 部分完成 | PF 上做过 box_head；当前主推 FPN |
| A2 | **门控坐标消融**：无坐标 / 仅 radius / radius+angle / +level | 证明频率/层级坐标的必要性 | 部分完成 | `--no-fpn-sm-use-freq-coords` 已优于有坐标 |
| A3 | **alpha 初始化扫描**：0.001, 0.01, 0.1, 1.0 | 找到稳定且有效的初始化 | 部分完成 | PF 上 0.01 最优 |
| A4 | **latent/hidden 维度扫描**：32/64, 64/128, 128/256 | 参数量 vs 性能 trade-off | 部分完成 | 当前默认 64/128 |
| A5 | **幅度 vs 相位 vs 完整分支**：mag-only / phase-only / full | 解耦模块机制 | 部分完成 | PF 上 phase-only 已达完整效果 |
| A6 | **门控激活函数**：sigmoid / sigmoid2 / tanh / none | 门控动态范围影响 | 未开始 | 当前 sigmoid |
| A7 | **残差模式**：current / delta / norm_delta | 确认残差形式 | 部分完成 | 旧 AFM 做过 |
| A8 | **共享 vs 每层级独立 adapter** | 验证跨尺度共享是否足够 | 未开始 | 当前 adapter 共享，alpha 独立 |
| A9 | **与 box_head AFM 的对比**（fpn_sm vs roi_sm vs 两者叠加） | 判断哪一层级频谱信息最有效 | 未开始 | 工程上容易实现 |
| A10 | **是否冻结 backbone/RPN/box head 后训练** | 后训练稳定性与数据效率 | 部分完成 | PF 上 freeze_rpn / freeze_box 做过 |

### 3.3 统计显著性（中优先级，可在主实验后批量做）

| # | 实验 | 目的 | 状态 |
|---|---|---|---|
| S1 | 所有主实验跑 **3 seeds 以上**，报告 mean ± std | 排除随机波动 | 进行中 |
| S2 | 对 baseline vs fpn_sm 的 AP50/AP75 做 **配对 t 检验 / Wilcoxon** | 量化显著性 | 未开始 |
| S3 | 绘制 **训练曲线**（loss、AP50、AP75、ECE per epoch） | 展示收敛与稳定性 | 未开始 |
| S4 | **消融实验的置信区间**（Bootstrap） | 小数据集结论更可信 | 未开始 |

### 3.4 对比实验（中-高优先级）

| # | 对比方法 | 目的 | 状态 | 备注 |
|---|---|---|---|---|
| C1 | **标准 Faster R-CNN baseline** | 主对比 | 完成 |
| C2 | **+FcaNet / ECA / CBAM 通道注意力** | 证明频谱残差不只是另一种注意力 | 未开始 | 可用现成实现 |
| C3 | **+DCT-SA 或类似频域模块** | 与同类频率方法比较 | 未开始 | 需调研代码 |
| C4 | **+AutoAugment / Mosaic / MixUp** | 区分“数据增强收益”与“架构收益” | 未开始 | 数据增强基线 |
| C5 | **+GFL / GFLv2（若可用）** | 与定位质量估计方法比较 | 未开始 | 属于 stronger baseline |
| C6 | **其他检测器**：RetinaNet / FCOS / DETR | 验证 fpn_sm 是否限于 two-stage | 未开始 | 工作量较大，可选 |

### 3.5 分析与可视化（中优先级，投稿前必须）

| # | 分析 | 目的 | 状态 | 备注 |
|---|---|---|---|---|
| V1 | **小/中/大目标 AP**（VOC COCO 标准尺寸划分） | 回答“小目标是否受益” | 未开始 | 用户特别关注 |
| V2 | **定位误差分解**：center error、size error、IoU distribution | 解释 AP75 提升来源 | 部分完成 | 旧脚本有，需适配 fpn_sm |
| V3 | **ECE / MCE / 温度缩放后 ECE** | 校准分析 | 部分完成 | 已自动输出 ECE |
| V4 | **score-IoU correlation / AP@fixed score** | 验证分数质量 | 部分完成 | 旧脚本有 |
| V5 | **频谱残差可视化**（|F_out - F_in| 热力图） | 说明模块学到了什么 | 未开始 | 需新增 hook |
| V6 | **门控响应可视化**（radius-angle 门控图） | 解释门控是否做频率选择 | 未开始 | 当前 gate 可能是 uniform |
| V7 | **FP/TP 样例对比**（失败案例） | 直观证明改进 | 未开始 | 从 VOC 验证集挑选 |
| V8 | **alpha 轨迹跟踪**（per-level residual strength） | 证明模块确实在学习 | 未开始 | 可在训练 hook 中记录 |

### 3.6 效率与可部署性（中优先级）

| # | 实验 | 目的 | 状态 |
|---|---|---|---|
| E1 | **参数量对比**（baseline vs +fpn_sm） | 证明轻量 | 未开始 |
| E2 | **FLOPs 对比**（用 fvcore / ptflops） | 计算开销 | 未开始 |
| E3 | **推理 FPS / latency**（单 GPU，batch=1） | 实际部署影响 | 未开始 |
| E4 | **训练显存占用曲线** | 训练成本 | 部分完成 | 监控中 |
| E5 | **fpn_sm 前向耗时占比** | 瓶颈分析 | 未开始 |

### 3.7 可复现性与代码（投稿前必须）

| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| R1 | 所有核心脚本、配置、运行命令整理到 `scripts/` 与 `nni_configs/` | 部分完成 |
| R2 | 训练/评估 log、checkpoint、metrics 归档规范 | 部分完成 | 目前按 run 目录保存 |
| R3 | README 包含一键复现命令、环境、数据准备 | 未开始 |
| R4 | 关键结果汇总表自动化（`scripts/summarize_round.py` 已存在） | 部分完成 | 可扩展 |
| R5 | 开源许可证与代码清理 | 未开始 |

---

## 4. 当前进度映射

| 大项 | 当前状态 | 完成度 |
|---|---|---|
| 核心模块 `fpn_sm` | 已实现并进入 VOC 主实验 | 80% |
| Penn-Fudan 消融 | 已完成多轮，需整理成统一表格 | 75% |
| NWPU VHR-10 验证 | 已有多分辨率结果，需补第三 seed 与统计 | 60% |
| VOC 20-class 主实验 | 进行中（MobileNetV3 seed42 epoch 9/12） | 35% |
| COCO 验证 | 未开始 | 0% |
| 消融 A2-A10 | 部分做过旧版本，需针对 fpn_sm 系统重跑 | 40% |
| 统计分析 S1-S4 | 未开始 | 0% |
| 对比实验 C2-C6 | 未开始 | 0% |
| 可视化 V1-V8 | 部分指标已有，系统可视化未做 | 30% |
| 效率 E1-E5 | 未开始 | 0% |
| 可复现性 R1-R5 | 部分完成 | 40% |

---

## 5. 建议执行顺序（未来 2-4 周）

1. **跑完 VOC 20-class 矩阵**（M1）：当前最高优先级，预计 2-3 天完成剩余 8 个 run。
2. **汇总 VOC 结果并做统计显著性检验**（M2, S1-S2）：判断 fpn_sm 在 VOC 是否有效。
3. **若 VOC 有效**：
   - 立刻补 COCO 小规模 smoke（M3，至少 1 seed，12 epoch）。
   - 系统做消融 A2-A10（优先 A2/A3/A5/A9）。
   - 补齐 NWPU 第三 seed（M4）。
4. **若 VOC 无效**：
   - 回到 Penn-Fudan/NWPU 深度机制分析，检查是否 fpn_sm 仅对小数据/低分辨率有效。
   - 考虑把论文叙事改成“任务条件分析”：fpn_sm 在什么条件下有效。
5. **并行启动效率分析**（E1-E3）和可视化脚本（V5-V8），不依赖主实验完成。
6. **最后整理 README、代码、配置**（R1-R5）并撰写论文。

---

## 6. 风险与预案

| 风险 | 影响 | 预案 |
|---|---|---|
| VOC ResNet50 fpn_sm bs4 仍 OOM | M1 不完整 | 降至 bs2 或启用 gradient checkpointing |
| MobileNetV3 bs32 lr0.024 最终 AP 仍低于 baseline | fpn_sm 在轻量 backbone 上失效 | 保留 bs16 lr0.024 结果作为对照；论文说明 bs 影响 |
| VOC fpn_sm 无统计显著提升 | 核心贡献不成立 | 转向“任务条件/分辨率/轻量 backbone 专用”叙事；或结合分割结果 |
| COCO 训练太慢 | 无法在投稿前完成 | 用 COCO 2017 val 子集或 1× schedule；或放补充材料 |
| 对比方法实现复杂 | 拖延进度 | 优先做 FcaNet/ECA；更复杂方法放补充材料 |

---

*本文件由 Kimi Code 于 2026-06-22 维护，建议随实验进展每 1-2 周更新一次状态。*
