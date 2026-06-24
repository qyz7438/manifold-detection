# 面向视觉感知模型的可验证奖励后训练研究报告

项目：RLimage  
日期：2026-06-04  
作者：青云志与 Codex 协作实验记录  
版本：v1.0  

## 摘要

本报告系统总结了 RLimage 项目从最初的多源视觉反馈分类设想，到目标检测 RLVR 后训练框架，再到语义分割迁移计划的完整研究过程。项目最初希望验证一个直观假设：预训练视觉模型在普通监督微调之后，是否可以通过频域扰动、局部无意义 Patch 干扰和多视图一致性后训练，获得更强的鲁棒性、更稳定的预测分布和更少的高置信错误。随着讨论深入，任务被重新界定为一个更有研究价值的目标：把文本模型领域的 RLVR - Reinforcement Learning with Verifiable Rewards，可验证奖励后训练思想迁移到视觉感知任务中，使目标检测模型不只是对已有候选框做重排序，而是能够基于可验证奖励逐步学会哪些框可信、应该看哪里、以及怎样框得更准。

这一目标受到 Visual-RFT、DeepSeekMath/GRPO、Tulu 3 RLVR、GFL 质量估计、MPLSeg 幅度-相位解耦等工作的启发。与普通数据增强或校准方法不同，本项目试图把频域证据变成视觉后训练中的 verifier 组成部分。实验路线经历了三条主线。第一条是 Penn-Fudan 目标检测 MVP：使用 TorchVision Faster R-CNN 在 person detection 上训练 baseline，再用 ROI 幅度谱相似度 R_amp 诊断 TP 与 FP。结果显示 R_amp 的 TP/FP 排序 AUC 可达 0.93 左右，说明频域证据存在统计判别信号；但直接把 image-level reward 权重乘到 detector loss 上，会导致模型极度保守，AP50 从 0.863 降到约 0.435，Recall 从 0.879 降到约 0.462。第二条是 Spectral Quality Head：冻结 detector，将 ROI feature、幅度 profile 和结构特征输入质量头，用于候选框质量估计和重排序。该路线工程上稳定，TP/FP AUC 约 0.95，并显著改善 ECE 和高置信 FP，但最优结果来自 ROI-only，频域分支没有形成明确增量。第三条是 RLVR 式检测后训练：实现 rollout、verifier、signed advantage、KL anchor、frozen baseline 等机制，逐轮修复训练崩塌、读数不清、BatchNorm 状态漂移和 shuffled control 失效等问题。Round 2.3 之后 RLVR shell 终于稳定，AP50 与 baseline 差距小于 0.01，预测数正常；但 Round 2.5 及其补充结果显示 R_amp、structure、phase/edge 以及 shuffled controls 的 AP50 几乎都挤在 0.884-0.886，差距小于 0.001，说明手工 box-level 频谱 verifier 在 Penn-Fudan 检测上没有可测量的因果信号。

项目随后尝试把 MPLSeg 的 magnitude-aware 与 phase-sensitive 思想从外部 verifier 转成网络内部 AFM/FFT 模块。Round 2.6 的普通 AFM 损伤 AP75 和 precision；Round 2.7 的残差 identity 初始化使 AP50 接近 baseline；Round 2.8 进一步证明 AFM 的 `mag_scale` 与 `phase_scale` 始终为 0，AP75 提升主要来自 box head 适配，而不是 FFT 门控本身。Round 2.9 和 2.10 进一步修复 checkpoint 与 edge-mix 控制，证明后训练可以稳定运行，但仍没有形成频域因果证据。基于这些结果，本项目形成一个重要负结论：在 Penn-Fudan 这种小规模、单类别、目标结构相对简单的两阶段检测任务中，手工傅里叶 ROI reward 作为 detector 后训练信号不可直接成立。它可以作为诊断特征或校准信号，但不足以驱动模型本体学习。

报告最后给出下一步路线。Plan 2.11 被修订为 VOC spectral signal gate，用更复杂的 VOC 3 类检测子集检验手工频谱 verifier 是否能超过 shuffled control；补齐结果后，真实 spectral loggate 仍未超过 shuffled spectral。Plan 2.12 随后转向 MPLSeg-style AFM 梯度修复：去掉旧 AFM 中导致 `mag_scale`、`phase_scale` 严格零梯度的 learnable scale 门控，改用硬激活的幅度门控与相位残差。单元测试 8 项通过，三组对照显示 MPLSeg AFM 的 `residual_scale=0.9720`，AP50 0.8678，AP75 0.6534，ECE 0.0666，说明 FFT 路径可以被真正激活；但这更像是网络内部结构修复，而不是已经证明手工频域 reward 有效。与此同时，Plan 4.0/4.1 将主线迁移到语义分割，因为 dense mask 能为幅度、相位、结构 verifier 提供像素级可验证目标，减少 NMS、候选框匹配和 ROI crop 位移带来的噪声。总体而言，本项目的贡献不是证明“傅里叶 reward 已经有效”，而是建立了一条可复现实验证据链：RLVR shell 在目标检测上可以稳定实现；频域证据在候选框层面有统计信号但缺乏因果后训练效果；MPLSeg 式幅度/相位思想需要以更贴近原论文的激活门控和 dense task 来继续验证。

## 1. 研究背景

视觉模型的常规训练流程通常分为预训练、监督微调和任务评估。对于图像分类，预训练 ViT、DeiT 或 ResNet 在 ImageNet 上获得通用表征，再在 CIFAR、VOC 或下游数据集上微调；对于目标检测，常见方案是加载 COCO 预训练的 Faster R-CNN、Mask R-CNN、RetinaNet 或 FCOS，再修改分类头并在目标数据上训练。传统微调的优势是稳定、工程成熟、指标直接，但它也有两个局限。第一，监督 loss 只告诉模型“这个样本应该输出什么”，不直接告诉模型“为什么当前输出不可信”。第二，检测模型的 score 往往同时承担类别置信度和定位质量含义，容易出现 score 高但定位差、score 高但局部干扰误导、score 与真实 IoU 不一致的问题。

目标检测中的质量估计长期是一个重要问题。Faster R-CNN 通过 RPN 生成区域 proposal，再通过 ROI head 分类和回归边界框。Mask R-CNN 在 Faster R-CNN 之上增加 mask 分支和 RoIAlign，用于实例分割。GFL 和 GFLv2 等工作指出，检测中的分类分数与定位质量之间存在训练和推理不一致，因此需要显式建模 localization quality。也就是说，一个候选框是否“可信”，不能只看分类 softmax，还要看它与真实目标的空间关系、定位质量、区域内部是否包含目标证据。

与此同时，鲁棒性和不确定性估计研究也表明，模型在自然图像外的干扰上可能出现高置信错误。AugMix 的核心思想是通过多种增强视图和一致性约束提升鲁棒性与校准，而不是只追求 clean accuracy。项目最初的分类设想正是受此启发：先训练一个 clean baseline，再用 low-pass、high-frequency perturbation 和 random patch 等视图进行后训练，要求模型对语义不变的扰动保持一致，并降低高置信错误。这个思路在分类任务上是合理的，但用户很快澄清，真正目标不是图像分类，也不是图像生成，而是面向目标检测模型的 RLVR 式后训练框架。

RLVR 的关键思想来自语言模型后训练。Tulu 3 将 RLVR 命名为使用可验证奖励进行强化学习后训练的方法；DeepSeekMath 引入 GRPO，用组内相对奖励替代传统 PPO critic，从而降低价值模型成本；DeepSeek-R1 和后续 RFT 工作说明，在数学、代码、工具调用等可验证任务中，模型可以通过规则奖励、执行反馈或答案校验进行自我改进。Visual-RFT 将类似思想迁移到视觉感知和视觉语言任务，对图像分类、目标检测、grounding 等任务构造可验证奖励，并使用 GRPO 等策略优化算法更新模型。本项目的研究问题就是：目标检测中的 IoU、类别正确性、高置信 FP、频域区域证据，能否共同构成一个视觉 verifier，让 detector 通过后训练学得更鲁棒、更准、更少高置信错误？

频域证据的引入来自两个方向。第一，图像的低频通常对应整体形状、语义结构和大尺度颜色分布，高频通常对应边缘、纹理、噪声和局部细节。PyTorch 的 `torch.fft.fft2` 可以直接对图像张量最后两个空间维进行二维离散傅里叶变换，因此很适合构造 low-pass/high-frequency 视图或 ROI 幅度谱特征。第二，MPLSeg 提出 magnitude-aware 与 phase-sensitive 的解耦思想，认为幅度更偏向语义证据，相位更偏向定位和结构证据。这个观点对本项目很关键，因为它提醒我们不能把“傅里叶相似度”粗暴塞进 detector loss；更合理的方式是将幅度作为语义 verifier，将相位/边缘作为结构 verifier，并尽量让它们作用在有稳定空间对齐的任务上。

## 2. 初始方案与任务重构

项目最初被理解成一个针对预训练视觉分类模型的多源视觉反馈后训练实验。第一版方案使用 CIFAR-100、timm DeiT/ViT 预训练权重，训练分两阶段：阶段一普通监督微调，阶段二加载 baseline checkpoint 并构造四种视图 `x, x_low, x_high, x_patch`。损失函数包括分类交叉熵、多视图一致性 KL 和高置信错误惩罚。评估指标包括 clean accuracy、low-pass accuracy、high-frequency accuracy、patch accuracy、prediction consistency、high-confidence error rate 和 ECE。这套设计从图像分类角度看是完整的，且与 AugMix 风格的一致性鲁棒训练有清楚关系。

但用户指出这个理解偏了。真实目标不是“让分类模型对扰动更稳”，而是“构建一种面向目标检测模型的 RLVR 式后训练框架”。这带来三个根本变化。第一，输出空间从一个类别分布变成候选框集合，每个框有类别、score、坐标和 NMS 关系。第二，reward 不再只是图像级正确/错误，而应在候选框层面衡量 IoU、类别正确性、定位偏差、是否高置信 FP、是否漏检。第三，频域证据不能再只是增强视图，而是 verifier 的一部分，用来判断区域内部是否像真实目标。

于是项目目标被重新表述为：

| 层级 | 目标 |
|---|---|
| 基础模型 | 使用预训练 Faster R-CNN 或 Mask R-CNN，在 Penn-Fudan person detection 上得到 supervised baseline |
| 可验证奖励 | 构造 IoU、class correctness、high-confidence FP penalty、R_amp、structure 等 verifier |
| 后训练方式 | 使用 rollout、group reward、signed advantage、KL anchor 更新 ROI head 或 detector 部分模块 |
| 关键假设 | 频域区域证据能帮助模型减少局部 patch 干扰和高置信错误，并提升定位稳定性 |
| 必要控制 | 与 IoU-only、shuffled spectral、det-only continuation、quality reranking 对照 |

这个重构使项目从普通增强训练转向真正的视觉 RLVR 研究。它也让后续每次失败都更有诊断价值：如果直接 reward-weighted loss 崩塌，问题可能在训练目标；如果 quality head 有效但频域没增量，问题可能在 ROI feature 已经足够强；如果 RLVR shell 稳定但 real 与 shuffled 不分，问题可能在 verifier 信号本身；如果 AFM gate 不激活，问题可能在预训练 detector 的特征分布不允许频域模块快速介入。

## 3. 实验基础设施

第一阶段使用 Penn-Fudan Pedestrian 数据集。TorchVision 官方教程用该数据集演示预训练 Mask R-CNN 的检测和分割微调，该数据集包含 170 张图像和 345 个行人实例，规模小、可快速迭代，适合 MVP。项目实际使用 `fasterrcnn_mobilenet_v3_large_320_fpn`，加载 COCO 预训练权重，将 ROI predictor 改为二分类：background/person。图像尺寸上限设置为 320，以减少训练和评估开销。指标从 AP50 扩展到 AP75、precision、recall、ECE、high-conf FP、prediction count、miss rate、score-IoU correlation 等。

为了制造局部干扰，项目实现了 random patch、checkerboard patch、object-edge、object-inside、near-object 等场景。最初只做 clean、random 和 checkerboard；后续为了判断 q_spec 或 RLVR 是否影响不同错误类型，将 patch 位置分组：background patch 检查是否产生 FP，object-inside patch 检查是否漏检，object-edge patch 检查是否框偏移，near-object patch 检查是否重复检测或框漂移。

频域特征实现分为几类。`fft_features.py` 负责 FFT 幅度计算；`radial_profile.py` 将幅度谱按半径分 bin 求均值，得到从低频到高频的一维能量 profile；`spectral_reward.py` 使用 ROI_pred 与 ROI_gt 的 radial profile cosine similarity，再经过压缩得到 R_amp；`roi_spectral_dataset.py` 缓存候选框、匹配关系、ROI feature、amplitude profile 和 structure feature；后续 `rlvr_reward.py`、`detection_verifier.py` 和 `roi_policy_loss.py` 则将这些信号接入 RLVR 后训练。

工程上，项目逐步形成以下目录结构：

| 模块 | 作用 |
|---|---|
| `spectral_detection_posttrain/datasets` | Penn-Fudan、VOC、patch transform |
| `spectral_detection_posttrain/eval` | detector evaluation、rerank evaluation、spectral reward eval |
| `spectral_detection_posttrain/spectral` | FFT、ROI crop、radial profile、spectral reward |
| `spectral_detection_posttrain/models` | detector builder、MicroAFM、SpectralQualityHead |
| `spectral_detection_posttrain/rlvr` | detection verifier、ROI policy loss |
| `spectral_detection_posttrain/train` | supervised baseline、reward-weighted posttrain、RLVR posttrain |
| `scripts` | round-specific matrix runner、diagnostics、summary |
| `docs/superpowers/plans` | 每一轮计划、修正和结论 |
| `runs` | 每组实验配置、checkpoint、metrics 和可视化 |

这套基础设施的价值在后续逐渐体现。很多早期实验结果不理想，但因为保存了 checkpoint、eval metrics、train jsonl、r_amp stats 和 diagnostics，后续才能定位是 reward 信号问题、训练状态问题、checkpoint 加载问题，还是结果可读性问题。

## 4. MVP：R_amp 作为候选框 verifier 的初步证据

最小可行版本使用 Penn-Fudan、Faster R-CNN、person detection。训练 baseline 1 epoch，然后做两件事。第一，计算预测框与 GT 框的 ROI 幅度谱 radial profile 相似度，比较 TP 与 FP 的 R_amp 分布。第二，加载 baseline checkpoint 做 5 epoch image-level reward-weighted fine-tuning，再评估 clean、random patch、checkerboard patch。

MVP 结果中，baseline clean AP50 为 0.8630，precision 为 0.6838，recall 为 0.8791，预测数 117；random patch AP50 为 0.8342，checkerboard AP50 为 0.8624。R_amp 诊断显示 baseline TP mean 约 0.9989，FP mean 约 0.9899，绝对差值很小，但 TP vs FP AUC 达到 0.9311；post-train checkpoint 的 R_amp AUC 约 0.9411。这说明 R_amp 在排序意义上确实能分辨一部分好框和坏框，至少不是随机噪声。

然而后训练结果暴露了第一条失败模式。post-training clean AP50 降到 0.4349，Recall 降到 0.4615，预测数从 117 降到 61；random patch 和 checkerboard patch 上 AP50 也只有约 0.433-0.453。高置信 FP 变为 0，看似错误被抑制，但真正原因是模型变得过度保守，许多目标不再检测出来。这是典型的 reward 乘法压制：如果把 image-level reward 或 image_weight 直接乘到整套 detector loss 上，模型学到的不是“哪些框更可信”，而是整体降低响应，减少犯错机会。

MVP 的结论因此分成两层。正面结论是 R_amp 有统计信号，能作为 verifier 候选。负面结论是 image-level reward-weighted detector fine-tuning 不可用，因为它不改变具体候选框动作的梯度方向，只是粗暴调节整张图的 loss 强弱，导致 recall 和 AP 崩塌。这个结果直接推动了下一轮策略：把 R_amp 从训练主 loss 中拿出来，先作为候选框质量估计与重排序信号，而不是直接乘 detector loss。

## 5. Spectral Quality Head：从后训练转向质量估计

用户提出不要再把 R_amp 直接乘到 detector loss 上，而是借鉴 MPLSeg 的“幅度负责语义证据、相位负责定位/结构证据”思想，把频域信息改造成独立的 Spectral Quality Head。这个阶段的核心设计是固定 baseline detector，离线缓存候选框、ROI feature、GT 匹配、ROI crop、amplitude profile 和 structure feature，然后训练一个小 MLP 输出 `q_spec`，表示候选框内部的频域/结构证据是否像真实目标区域。

训练 target 不是原始 R_amp，而是复合质量：

`q_target = C_cls * IoU(box_pred, box_gt) * S_amp`

其中类别正确时 `C_cls=1`，否则为 0；`S_amp` 是标准化后的 R_amp；unmatched FP 的 target 设为 0。损失包括 BCE 或 SmoothL1 质量回归，以及 same-image TP/FP pairwise ranking loss。这个设计承认 R_amp 的强项是排序能力而不是数值尺度，因此直接优化 ranking。

实验结果显示，quality head 工程上明显优于 image-level post-training。ROI-only head 的 q_spec AUC TP vs FP 达到 0.9559，mean q TP 为 0.7654，mean q FP 为 0.1417，q-IoU corr 为 0.7538；ROI+Amp+Struct 的 AUC 为 0.9471，mean q FP 更低，为 0.0836，q-IoU corr 略升至 0.7692。这说明学习式质量头能够非常好地区分 TP/FP，但也揭示一个关键问题：ROI feature 本身已经非常强，频域/结构分支没有在 AUC 上超过 ROI-only。

在 reranking 评估中，baseline clean AP50 约 0.8736，precision 0.6983，recall 0.8901，ECE 0.3017。ROI+spectral 在 alpha=0.7 时 clean AP50 为 0.8583，precision 升至 0.7182，recall 降至 0.8681，high-conf FP 从约 0.0286 降到 0.0145，ECE 降至 0.1663；alpha=0.9 时 AP50 保持 0.8741，recall 0.8901，同时 ECE 约 0.1663。patch 场景也类似：AP50 没有明显提升，但 ECE 和 precision 改善。

这一阶段的判断是：quality head 是一个健康的校准/重排序工具，但它不是用户真正想要的“RLVR 后训练”。它没有让 detector 本体学会框得更准，只是在已有候选框上重排和校准。更重要的是，消融结果表明最优配置多次来自 ROI-only，频域证据没有稳定增量。于是项目不能停在 q_spec reranking，而必须回到 RLVR 主线：让 detector 通过可验证 reward 更新 ROI 分类/定位行为。

## 6. NNI Quality Matrix：确认 q_spec 的优势与边界

为了避免手调 alpha 和单次结果偶然性，项目使用 NNI 做了 quality head 变量矩阵。变量包括 detector baseline 训练 epoch：1、3、5；quality head 输入：ROI-only、ROI+Amp、ROI+Amp+Struct；QH 训练 epoch：8、20 且 early stopping；rerank alpha：0.95、0.9、0.85、0.8、0.75、0.7。总共 108 个 trial 全部成功。

目标函数设为：

`AP50 + Precision@Recall=0.85 - ECE - High-conf FP rate`

最佳结果来自 `detector_epochs=1, QH=ROI-only, QH epochs=8, alpha=0.70`，AP50 为 0.8595，precision 0.6752，recall 0.8681，high-conf FP 为 2，ECE 为 0.0656，Precision@Recall=0.85 为 0.8764。ROI+Amp 最佳 1 epoch 行的 ECE 更低，约 0.0566，但固定 recall precision 只有 0.8041；ROI+Amp+Struct 可达到类似的 fixed recall precision，但目标函数仍不超过 ROI-only。

这组 NNI 结果给出几个重要结论。第一，alpha 是真实 trade-off，越小越强烈地依赖 q_spec，越能降低 FP 和 ECE，但更容易损失 recall；alpha=0.9 更保守，alpha=0.7 更激进。第二，quality head 的最大收益是校准和错误抑制，而不是 AP 提升，因此后续指标不能只看 AP50，还应看 ECE、high-conf FP、Precision@fixed Recall、Recall@fixed Precision。第三，频域特征在 Penn-Fudan clean split 上没有击败 ROI-only，说明要证明频域贡献，必须做 patch 分组、shuffled amplitude、random amplitude、oracle R_amp 等控制。第四，quality head 仍是 reranking，不是 RLVR；它可以作为 verifier 上限或 baseline，但不能替代后训练。

## 7. RLVR 后训练：从错误实现到稳定 shell

用户明确指出，项目真正目标不是 reranking，而是把 RLVR/Visual-RFT 式可验证奖励迁移到目标检测任务。于是项目进入 RLVR 主线。理想流程是：detector 对同一图像生成多个 rollout 或候选动作；verifier 依据 IoU、类别正确、R_amp、structure、高置信 FP、miss rate 等给出奖励；组内归一化得到 advantage；用 signed policy loss 更新 ROI 分类头或定位头；用 KL anchor 保持模型不偏离 baseline；最终让 detector 本体学会更可信、更准确的检测行为。

第一版 RLVR/NII 搜索包含 ramp/qspec、cls/box unfreeze、AdamW/SGD 等组合。结果 8 个 trial 全部成功，但与 baseline 相比 AP50 大幅下降。最佳 ramp/box/AdamW clean AP50 约 0.668，checkerboard AP50 约 0.632；baseline clean AP50 约 0.863。ECE 从 0.302 降至 0.087，说明校准改善明显，但 AP 和 recall 代价过大。该阶段证明：RLVR 不再像 image-level reward 那样彻底崩到 0.43，但仍无法满足检测性能约束。

随后 Round 2.1 针对工程问题做稳定优先修复。发现的问题包括：用 post-NMS final boxes 训练 bbox head，导致 train/inference 坐标分布错位；shuffled_ramp 未实际调用；temperature 没传入 verifier config；R_amp z-score 将 FP 的 0 拉成极端负值；训练候选框阈值与 eval 阈值混用；s_amp 与 boxes 过滤不同步。Round 2.1 修复这些问题，采用更保守参数：只解冻 cls_score，box_loss_weight=0，reward_lambda=0.1，max_candidates=40，reward_score_threshold=0.2。但结果仍然不理想，best clean AP50 约 0.623，precision 约 0.21，prediction count 约 300，表现为低置信预测爆炸。

Round 2.2 的洞察是：正权重 CE 不是 GRPO/RLVR。低 reward 不应只是给 background target 更大 CE，而应降低模型对当前动作的概率；高 reward 提高当前动作概率。这要求 signed advantage objective，而不是 positive weighted CE。Round 2.2 引入：

`L = det_loss_weight * L_det + policy_loss_weight * L_signed_policy + baseline_kl_weight * L_baseline_kl + recovery_loss_weight * L_recovery`

并使用 frozen baseline rollout/logit replay。这个方向正确，但结果中仍出现 score distribution 损坏，预测数一度暴涨到约 1150，precision 约 0.058。问题不只是 loss，而是训练状态污染：`requires_grad=False` 冻住参数，但 `model.train()` 仍可能让 frozen BatchNorm running stats 漂移；此外结果行缺字段，无法判断 null/no-update 是否真的无更新。

Round 2.3 修复了可读性与 freeze-state。每个 trial 记录 name、loss weights、rollout source、policy objective、checkpoint、eval status、clean/edge metrics、num_predictions、precision、ECE 等完整字段；加入 null no-update 快速路径；冻结模块保持 eval mode；记录 initial ROI KL sanity。此后稳定 shell 第一次成立。Round 2.3 中 `signed_iou_0003_kl10` clean AP50 为 0.8726，AP75 0.6582，recall 0.8901，prediction count 125，high-conf FP 2；baseline/null no-update AP50 约 0.8723，prediction count 124。也就是说，`KL=10 + policy=0.0003 + frozen baseline + signed objective` 可以让 detector 在 AP 和预测数量上保持稳定。

这个阶段的核心成果不是频域 reward 有效，而是 RLVR shell 终于可解释、可稳定运行。没有这个 shell，后续任何 R_amp 结果都无法判断是 reward 信号、训练崩塌还是实现错误造成。

## 8. Round 2.5：频谱 verifier 的因果性检验

有了稳定 shell 后，项目补齐 real vs shuffled 与结构分支。Round 2.5 使用 KL=10、policy=0.0003、cls-only、AdamW、baseline rollout，并比较 IoU-only、amp、structure、amp+structure、shuffled structure、shuffled amp+structure 等。结果非常关键：所有稳定组的 AP50 几乎都在 0.884-0.886，预测数基本 129-131，Recall 0.9011，高置信 FP 2。比如 null_no_update clean AP50 0.8847，signed_iou 0.8860，signed_amp 0.8852，signed_structure 0.8860，signed_amp_struct 0.8853-0.8858，shuffled controls 也在同一区间。

从 AP75 看，signed_amp clean AP75 为 0.6640，略高于 null_no_update 0.6435；edge AP75 为 0.5831，高于 null 的 0.5609。但 structure、shuffled structure、amp+structure 也能达到类似 edge AP75，且 AP50 差别极小。因此不能得出“amp 有因果提升”的结论。更严格地说，Round 2.5 表明：稳定 RLVR shell 可以跑，且某些指标有轻微提升，但 real spectral 与 shuffled 或其他控制之间没有足够差距，无法证明手工频域 verifier 起作用。

这背后有一个数值矛盾。MVP 的 R_amp TP/FP 绝对 gap 只有约 0.008，虽然 AUC 高，但经过 reward 归一化、policy_weight=0.0003、KL=10 的强约束后，信号进入梯度的有效幅度极小。提高 policy_weight 会使 detector score 分布崩塌；降低 policy_weight 则信号被 KL anchor 中性化。检测任务中的 NMS、proposal matching、ROI crop 位移又进一步放大了 verifier 噪声。由此得到一个阶段性负结论：在 Penn-Fudan 检测上，手工 box-level spectral verifier 不能成为有效 RLVR reward。

## 9. MPLSeg 启发下的 AFM/FFT 网络内部路线

用户提出一个重要改进：R_amp 只有幅度没有相位，应该结合 MPLSeg 的思想，把 magnitude-aware 与 phase-sensitive 解耦。最初我们把这理解为在 verifier 中加入结构分支，例如 Sobel、Laplacian、low-frequency phase stats。但进一步分析后，项目尝试将傅里叶操作移到网络内部，设计 MicroAFM/AFM 模块，让 FFT 不再只是外部标量 reward，而成为 feature transform 或 gate。

Round 2.6 的 AFM 普通实现带来明显损伤。baseline full AP50 为 0.8850，AP75 0.6451，precision 0.6667，recall 0.9011；AFM full 256 AP50 降至 0.8527，AP75 降至 0.4093，precision 降至 0.4385，high-conf FP 增至 14，prediction count 增至 187。说明在预训练 two-stage detector 的 ROI head 中插入 FFT 模块会扰乱特征分布。

Round 2.7 修复为残差 identity 设计，tanh scale 初始为 0，使模块初始接近 no-op。结果 AP50 约 0.8761，与 baseline 0.8770 接近；但 AP75 从 0.6524 降至 0.5367，precision 从 0.6667 降至 0.5226，prediction count 从 123 增至 155。也就是说，残差 identity 能保住 AP50，但仍损伤定位质量和 score distribution。

Round 2.8 进一步做 AFM diagnostics。关键结果是：`round28_g08_identity_current_afm_box_head` AP50 0.8653，AP75 0.7378，ECE 0.0283，看起来非常好；`round28_g09_identity_delta_afm_box_head` AP75 0.7374，也很好。但诊断显示 `mag_scale=0`、`phase_scale=0`，只有 residual/box_head 适配在起作用。AFM-only 组崩塌，AP50 只有 0.0515，prediction count 1682。由此可知，AP75 提升不是 FFT gate 激活，而是 box head 训练适配、残差路径或训练轮次带来的效果。

这一阶段的结论非常重要。MPLSeg 的幅度/相位思想本身没有错，但它在 semantic segmentation 中有 dense mask 和 decoder 结构；直接迁移到 Faster R-CNN ROI head，会遇到预训练特征分布错配、ROI crop 对齐噪声、FFT/iFFT 引入统计漂移等问题。AFM 作为 in-network FFT 组件在 Penn-Fudan two-stage detector 上未能证明有效。

## 10. Round 2.9 与 2.10：公平性、checkpoint 与 edge-mix 控制

在 AFM 之后，Plan 2.9 讨论了是否继续做 fair AFM 控制和 post-training sanity。分析认为，Part A 的 noop_wrapper、AFM-frozen、AFM-trainable、strict parity 等公平性控制有一定价值，但 Round 2.8 已经回答核心问题：FFT 门控未激活，结果来自残差跳过与 head 适配。Part B 的 8 组 post-training sanity 不应执行，因为手工频谱 verifier 在 Penn-Fudan 检测上已经多轮无因果信号，再做同类 loss+verifier term 只会消耗 GPU。

用户随后提醒“那我后训练的部分呢”，于是 Plan 2.9/2.10 被修正为保留必要后训练维度，但不再重新烧一大组无意义实验。Round 2.10 修复了 checkpoint loading 与 edge-mix 评估。关键结果包括：`round210_b1_ckpt_eval` AP50 0.8198，AP75 0.6100，precision 0.6552，recall 0.8352，ECE 0.1099；`round210_b2_posttrain` AP50 0.8149，AP75 0.5252，precision 0.7353，recall 0.8242，ECE 0.0704，high-conf FP 从 5 降到 3；`round210_g9_rpn_mixed` AP50 0.8707，AP75 0.6295，recall 0.8901，ECE 0.0454。

这些结果说明，后训练和 edge-mix 可以稳定运行，也能改善 precision/ECE，但 AP75 可能下降，且改善来自 supervised continuation 或 RPN/box_head 适配，而不是频域 reward。Round 2.10 的价值是工程闭环：之前某些 checkpoint eval 异常被修复，后续再讨论 VOC/COCO 或 segmentation 才有可信的基线。

## 11. Plan 3.x：大规模搜索与阶段性总结

Plan 3.0 原计划做更大规模 Amp causality 与超参数搜索。它提出 Phase A 68 trials、Phase B 54 trials、Phase C 24 trials，总计 146 trials，目标是在 stable Round 2.3/2.5 shell 上判断 real amp 是否优于 shuffled amp，并搜索 reward_lambda、policy_weight、KL、structure 组合。理论上，这能给出更强的统计结论。

但结合 Round 2.3-2.5 结果，用户提出一个更尖锐判断：既然 stable shell 已经证明可行，而手工频谱 verifier 在 Penn-Fudan 上 8 组 AP50 差距小于 0.001，继续搜索参数可能没有意义。核心矛盾是：提高 policy_weight 会崩，降低又使信号中性化。此时更合理的选择有三条：换更大/更复杂数据集；用学习式 verifier 替代手工特征；接受当前结论写负结果论文。

Plan 3.1 和 3.2 的讨论进一步总结了阶段证据。外部 verifier RLVR 路线得到稳定 shell 但 reward 信号弱；in-network FFT 路线得到 AP50 可打平的 residual identity，但 AP75/precision 退化或 gate 不激活；继续在 Penn-Fudan 上堆参已经边际收益很低。于是下一步不应是盲目大搜，而是先写报告、识别真正缺口，再决定是扩展 VOC/COCO，还是迁移到 segmentation。

## 12. 为什么 Amp-only 经常优于 Struct

用户问过一个关键问题：为什么 Amp-only 优于 struct？这个问题可以从信号稳定性解释。

Amplitude profile 是把 ROI crop 的频谱能量按半径聚合成一维分布。它丢弃了方向和绝对空间位置，保留低频到高频的能量比例。因此它对轻微框偏移、目标姿态变化和边缘裁剪有一定鲁棒性。对于 Penn-Fudan 行人，TP 框往往包含完整人体轮廓和相似背景比例，FP 框可能包含碎片、背景纹理或局部图案，所以 amplitude profile 的排序 AUC 较高。

Struct/phase 分支则更敏感。完整相位谱对空间平移极其敏感；即使只取 low-frequency phase stats 或 Sobel edge density，检测框的一两个像素偏移、人体边缘被裁剪、背景线条进入 ROI，都会改变结构特征。目标检测候选框不是 dense mask，ROI crop 不保证与 GT 精确对齐；而 phase/structure 恰恰最依赖对齐。因此在 Penn-Fudan 这种小数据中，struct 特征更容易把定位误差、裁剪边界和背景边缘当成“结构差异”，噪声比幅度更大。

MPLSeg 中 phase-sensitive 学习可行，是因为它工作在 segmentation encoder-decoder 的 dense feature/mask 空间，有像素级监督和稳定空间网格。我们在 detection 中用 ROI crop 比较 phase/edge，是把一个需要密集对齐的信号放在候选框级别，因此不稳定。换句话说，Amp-only 优于 Struct 不是因为相位不重要，而是因为当前检测任务的观测形式不适合直接使用相位。

## 13. Plan 2.11：从 Penn-Fudan 走向 VOC 的门控验证

用户提出“会不会是任务简单了的原因”。这是合理质疑。Penn-Fudan 只有 person 单类，样本少，背景和目标结构相对简单；Faster R-CNN 在这种任务上很容易通过预训练特征和 box head 适配达到高 AP。若任务太简单，频域 verifier 的作用空间确实可能被压缩。

因此 Plan 2.11 被修复为 VOC spectral signal gate，而不是立即做完整 VOC/COCO 后训练矩阵。修复后的计划只做 3 个验证单元：E1 训练/eval 一个 VOC 3 类 baseline，E2 测量 `R_amp` TP/FP gap 与 shuffled AUC，E3 写出 go/no-go summary。固定 gate 为：

`tp_fp_gap > 0.02` 且 `auc_real - auc_shuffled >= 0.03`

只有满足该条件，才继续做 VOC/COCO detection post-training。如果 gap 仍接近 Penn-Fudan 的约 0.008，说明更复杂数据也没有救回手工傅里叶 verifier；如果样本数不足，则仅扩大 val limit，不添加新 reward 设计。

当前仓库中已有早期 VOC baseline 结果：`round211_voc_baseline` AP50 0.7086，AP75 0.3366，precision 0.3194，recall 0.8000，prediction count 864，num_gt 345。这个结果说明 VOC 子集更难、FP 更多、类别和背景更复杂，确实更适合测试频谱 gap。但由于旧 Plan 2.11 存在核心脚本缺口，新的修复版先把问题收窄为 spectral signal gate。这个决策比直接跑十几组 VOC post-training 更科学，因为它先验证 reward 信号是否存在。

Plan 2.11 后续已经补齐了一组 VOC 小规模结果。实验包含 V1 baseline eval、V2 detection-only post-training、V3 spatial verifier、V4 spatial+spectral loggate、V5 spatial+shuffled spectral。结果显示，VOC 任务确实比 Penn-Fudan 更难：V1 baseline 的 precision 只有 0.3191，预测数达到 865，说明 FP 压力明显更大；但频谱 loggate 仍没有产生有效增益。V4 的 AP50 为 0.7724，AP75 为 0.3751，precision 为 0.3054，recall 为 0.8551；它没有超过 V2 detection-only 的 AP50 0.7734，也没有超过 V5 shuffled spectral 的 AP50 0.7742。换句话说，即使换到更复杂的 VOC 3 类检测子集，真实频谱分支也没有优于 shuffled control。

| Plan 2.11 组别 | 模式 | AP50 | AP75 | Precision | Recall | ECE | High-conf FP | 预测数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 baseline eval | eval_only | 0.7084 | 0.3366 | 0.3191 | 0.8000 | 0.0485 | 16 | 865 |
| V2 detection-only | detection_only | 0.7734 | 0.3771 | 0.3067 | 0.8580 | 0.0584 | 16 | 965 |
| V3 spatial | spatial | 0.7724 | 0.3753 | 0.3060 | 0.8551 | 0.0581 | 16 | 964 |
| V4 spatial+spectral | spatial_spectral_loggate | 0.7724 | 0.3751 | 0.3054 | 0.8551 | 0.0591 | 16 | 966 |
| V5 spatial+shuffled | spatial_shuffled_spectral | 0.7742 | 0.3765 | 0.3075 | 0.8609 | 0.0558 | 16 | 966 |

Plan 2.11 的判定因此从“先测 gap”进一步变成“更难检测子集也没有救回手工频谱 verifier”。V4 没有超过 V3，且 shuffled V5 还略好于 V4，所以不能声称频域 evidence 对 VOC 检测后训练有因果贡献。这个结果强化了前面的负结论：在 two-stage detection 的 box-level reward 路线中，手工傅里叶证据更适合作为诊断或校准信号，而不是继续作为 detector 后训练 reward 的核心。

## 14. Plan 2.12：MPLSeg-style AFM 梯度修复

Plan 2.12 没有继续做 VOC/COCO 后训练矩阵，而是针对 Round 2.8 暴露出的架构根因做修复。旧 AFM 的 `mag_scale` 和 `phase_scale` 初始化为 0，同时 FFT 路径又被 `residual_scale=0` 乘住，导致幅度门控、相位残差和两个 scale 的梯度都被严格阻断。换句话说，Round 2.8 看到的 AFM box_head 收益并不来自频域门控学习，而来自 residual/box_head 适配。

Plan 2.12 按 MPLSeg 的思路重写 AFM：幅度分支使用硬激活 gate 和 InstanceNorm，形式上接近 `mag * (1 - gate(log(mag)))`；相位分支不再用可学习 scale 乘小残差，而是直接做 `phase + phase_residual`；模块只保留一个 `residual_scale` 控制 FFT 输出注入强度。这样做的目的不是把外部 R_amp reward 塞回 detector loss，而是先确认网络内部的 amplitude/phase 路径是否真的能获得梯度。

单元测试结果支持这一点。`tests/test_round212_mplseg_afm.py` 共 8 项全部通过，包括输出形状、非恒等初始化、无 NaN、幅度门控卷积梯度非零、相位残差卷积梯度非零、`residual_scale` 梯度非零和 factory 构造测试。测试运行只有一个 pytest cache 写入警告，不影响测试结果。

三组 1 epoch Penn-Fudan 对照结果如下：

| Plan 2.12 组别 | AFM 类型 | AP50 | AP75 | Precision | Recall | ECE | High-conf FP | 预测数 | AFM scale 读数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | none | 0.8010 | 0.5373 | 0.6981 | 0.8132 | 0.0742 | 3 | 106 | N/A |
| identity AFM | old scale-gated AFM | 0.8631 | 0.4375 | 0.4850 | 0.8901 | 0.0985 | 8 | 167 | `mag=0, phase=0, residual=-0.0727` |
| MPLSeg AFM | active MPLSeg-style AFM | 0.8678 | 0.6534 | 0.5586 | 0.8901 | 0.0666 | 2 | 145 | `residual=0.9720` |

这个结果改变了 AFM 路线的判断边界。与 identity AFM 相比，MPLSeg AFM 的 AP50 略高，AP75 从 0.4375 升到 0.6534，ECE 从 0.0985 降到 0.0666，高置信 FP 从 8 降到 2，并且 `residual_scale` 明显非零，说明 FFT 路径确实被激活。与同组 baseline 相比，它提升 recall 和 AP，但 precision 仍低于 baseline，预测数从 106 增至 145，说明它还没有完全解决检测 score 分布和 FP 控制问题。

因此 Plan 2.12 的准确结论是：MPLSeg-style 门控修复了旧 AFM 的梯度死锁，证明“网络内部频域路径无法学习”这个最强负命题不成立；但它还没有反转前面对手工 box-level frequency reward 的负结论。下一步若继续检测路线，应把它作为 architecture ablation 做多 seed、多 epoch、clean/edge stress 控制；若继续 RLVR/verifier 路线，则更合理的主战场仍是语义分割。

## 15. 迁移到语义分割的理由

在多轮检测实验后，用户提出“我们迁移到语义分割领域吧”。这不是逃避失败，而是基于证据的任务迁移。检测任务中的频域 reward 有三个结构性困难：

1. 候选框是稀疏对象，reward 依赖预测框与 GT 框匹配，NMS 和 score threshold 会改变候选集合。
2. ROI crop 与 GT crop 存在位移和尺度差，幅度还能容忍，phase/structure 难以容忍。
3. detector score 分布脆弱，轻微 policy loss 就可能造成 prediction explosion 或 recall collapse。

语义分割则天然更适合可验证奖励。Mask IoU、Dice、Boundary F1、foreground/background pixel error 都是 dense verifier；幅度可以在 foreground mask 区域聚合，phase/edge 可以和边界、结构保持一致；KL anchor 可以作用在 pixel logits 上，而不必处理 NMS 和 proposal matching。MPLSeg 的 magnitude/phase 解耦也本来就是为 segmentation 设计，其 inductive bias 与 dense mask 更契合。

Plan 4.0 因此提出新包 `spectral_segmentation_posttrain`，训练 Penn-Fudan binary person segmentation baseline，再运行 KL-stabilized signed RLVR，reward 包括 Dice/IoU、Boundary F1、foreground amplitude consistency、phase/edge structure consistency 和 shuffled controls。Plan 4.1 进一步明确 7 组实验：baseline eval、supervised-only posttrain、spatial、amp、structure、spatial+amp+structure、shuffled amp+structure。Promotion rule 是：S6 必须超过 S3，且 S7 不能匹配 S6，才能声称 spectral evidence 有因果价值。

这个迁移并不否定检测实验。相反，检测实验提供了必要教训：不要把外部频谱标量直接乘 loss；要区分 calibration/reranking 与本体后训练；必须做 shuffled control；必须用 KL anchor 和 signed objective；必须记录 prediction count、ECE、high-conf FP 等稳定性指标。这些经验会直接进入 segmentation RLVR 的设计。

## 16. 总体结果表

以下表格提炼本项目最关键的实验读数。

| 阶段 | 方法 | AP50 | AP75 | Precision | Recall | ECE | 结论 |
|---|---:|---:|---:|---:|---:|---|
| MVP baseline clean | Faster R-CNN 1 epoch | 0.8630 | N/A | 0.6838 | 0.8791 | N/A | 可用 baseline |
| MVP reward posttrain clean | image-level reward-weighted 5 epoch | 0.4349 | N/A | 0.6885 | 0.4615 | N/A | high-conf FP 消失但 recall 崩塌 |
| QH rerank clean alpha=0.9 | ROI+spectral quality head | 0.8741 | N/A | 0.7043 | 0.8901 | 0.1663 | 校准提升，AP 基本持平 |
| NNI QH best | ROI-only, alpha=0.7 | 0.8595 | N/A | 0.6752 | 0.8681 | 0.0656 | ROI-only 最优，频域无明确增量 |
| Round 2.3 stable shell | signed IoU, KL=10 | 0.8726 | 0.6582 | 0.6480 | 0.8901 | 0.0494 | RLVR shell 稳定 |
| Round 2.5 amp | signed amp, KL=10 | 0.8852 | 0.6640 | 0.6260 | 0.9011 | 0.0439 | 指标微升但与 controls 不分 |
| Round 2.6 AFM full | in-network FFT | 0.8527 | 0.4093 | 0.4385 | 0.9011 | 0.1026 | AP75/precision 受损 |
| Round 2.8 AFM box_head | identity AFM + box head | 0.8653 | 0.7378 | 0.6154 | 0.8791 | 0.0283 | gate 未激活，收益来自 head 适配 |
| Round 2.10 posttrain | checkpoint posttrain | 0.8149 | 0.5252 | 0.7353 | 0.8242 | 0.0704 | precision/ECE 改善，AP75 降 |
| VOC baseline | VOC 3 类 baseline | 0.7086 | 0.3366 | 0.3194 | 0.8000 | 0.0472 | 更难任务，可用于 signal gate |
| Plan 2.11 V4 | VOC spatial+spectral loggate | 0.7724 | 0.3751 | 0.3054 | 0.8551 | 0.0591 | 未超过 spatial 和 shuffled control |
| Plan 2.11 V5 | VOC shuffled spectral | 0.7742 | 0.3765 | 0.3075 | 0.8609 | 0.0558 | shuffled 略优，频谱因果不成立 |
| Plan 2.12 identity AFM | old scale-gated AFM | 0.8631 | 0.4375 | 0.4850 | 0.8901 | 0.0985 | mag/phase 仍为 0，旧门控死锁 |
| Plan 2.12 MPLSeg AFM | active MPLSeg-style AFM | 0.8678 | 0.6534 | 0.5586 | 0.8901 | 0.0666 | residual=0.9720，FFT 路径被激活 |

## 17. 失败并不等于没有成果

这个项目最有价值的部分，是它没有停留在“某个指标没涨所以失败”的层面，而是逐步拆解了失败原因。

第一，R_amp 有统计信号但绝对尺度压缩。TP 和 FP 的 R_amp 均接近 1，差值只有千分量级。AUC 高说明排序有用，但 reward 学习需要可形成稳定梯度的数值差异。作为 oracle verifier 或 rerank signal，R_amp 可以有意义；作为 detector policy reward，它太弱。

第二，检测后训练比分类后训练脆弱。分类模型每张图只有一个标签，view consistency 容易定义；检测模型有多个候选框、NMS、score threshold、box regression 和 matching。任何 reward 如果不能绑定到具体 ROI action，就容易变成全局 loss 缩放，导致 conservative collapse 或 prediction explosion。

第三，频域结构信号需要对齐。相位和结构对空间位移敏感，在 ROI crop 上噪声大；如果没有 dense mask 或稳定 feature grid，结构 verifier 容易误伤定位微小偏移。

第四，calibration 和 AP 是不同目标。Quality head 和 RLVR 多次显著改善 ECE、高置信错误或 precision，但 AP50/AP75 不一定提升。研究叙事不能强行把校准收益包装成 AP 提升，而应明确说明：当前最可靠收益在校准和错误抑制，尚未形成鲁棒 AP 提升。

第五，negative result 有研究价值。我们证明了一条看似合理的路线 - “ROI FFT 相似度作为检测后训练 reward” - 在小规模 two-stage detector 上并不成立，并给出原因、控制和可复现实验。这比只报告一个成功 reranking 更诚实，也更能指导下一步。

## 18. 后训练技术总结

本项目最终沉淀出一套视觉后训练技术原则。

1. Verifier 必须绑定到 action。检测任务中的 action 不是整张图，而是候选框的类别和坐标。reward 如果只作用到整图 loss，会改变 loss 强度而不是行为方向。

2. 低 reward 应降低当前动作概率。正权重 CE 只会把 FP 推向 background，但不是 GRPO 风格的 signed objective。Signed advantage 让高 reward 增强当前动作，低 reward 抑制当前动作。

3. KL anchor 是必要的。没有 KL 或 policy_weight 太大，detector score distribution 很容易崩。Round 2.3 的稳定配方是 KL=10、policy=0.0003、frozen baseline、cls-only、signed objective。

4. 冻结参数不等于冻结状态。BatchNorm running stats、dropout mode、train/eval 状态都会影响结果。RLVR 后训练必须控制 frozen modules 的 eval mode，并记录 initial KL sanity。

5. Shuffled control 必不可少。任何频域信号都必须与 shuffled amp、shuffled structure、random amplitude 对照，否则无法判断收益来自真实频域因果，还是来自训练扰动、head adaptation 或 seed。

6. 指标必须覆盖稳定性。AP50 不够，必须看 AP75、Recall、Precision、ECE、High-conf FP、num_predictions、miss rate、fixed-recall precision、score-IoU correlation。

7. 先判断信号，再做大矩阵。Plan 2.11 的修正体现了这一点：先测 VOC 频谱分支是否超过 shuffled control；若不过，不要盲目扩展 VOC/COCO detection 后训练。Plan 2.12 则说明另一个原则：当 verifier 失败时，要区分“信号无效”和“架构根本没有梯度”，先做梯度诊断再谈后训练。

## 19. 下一步建议

短期建议分两条并行。

第一，把 Plan 2.12 作为检测 AFM 路线的补充证据，而不是立刻扩大成新的大矩阵。最小后续验证应是多 seed、多 epoch、clean/object-edge/near-object stress 对照，并同时报告 AP75、precision、ECE、high-conf FP、prediction count 和 `residual_scale` 轨迹。只有 MPLSeg AFM 在这些控制下稳定优于 identity AFM，才能说 in-network amplitude/phase 路线重新获得检测价值。

第二，推进 Plan 4.1 segmentation spatial-spectral post-training。优先做 Penn-Fudan binary mask smoke：baseline 1 epoch，posttrain 1 epoch，评估 clean、checkerboard、object-inside、boundary patch。核心对照是 spatial-only vs spatial+amp+structure vs shuffled spectral。只有 S6 超过 S3 且 S7 不匹配 S6，才声称 spectral causality。

中期建议是写论文式负结果报告。论文主线可以是：我们提出一个检测 RLVR shell，系统评估 handwritten frequency verifier 与 in-network FFT adaptation；结果显示 RLVR shell 稳定可行，但 Penn-Fudan two-stage detection 中频域 verifier 因信号压缩、ROI 对齐噪声和预训练特征分布错配而无因果效果；进一步提出 segmentation 是更适合 MPLSeg 式幅度/相位 verifier 的任务。这样的报告比“我们调参没调出来”更有学术表达。

长期建议是学习式 verifier。手工 R_amp 的瓶颈是尺度压缩和对齐噪声。未来可训练一个 mask/ROI verifier，用 dense masks、edge maps、feature-level ROI 和 patch hard negatives 学习 reward；但在学习式 verifier 之前，必须先有可靠的 reward validation：oracle 上限、random/shuffled control、跨数据集泛化和固定 recall 下错误抑制。

## 20. 参考资料

1. Visual-RFT: Visual Reinforcement Fine-Tuning, arXiv 2503.01785, https://arxiv.org/abs/2503.01785
2. Tulu 3: Pushing Frontiers in Open Language Model Post-Training, arXiv 2411.15124, https://arxiv.org/abs/2411.15124
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv 2402.03300, https://arxiv.org/abs/2402.03300
4. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, arXiv 2501.12948, https://arxiv.org/abs/2501.12948
5. TorchVision Object Detection Finetuning Tutorial, Penn-Fudan dataset description, https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
6. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv 1506.01497, https://arxiv.org/abs/1506.01497
7. Mask R-CNN, arXiv 1703.06870, https://arxiv.org/abs/1703.06870
8. Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection, NeurIPS 2020, https://papers.nips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html
9. Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection, CVPR 2021, https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.pdf
10. AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty, ICLR 2020, https://openreview.net/pdf?id=S1gmrxHFvB
11. Decoupling semantic and localization for semantic segmentation via magnitude-aware and phase-sensitive learning, MPLSeg, https://www.sciencedirect.com/science/article/pii/S1566253524000927
12. PyTorch `torch.fft.fft2` documentation, https://docs.pytorch.org/docs/stable/generated/torch.fft.fft2.html

## 附录 A：阶段迭代清单

| 阶段 | 用户意图或问题 | 我们的调整 | 结果 |
|---|---|---|---|
| 分类初稿 | 多源视觉反馈后训练 | CIFAR-100 + ViT/DeiT + Fourier/Patch views | 被用户纠正，任务不是分类 |
| 任务重构 | 目标检测 RLVR | Faster/Mask R-CNN + verifier + post-training | 形成 detection RLVR 主线 |
| MVP | Penn-Fudan 最小验证 | Baseline + R_amp TP/FP + reward posttrain | R_amp AUC 高，posttrain 崩 recall |
| QH | 不直接乘 detector loss | Spectral Quality Head rerank | ECE/precision 改善，频域无明确增量 |
| NNI QH | alpha 与 head 消融 | 108 trials | ROI-only 最优，确认 calibration 价值 |
| RLVR Round 1 | 真正后训练 | rollout + reward + NNI | AP 降，ECE 改善 |
| Round 2.1 | 修 bug 稳定 | percentile R_amp、shuffled、threshold sync | 预测爆炸仍存在 |
| Round 2.2 | signed objective | KL + signed advantage + frozen baseline | score 分布仍漂移 |
| Round 2.3 | 读数与状态修复 | result schema、BN eval、initial KL | stable shell 成立 |
| Round 2.5 | 频谱因果检验 | amp/struct/shuffled controls | real 与 shuffled 不分 |
| Round 2.6 | MPLSeg in-network | AFM FFT 模块 | AP75/precision 损伤 |
| Round 2.7 | identity residual | scale=0 no-op 初始化 | AP50 接近，AP75 仍降 |
| Round 2.8 | AFM diagnostics | frozen parity、afm-only、afm-boxhead | gate 未激活，收益来自 head |
| Round 2.10 | 后训练补洞 | checkpoint 与 edge-mix 修复 | 后训练稳定但非频域因果 |
| Plan 2.11 | 是否任务太简单 | VOC spectral signal gate | 先测 gap，再决定后训练 |
| Plan 2.12 | AFM 梯度死锁 | MPLSeg-style active gate + gradient tests | FFT 路径恢复梯度，仍需多 seed 验证 |
| Plan 4.x | 迁移到分割 | Dense mask RLVR + spatial/spectral verifier | 下一主线 |

## 附录 B：核心结论的最短版本

如果只保留一句话，结论是：

本项目已经证明视觉检测 RLVR shell 可以稳定实现，但手工 ROI 傅里叶 verifier 在 Penn-Fudan/VOC two-stage detection 中没有足够因果信号；Plan 2.12 进一步证明 MPLSeg-style AFM 可以修复旧 FFT 模块的梯度死锁，但这仍不同于证明 box-level frequency reward 有效，因此下一主线应迁移到更适合幅度/相位解耦的语义分割任务。
