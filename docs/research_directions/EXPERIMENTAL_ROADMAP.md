# 端到端实验路线图

## 1. 目标

在真实数据集上验证 Plan A-F 各模块的有效性，产出可量化的指标对比，为后续论文/技术报告提供实验支撑。

## 2. 实验原则

1. **从快到慢**：先跑无需训练 backbone 的实验，再跑需要端到端训练的实验；
2. **从小到大**：先在小数据集/子集上做 smoke test，验证流程通顺后再跑完整实验；
3. **先防御后生成**：Plan B 对抗防御能最快验证核心思想，优先执行；
4. **模块可替换**：每个实验都要能方便地切换 baseline / 我们的方法；
5. **可复现**：固定 seed、`cudnn.benchmark=False`、记录完整配置。

## 3. 各方向实验设计

### 3.1 Plan B：对抗防御（最高优先级）

#### 数据集
- **Penn-Fudan Pedestrian**：训练 170 张，测试 ~150 张，单类别行人。

#### 模型
- **Faster R-CNN MobileNetV3-Large-FPN**（项目现有 baseline checkpoint）。

#### 攻击方法
- 无目标数字对抗补丁攻击（RP2 风格简化版）；
- 补丁大小：$40 \times 40$ 到 $80 \times 80$；
- 位置：图像中心偏下（模拟胸口/躯干位置）；
- 优化步数：200–500 步；
- 目标：最小化行人检测置信度。

#### 防御方法
- **Baseline 1**：无防御；
- **Baseline 2**：频域硬阈值（Hard Threshold）；
- **Baseline 3**：JPEG 压缩；
- **Ours**：`SpectralChordDefense`。

#### 评估指标
- $AP50_{\mathrm{clean}}$：干净图像；
- $AP50_{\mathrm{adv}}$：对抗图像；
- $AP50_{\mathrm{defended}}$：防御后对抗图像；
- $AP50_{\mathrm{clean\_defended}}$：干净图像经防御后；
- $Recovery = \frac{AP50_{\mathrm{defended}} - AP50_{\mathrm{adv}}}{AP50_{\mathrm{clean}} - AP50_{\mathrm{adv}}}$；
- $Clean\_Drop = AP50_{\mathrm{clean}} - AP50_{\mathrm{clean\_defended}}$。

#### Smoke Test
- 在 10 张测试图像上跑通攻击+防御流程。

#### 完整实验
- 全部测试图像；
- 3 个随机 seed；
- 消融：latent_dim、delta、anomaly_threshold、patch_size。

---

### 3.2 Plan C：语义分割

#### 数据集
- **Penn-Fundan Pedestrian 分割**：二分类（人/背景）；
- **Pascal VOC 2012**：20 类分割（扩展验证通用性）。

#### 模型
- **FCN-ResNet50**；
- 在 layer3/layer4 后插入 `ManifoldAFMBlock`。

#### 对比方法
- **Baseline**：标准 FCN-ResNet50；
- **mid06 AFM**：现有 MPLSegAFMBlock；
- **Ours**：`ManifoldAFMBlock`；
- **Ours + OT Loss**：`ManifoldAFMBlock` + `OTSegmentationLoss`。

#### 评估指标
- mIoU；
- Boundary IoU；
- ECE（Expected Calibration Error）；
- 训练收敛速度。

#### Smoke Test
- Penn-Fudan 上 1 epoch，验证模块能训练且不崩溃。

#### 完整实验
- Penn-Fudan：20 epoch；
- VOC：3–5 epoch（数据集更大）；
- 3 seed 均值。

---

### 3.3 Plan D：图像分类

#### 数据集
- **CIFAR-100**：100 类，32×32；
- **CIFAR-10**（可选 smoke test）。

#### 模型
- **ResNet-18 / ResNet-50**；
- 替换最后一层分类头为 `SpectralClassifierHead` 或 `OTPrototypeClassifier`。

#### 对比方法
- **Baseline**：标准 ResNet + 线性头；
- **SpectralHead**：magnitude/phase 解耦头；
- **OT Prototype**：Sinkhorn 原型分类器；
- **SpectralMixup**：OT 频域 mixup 数据增强。

#### 评估指标
- Top-1 / Top-5 Accuracy；
- CIFAR-100-C corruption robustness（可选）；
- 训练时间。

#### Smoke Test
- CIFAR-100 上 10 epoch。

#### 完整实验
- 200 epoch；
- 标准数据增强；
- 3 seed 均值。

---

### 3.4 Plan E：多模态图像-文本对齐

#### 数据集
- **Flickr30k 子集**（约 1000 张图）；
- 或 **COCO Captions 子集**。

#### 模型
- 冻结 **CLIP ViT-B/32** 图像/文本 encoder；
- 训练 `OTImageTextAlignment` 头或 `CrossModalTransport`。

#### 对比方法
- **Baseline**：CLIP + InfoNCE；
- **Ours**：CLIP + OT Alignment；
- **Ours + CrossModalTransport**。

#### 评估指标
- Image-to-Text R@1/R@5/R@10；
- Text-to-Image R@1/R@5/R@10；
- 训练稳定性（loss 曲线）。

#### Smoke Test
- 1000 张图 + 5000 个 caption 对，训练 5 epoch。

#### 完整实验
- 全量 Flickr30k，50 epoch；
- 冻住 CLIP，只训练对齐头。

---

### 3.5 Plan F：遥感目标检测

#### 数据集
- **NWPU VHR-10**：10 类遥感目标检测；
- **VisDrone**（可选，无人机视角）。

#### 模型
- **Faster R-CNN MobileNetV3-Large-FPN**；
- 在 FPN/ROI 层插入 `RemoteSensingAFM`。

#### 对比方法
- **Baseline**：标准 Faster R-CNN；
- **mid06 AFM**：现有 MPLSegAFMBlock；
- **Ours**：`RemoteSensingAFM`；
- **Ours + RotationEquivariantFFT**。

#### 评估指标
- AP50 / AP75；
- 小目标 AP（按面积 < 32×32 过滤）；
- 旋转敏感类别 AP（飞机、船只）。

#### Smoke Test
- NWPU VHR-10 上 3 epoch。

#### 完整实验
- 10–20 epoch；
- 3 seed 均值。

---

### 3.6 Plan A：流形基础设施验证

Plan A 已通过单元测试，额外实验：
- 在 CIFAR-10 训练集上拟合 `ComplexSpectralManifold`，测量重构误差随 latent_dim 变化；
- 可视化学到的流形坐标（t-SNE/UMAP）；
- 验证 `ChordTransport` 在真实图像谱系数上的能量收缩。

---

## 4. 优先级与执行顺序

```
Week 1:
  Day 1-2: Plan B smoke test + 完整实验
  Day 3-4: Plan C smoke test
  Day 5-7: Plan C 完整实验

Week 2:
  Day 1-3: Plan D smoke test + 完整实验
  Day 4-5: Plan A 可视化验证
  Day 6-7: Plan B 消融与论文图

Week 3:
  Day 1-4: Plan F smoke test + 完整实验
  Day 5-7: Plan E smoke test

Week 4:
  Day 1-5: Plan E 完整实验
  Day 6-7: 汇总、写报告、整理图表
```

## 5. 资源需求

| 实验 | GPU 显存 | 预计时间 | 备注 |
|------|---------|---------|------|
| Plan B | 4–6 GB | 2–6 h | 攻击优化耗时 |
| Plan C | 6–8 GB | 6–12 h | 分割训练 |
| Plan D | 4–6 GB | 8–16 h | 分类训练 |
| Plan E | 6–8 GB | 4–8 h | 冻结 CLIP 较快 |
| Plan F | 6–8 GB | 8–16 h | 遥感检测训练 |

## 6. 评估标准

每个实验必须产出：
1. 指标表格（baseline vs ours）；
2. 消融表格（关键超参数）；
3. 至少 1 张可视化图；
4. 运行日志（loss/AP/accuracy 曲线）。

### 成功标准

| 方向 | 最低成功标准 |
|------|-------------|
| Plan B | 对抗 AP 恢复 ≥ 50%，干净 AP 损失 ≤ 5% |
| Plan C | mIoU 较 mid06 AFM 提升 ≥ 2% |
| Plan D | Top-1 准确率 ≥ baseline，或鲁棒性提升 ≥ 3% |
| Plan E | R@1 ≥ InfoNCE baseline |
| Plan F | AP50 较 baseline 提升 ≥ 1% |

## 7. 实验产物

```
runs/experiments/
├── plan_b_defense/
│   ├── smoke_test/
│   └── full/
├── plan_c_segmentation/
│   ├── smoke_test/
│   └── full/
├── plan_d_classification/
│   ├── smoke_test/
│   └── full/
├── plan_e_multimodal/
│   ├── smoke_test/
│   └── full/
└── plan_f_remote_sensing/
    ├── smoke_test/
    └── full/
```

每个子目录包含：
- `config.yaml`：实验配置；
- `metrics.json`：最终指标；
- `log.txt`：运行日志；
- `figures/`：可视化图表。

## 8. 下一步

确认此路线图后，立即开始实施 **Plan B smoke test**：
1. 写 `scripts/experiments/plan_b_smoke_test.py`；
2. 在 10 张 Penn-Fudan 测试图像上跑通攻击+防御；
3. 输出 AP 指标和可视化对比图。
