# NWPU 上频域方法未提升的原因分析

## 当前实验结果

| 方法 | mean AP50 | mean AP75 | vs baseline AP50 |
|---|---|---|---|
| Baseline v3 | 0.2005 | 0.0742 | — |
| AFM mid | 0.1874 | 0.0669 | **-0.0131** |
| FSBG v3 | 0.1852 | 0.0667 | **-0.0153** |
| LSG no-radius s42（进行中）| ~0.23 | ~0.08 | 接近 baseline |

三种频域增强方法在 NWPU 上均未超过 baseline，甚至 AFM/FSBG 明显更差。

## 核心原因分析

### 1. ROI 特征分辨率太低（7×7）

当前所有方法都插在 box_head 之前，操作对象是 **7×7 的 ROI feature**。

对 7×7 做 rFFT2：
- 输出尺寸：7 × (7//2 + 1) = **7 × 4 = 28 个频率 bin**
- 频率分辨率极低，基本只能区分 DC、几个低频、几个高频
- 对于小目标和细粒度边界，28 个 bin 无法承载有意义的频谱结构

对比：
- Penn-Fudan 上 AFM 有效，但 Penn-Fudan 主要是行人，有强垂直边缘结构
- NWPU 是遥感小目标，目标形态多样，7×7 ROI 本身已经丢失了大量空间细节

**结论**：在 7×7 上学习频域门控，信息瓶颈太严重。

### 2. LSG 没有真正学会频率选择

检查 LSG no-radius seed 42（训练到 epoch 11 的 best checkpoint）：

```
alpha       = 0.0910   （几乎没动）
gate mean   = 1.0034
gate std    = 0.0151   （极小）
gate < 0.9  = 0.03%
gate > 1.1  = 0.00%
```

99.97% 的 gate 值都在 [0.9, 1.1] 之间，模块几乎还是 identity。

这说明：
- 网络没有从 LSG 中获得足够强的梯度信号去偏离 identity
- 或者偏离 identity 并不能降低检测 loss，所以网络选择不学

### 3. no-radius 版本先天不足

LSG no-radius 的输入只有 `log_mag`，没有 `radius_map`：

```python
mag_gate = Conv1x1(log_mag)
```

1×1 conv 的权重在所有频率位置共享，因此它**无法区分低频和高频位置**。它能学的只是：

- 某些通道的幅度整体放大/缩小
- 这等价于一个弱化的 SE block，不是真正的频率选择

即使它学会了一些东西，也只是 per-channel scaling，不是频域门控。

### 4. NWPU 目标尺度/形态差异太大

NWPU VHR-10 包含 10 类遥感目标：
- 飞机、船、储罐、棒球场、网球场、篮球场、跑道、港口、桥梁、车辆

这些目标的频谱特征差异巨大：
- 飞机：细长结构 + 机翼高频
- 储罐：圆形低频
- 桥梁：长条低频
- 车辆：小方块中频

一个共享的频域门控不可能同时优化所有类别的特征响应。相比之下，Penn-Fudan 只有行人一类，频谱结构更一致。

### 5. 数据量太小，难以学习额外参数

NWPU 训练集约 650 张图。LSG/AFM/FSBG 都在 ROI head 引入额外参数：
- LSG：约 C×C/4 + C×C ≈ 20K 参数
- AFM：类似量级
- 每个 ROI 的样本数虽然多，但监督信号只有最终 detection loss

小数据集 + 弱监督 = 学到的门控容易过拟合或干脆不学习。

### 6. 真实瓶颈可能在 RPN / Backbone，不在 ROI Head

频域方法都在 box_head 前修改 ROI feature。但如果 NWPU 的瓶颈是：
- RPN proposal 质量差（小目标召回低）
- Backbone FPN 特征不够 discriminative
- 分类头容量不足

那么修改 7×7 ROI feature 就无法解决根本问题。ROI feature 已经是 backbone 提取后的低维表示，其中的频率信息可能已经被破坏。

### 7. 指标方差大，小提升难以检测

baseline 3 seeds 的 AP50：
- seed 42: 0.2444
- seed 123: 0.1604
- seed 2024: 0.1967

cross-seed 极差达到 **0.084**。这意味着在 NWPU 上，±0.01 的 AP 差异可能完全是 noise。即使某个方法真实有效 +0.01，也容易被淹没在 seed 方差中。

## 对 LSG radius 的期望

LSG radius 加入了 `radius_map`，理论上可以学习低频/高频选择。但它仍然面临：
- 7×7 分辨率太低
- 类别差异太大
- 数据量小
- 监督信号弱

所以 LSG radius 可能略好于 no-radius，但**大幅提升的概率不高**。

## 下一步建议

如果 LSG radius 也没有显著提升，说明问题不在门控设计，而在**操作位置和分辨率**。可能方向：

1. **提高 ROI 分辨率到 14×14 或更高**
   - 更多 frequency bin，频域操作才有意义
   - 但会增加计算量

2. **把频域操作移到 FPN/backbone**
   - 在更高分辨率的 FPN feature 上做频域增强
   - 而不是等 ROI Align 压缩到 7×7 之后

3. **per-class 频域门控**
   - 不同类别学不同门控
   - 解决 NWPU 多类别频谱差异问题

4. **放弃 NWPU，回到 Penn-Fudan 验证核心机制**
   - 在 Penn-Fudan 上确认 LSG 有效
   - 再迁移到更复杂数据集

5. **直接做大模型/大数据集**
   - NWPU 可能太小，信号被 noise 淹没
   - VOC/COCO 上更容易验证方法是否有效

## 结论

NWPU 上频域方法不工作，**不是某个具体模块设计问题，而是这个任务-位置-分辨率组合本身不适合做频域门控**：
- 7×7 ROI 频率分辨率太低
- NWPU 多类别小目标频谱差异大
- 数据量不足以学习额外参数
- 真实瓶颈可能在 RPN/backbone

LSG no-radius 已经显示 gate 几乎没学动。LSG radius 值得跑完，但不应期待大幅超越 baseline。
