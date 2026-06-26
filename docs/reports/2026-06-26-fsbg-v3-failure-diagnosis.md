# FSBG v3 失败原因诊断

## 核心结论

FSBG v3 没有显著提升，是因为它在训练后**退化成了一个近似的 uniform feature scaler**，而不是预期的空间边界门控。

具体表现为：
- `alpha` 学习为**负值**且绝对值较大（-0.25 ~ -1.17）
- `gate` 大量饱和在 **-1.0**（一个 seed 中 71% 的 gate 值 < -0.95）
- gate 的空间方差很小，残差项 `alpha * gate * x` 近似为一个常数缩放

## 实验数据

### 最终检测指标（NWPU 50ep, bs=16, lr=0.024）

| 方法 | mean AP50 | mean AP75 | mean Precision | mean Recall | mean Miss Rate |
|---|---|---|---|---|---|
| Baseline v3 | 0.2005 | 0.0742 | 0.3920 | 0.2464 | 0.7536 |
| FSBG v3 | 0.1852 | 0.0667 | 0.4062 | 0.2366 | 0.7634 |

FSBG v3  precision 略高，但 recall 更低，miss rate 更高，最终 AP 下降。

### 训练后 FSBG 模块参数

| seed | alpha | fusion_bias | gate mean | gate std | gate < -0.95 | effective scale ≈ 1+α·gate |
|---|---|---|---|---|---|---|
| 42 | -1.1662 | -0.0347 | -0.9920 | 0.0180 | 71.1% | ~2.16x |
| 123 | -0.2458 | -0.0206 | -0.8365 | 0.1738 | — | ~1.21x |
| 2024 | -0.8209 | -0.0386 | -0.9418 | 0.0711 | — | ~1.77x |

*effective scale* 估算：输出 `y = x + alpha * gate * x ≈ (1 + alpha * gate) * x`。

seed 42 中 gate 几乎恒为 -1，alpha≈-1.17，因此模块相当于把 ROI feature 整体放大了约 2.16 倍。这不是空间选择性的边界增强，而是 uniform scaling。

## 为什么 gate 会饱和到 -1？

FSBG v3 的 gate 形式为：

```python
gate = tanh(spatial + phase + fusion_bias)
output = x + alpha * gate * x
```

其中：
- `tanh` 输出范围 `(-1, 1)`
- `alpha` 是无约束可学习标量
- 当 `alpha < 0` 且 `gate < 0` 时，`alpha * gate > 0`，相当于对特征做正向缩放
- 网络发现：与其学习复杂的空间边界门控，不如把 gate 压到饱和区 `-1`，然后调 `alpha` 来得到一个 uniform scale

这是更省力的优化路径。由于 phase boundary 信号本身弱（与 Sobel 边缘相关性仅 ~0.3），网络没有动机去维持 gate 的空间变化。

## 关键观察

1. **Gate 与边缘的相关性虽然存在，但不是驱动因素**
   - seed 42 中 `corr(gate, Sobel gradient) = -0.56`
   - 但 71% 的 gate 已经饱和在 -1，相关性主要由少数未饱和点贡献
   - 整体上 gate 没有提供有意义的空间调制

2. **Alpha 无约束导致退化**
   - alpha 初始为 `1e-2`，最终变为 `-0.25 ~ -1.17`
   - 如果 alpha 被限制为正小值，网络就不能用“负 gate + 负 alpha”来作弊

3. **FSBG v3 的 gate 动态范围虽大，但没用对地方**
   - 探针显示 gate std 可达 0.12（比 v2 的 0.009 好）
   - 但训练后网络选择把 gate 压平到饱和区
   - 说明空间边界检测这个任务本身对检测 loss 的边际贡献太小

## 与 LSG 的对比启示

| | FSBG v3 | LSG v1 |
|---|---|---|
| 空间门控 | gate ∈ (-1,1), 可饱和 | gate = 1 + tanh(...), 中心在 1 |
| alpha | 无约束，可学为负 | 残差修正 `x + α(x_filt - x)`，α=0.1 固定小值 |
| 初始状态 | gate≈0, 输出≈x | gate≈1, x_filt≈x, 严格 identity |
| 操作对象 | 空间边缘图 | 频域幅度谱 |

LSG 通过以下设计避免 FSBG 的退化：
1. 初始严格 identity，不会破坏预训练分布
2. 残差修正形式 `(x_filt - x)` 强迫模块学习“差异”而不是“缩放”
3. alpha 初始小且为正，不会变成全局缩放因子
4. 直接对频域幅度谱操作，避免弱相位重建

## 结论

FSBG v3 的失败不是梯度问题，也不是 gate 动态范围问题，而是**优化目标让网络找到了最简单的捷径：把 gate 压到饱和区，用 alpha 做 uniform scaling**。这解释了为什么 precision/recall/ECE 有波动但 AP 没有提升。

要验证“频域门控对检测有用”，必须让模块无法退化为 uniform scaler。LSG v1 的 identity 初始化和残差修正形式就是为了堵住这个退化路径。
