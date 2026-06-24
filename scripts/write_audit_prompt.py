import json

with open(r'E:\CLIproject\RLimage\scripts\round2112_manifold_dpo.py', 'r', encoding='utf-8') as f:
    code = f.read()

with open(r'E:\CLIproject\RLimage\scripts\round2108_dpo.py', 'r', encoding='utf-8') as f:
    base = f.read()

prompt = f"""# GPT-5.5 代码审计请求

## 审计目标
审计 RLIimage 项目的流形 DPO 实验脚本 scripts/round2112_manifold_dpo.py。

## 背景
- 该实验失败了：amp_lo per-channel stats manifold distance 在静态分析中达到 70% pair 一致率，但作为 DPO pair 选择信号后未产生训练增益（-0.004 AP75 vs fine-tune）
- 对比基准：round2108_dpo.py（IoU-based pair selection）成功达到 AP75=0.724
- 两个脚本除了 pair 选择方式不同，其他结构几乎相同

## 待审计代码（round2112_manifold_dpo.py）

```python
{code}
```

## 对比基准（round2108_dpo.py）

```python
{base}
```

## 审查点
1. Manifold 预处理 Pipeline 在训练中是否可靠？StandardScaler + PCA 是否用全局统计量？TP 簇中心是否随训练更新？
2. DPO pair 选择逻辑：chosen/rejected 是否正确？
3. 梯度流：whitening 操作是否切断了梯度？
4. 与 round2108 的差异：除了 pair 选择方式，还有什么不同？
5. 数据流：FFT 提取、per-channel stats、标准化是否正确？
6. 样本量：borderline proposal 每图有多少有效 pair？

## 请用中文给出
- 发现的 bug 列表（按严重度排序）
- 修复建议
- 这个方向是否值得修复后重试
"""

with open(r'E:\CLIproject\RLimage\scripts\audit_prompt.txt', 'w', encoding='utf-8') as f:
    f.write(prompt)

print(f"Prompt written, length: {len(prompt)}")
