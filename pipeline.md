# RLIimage 自动化研究流水线

> 版本: 1.0 | 日期: 2026-06-15 | 适用项目: RLIimage (RLVR Post-Training for Object Detection)

---

## 1. 流水线架构

```
用户输入 Idea
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: 方向发现 (Direction Discovery)                       │
│ 1.1 文献调研 → 1.2 脑暴方向 → 1.3 查新 → 1.4 排名            │
│ 工具: /research-lit, /idea-creator, /novelty-check          │
│ 输出: IDEA_CANDIDATES.md (3-5 个方向, 含可行性预估)          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: 可行性验证 (Feasibility Verification)                │
│ 2.1 数学分析 (Codex/opus) + 2.2 论文调研 (k2.7/sonnet)        │
│ 2.3 交叉质疑 (并行评审)                                       │
│ 工具: 子 Agent 通信 (Codex critic + k2.7 explorer)           │
│ 输出: FEASIBILITY_REPORT.md (PASS / NEEDS_RISK / FAIL)      │
└─────────────────────────────────────────────────────────────┘
    │ 方向通过
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 3: 实验执行 (Experiment Execution)                     │
│ 3.1 写 runner → 3.2 检查 → 3.3 排队跑 → 3.4 聚合报告 → 3.5 bug 审计│
│ 工具: /experiment-bridge, /experiment-queue, /analyze-results│
│     /experiment-audit, /auto-review-loop                     │
│ 输出: EXPERIMENT_TRACKER.md + eval_metrics.json (per run)    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 4: 结论+迭代 (Conclusion & Iteration)                   │
│ 4.1 结构化证据 → 4.2 跨模型评估 → 4.3 生成下轮方向建议          │
│ 工具: /result-to-claim, /research-review, /idea-creator       │
│ 输出: CLAIMS_FROM_RESULTS.md + 下轮 IDEA_CANDIDATES.md        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ 回到阶段 1 (方向发现)
```

---

## 2. ARIS Skill 复用映射

### 2.1 直接复用 (无需改写)

| ARIS Skill | RLIimage 对应阶段 | 复用说明 |
|---|---|---|
| `/research-lit` | 1.1 文献调研 | 直接调用, 搜索 arXiv/Semantic Scholar/OpenAlex |
| `/idea-creator` | 1.2 脑暴方向 | 直接调用, 生成 8-12 个 idea |
| `/novelty-check` | 1.3 查新 | 直接调用, 验证 idea 是否已被发表 |
| `/experiment-plan` | 3.1 实验规划 | 直接调用, 将 refined proposal 转为实验路线图 |
| `/analyze-results` | 3.4 聚合报告 | 直接调用, 分析 JSON/CSV 结果, 生成对比表 |
| `/result-to-claim` | 4.1 结构化证据 | 直接调用, 判断结果支持哪些 claim |
| `/research-review` | 4.2 跨模型评估 | 直接调用, 外部评审员评估工作质量 |
| `/auto-review-loop` | 3.5 bug 审计 / 4.2 | 直接调用, 自动评审-修复循环 |
| `/ablation-planner` | 3.1 消融规划 | 直接调用, 主实验通过后设计消融 |

### 2.2 需要改写适配

| ARIS Skill | RLIimage 对应阶段 | 改写点 |
|---|---|---|
| `/experiment-bridge` | 3.1 写 runner + 3.2 检查 | **关键适配**: ARIS 的 bridge 默认写训练脚本; RLIimage 已有 `run_one(name, mode, seed)` 模板。改写为: 读取 `EXPERIMENT_PLAN.md` → 生成 `round[NNN]_[name].py` runner → 复用现有 `runner_utils` |
| `/experiment-queue` | 3.3 排队跑 | **关键适配**: ARIS 的 queue 面向 SSH 远程 GPU; RLIimage 在本地 8G GPU 运行。改写为: 本地进程队列 (subprocess.Popen), 支持 OOM 检测、seed 并行、checkpoint 依赖 |
| `/experiment-audit` | 3.5 bug 审计 | **适配**: 审计清单增加 RLIimage 特有问题: ① GT 是否来自 dataset 而非 model output ② AP75/AP50 计算是否用官方 COCO API ③ 能量门控是否 per-group 而非 per-sample ④ 基线 checkpoint 是否 frozen |

### 2.3 需要从零写

| 组件 | 阶段 | 说明 |
|---|---|---|
| **子 Agent 通信协议** | 2.3 交叉质疑 | 主 Agent spawn Codex(opus, critic) + k2.7(sonnet, explorer), 收集反馈做交叉质疑。无现成 ARIS skill, 需自研 |
| **RLIimage runner 模板生成器** | 3.1 写 runner | 基于现有 `round286_runner.py` 等, 提取公共模板, 根据 idea 自动生成新 runner |
| **本地实验队列调度器** | 3.3 排队跑 | 基于 `master_pipeline.py` 扩展: 支持多 seed 并行、OOM 自动重试、wave 依赖、状态持久化 |
| **结构化证据聚合器** | 4.1 | 读取所有 `eval_metrics.json`, 按 claim 组织证据链, 生成 `CLAIMS_FROM_RESULTS.md` |

---

## 3. 文件结构

```
RLimage/
├── pipeline.md                    # 本文档 (流水线设计)
├── scripts/
│   ├── pipeline_start.py          # 启动脚本 (接收 idea, 触发全链)
│   ├── pipeline_utils/            # 流水线工具库
│   │   ├── __init__.py
│   │   ├── agent_bridge.py        # 子 Agent 通信协议
│   │   ├── runner_generator.py    # 从 idea 生成 runner 模板
│   │   ├── local_queue.py         # 本地实验队列调度器
│   │   └── evidence_aggregator.py # 结构化证据聚合器
│   ├── roundXXX_runner.py         # 现有 runner (保持不动)
│   └── ...
├── pipeline_state/                # 流水线状态 (gitignore)
│   ├── current_idea.json          # 当前 idea 序列化
│   ├── stage_1_output/            # 方向发现输出
│   ├── stage_2_output/            # 可行性验证输出
│   ├── stage_3_output/            # 实验执行输出
│   └── stage_4_output/            # 结论迭代输出
└── .aris/                         # ARIS skill 输出 (如安装)
    └── ...
```

---

## 4. 启动流程详解

### 4.1 触发方式

```bash
# 方式 1: 命令行
python scripts/pipeline_start.py "idea: 用频域相位一致性作为 RLVR reward, 在 NWPU 上验证"

# 方式 2: 从文件
python scripts/pipeline_start.py --from-file idea.txt

# 方式 3: 恢复中断流水线
python scripts/pipeline_start.py --resume pipeline_state/current_idea.json
```

### 4.2 阶段流转

#### 阶段 1: 方向发现

1. **输入**: 用户 idea (一句话或一段描述)
2. **调用 ARIS `/research-lit`**: 搜索相关文献 (Visual-RFT, MPLSeg, RLVR, GRPO, 频域检测)
3. **调用 ARIS `/idea-creator`**: 基于文献和 idea, 脑暴 3-5 个具体方向
4. **调用 ARIS `/novelty-check`**: 对每个方向查新, 标记 `novel` / `published` / `marginal`
5. **输出**: `pipeline_state/stage_1_output/IDEA_CANDIDATES.md`

#### 阶段 2: 可行性验证

1. **输入**: `IDEA_CANDIDATES.md` 中排名前 2 的方向
2. **并行 spawn 子 Agent**:
   - **Agent A (Codex/opus, critic 角色)**: 深度数学分析 — 奖励函数是否可微、梯度链是否完整、方差是否可控
   - **Agent B (k2.7/sonnet, explorer 角色)**: 论文调研 — 找到最相关的 3 篇论文, 提取方法细节和失败教训
3. **交叉质疑**: 主 Agent 收集 A 和 B 的输出, 提出 3-5 个质疑问题, 分别回传要求回答
4. **综合判断**: 生成 `FEASIBILITY_REPORT.md`, 标记 `PASS` / `NEEDS_RISK` / `FAIL`
5. **输出**: `pipeline_state/stage_2_output/FEASIBILITY_REPORT.md`

#### 阶段 3: 实验执行

1. **输入**: `FEASIBILITY_REPORT.md` (PASS 的方向)
2. **调用改写后的 `/experiment-bridge`**:
   - 读取 `EXPERIMENT_PLAN.md` (由 `/experiment-plan` 生成)
   - 调用 `runner_generator.py` 生成 `round[NNN]_[name].py`
   - 复用现有 `runner_utils` (build_penn_fudan_loaders_320, decode_boxes, evaluate_model 等)
   - 自审查: 检查 `run_one(name, mode, seed)` 签名、GT 来源、AP75 计算
3. **调用改写后的 `/experiment-queue`** (本地版):
   - 生成 manifest.json (多 seed / 多 config)
   - 本地调度: subprocess.Popen, 最多 1 个 job (8G GPU 限制)
   - OOM 检测: 捕获 `torch.OutOfMemoryError`, 自动减半 batch size 重试
   - 状态持久化: `pipeline_state/stage_3_output/queue_state.json`
4. **调用 `/analyze-results`**: 所有 job 完成后, 聚合 `eval_metrics.json`, 生成对比表
5. **调用 `/experiment-audit`**: 交叉模型审计, 检查 GT 来源、score normalization、死代码
6. **输出**: `pipeline_state/stage_3_output/EXPERIMENT_TRACKER.md` + 各 run 的 `eval_metrics.json`

#### 阶段 4: 结论+迭代

1. **输入**: `EXPERIMENT_TRACKER.md` + `eval_metrics.json`
2. **调用 `/result-to-claim`**: 判断结果支持哪些 claim, 标记 `yes` / `partial` / `no`
3. **调用 `/research-review`**: 外部评审员评估整体工作质量 (1-10 分)
4. **生成下轮方向**:
   - 若 claim 支持度 >= partial: 调用 `/ablation-planner` 设计消融, 或扩展数据集
   - 若 claim 支持度 = no: 调用 `/idea-creator` 基于失败教训生成新方向
5. **输出**: `pipeline_state/stage_4_output/CLAIMS_FROM_RESULTS.md` + 下轮 `IDEA_CANDIDATES.md`
6. **循环**: 回到阶段 1

---

## 5. 关键设计决策

### 5.1 为什么改写 `/experiment-bridge` 而不是直接用?

ARIS 的 `/experiment-bridge` 面向通用 ML 项目, 默认从零写训练脚本。RLIimage 已有:
- 稳定的 `runner_utils` (build_penn_fudan_loaders_320, decode_boxes, evaluate_model, unfreeze_rlvr, grpo_advantage, gaussian_log_prob)
- 标准化的 `run_one(name, mode, seed)` 模板
- 统一的 `eval_metrics.json` 输出格式

改写后的 bridge 只需: 读取 idea → 填充模板 → 生成 runner → 复用 utils。避免重复造轮子。

### 5.2 为什么改写 `/experiment-queue` 为本地版?

RLIimage 的硬件约束:
- 单卡 8G GPU (本地, 非远程)
- batch_size=1~2, 无法并行多个 job
- 不需要 SSH / Vast.ai / Modal

本地版 queue 保留核心功能:
- 多 seed 顺序执行 (seed 42 → 123 → 456)
- OOM 自动重试 (batch_size 减半)
- 状态持久化 (resume after crash)
- Wave 依赖 (如: baseline 必须先跑完)

### 5.3 子 Agent 通信协议设计

```python
# agent_bridge.py 核心接口

class AgentBridge:
    def spawn_critic(self, idea: str, context: dict) -> str:
        """Spawn Codex(opus) as critic. Return thread_id."""
        pass

    def spawn_explorer(self, idea: str, context: dict) -> str:
        """Spawn k2.7(sonnet) as explorer. Return thread_id."""
        pass

    def cross_examine(self, thread_a: str, thread_b: str, questions: list[str]) -> dict:
        """Collect responses from both agents, return structured comparison."""
        pass

    def synthesize(self, responses: dict) -> FeasibilityReport:
        """Synthesize into PASS/NEEDS_RISK/FAIL."""
        pass
```

**通信方式**: 通过文件系统 + 状态轮询 (非网络 socket):
1. 主 Agent 写 `agent_input/{thread_id}.json`
2. 子 Agent 读输入, 写 `agent_output/{thread_id}.json`
3. 主 Agent 轮询输出文件, 超时 5 分钟

**为什么不用 MCP / 网络**: 简化部署, 不依赖外部服务, 适合本地 8G GPU 环境。

---

## 6. 状态恢复机制

流水线支持任意阶段中断后恢复:

```python
# pipeline_start.py --resume

STATE_FILE = "pipeline_state/current_idea.json"

class PipelineState:
    def save(self):
        """Serialize current stage, inputs, outputs to JSON."""
        pass

    def load(self, path: str) -> "PipelineState":
        """Deserialize and resume from last completed stage."""
        pass

    def resume(self):
        """Skip completed stages, continue from next."""
        pass
```

**恢复规则**:
- 阶段 1 完成 (IDEA_CANDIDATES.md 存在) → 从阶段 2 开始
- 阶段 2 完成 (FEASIBILITY_REPORT.md 存在) → 从阶段 3 开始
- 阶段 3 部分完成 (queue_state.json 存在, 有 pending jobs) → 恢复队列, 继续跑
- 阶段 4 完成 (CLAIMS_FROM_RESULTS.md 存在) → 生成下轮方向, 回到阶段 1

---

## 7. 与现有代码的集成点

### 7.1 复用现有 runner 模板

现有 runner (如 `round286_runner.py`) 的公共结构:

```python
def run_one(cfg_name, mode, seed):
    # 1. set_seed(seed)
    # 2. build model + load checkpoint
    # 3. unfreeze_rlvr(model)
    # 4. register hooks (sampled_props, box_head_in, fpn_feats)
    # 5. build optimizer (body_lr + head_lr)
    # 6. training loop (det_loss + rl_loss + kl_loss)
    # 7. evaluate + save_json(eval_metrics.json)
    # 8. return metrics dict
```

`runner_generator.py` 将提取此模板, 允许通过 YAML/JSON 配置注入:
- 新的 reward 函数
- 新的 gate 逻辑
- 新的 loss 组合
- 新的 dataset (PennFudan / NWPU / VOC)

### 7.2 复用现有评估体系

```python
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
# 已支持: AP50, AP75, precision, recall, ECE
# 输出格式: dict -> save_json -> eval_metrics.json
```

`analyze-results` 直接读取这些 JSON, 无需改动。

### 7.3 复用现有通知机制

```python
from scripts.notify_feishu import notify
# 实验完成 / 失败 / 审计通过 时自动通知
```

---

## 8. 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| 子 Agent (Codex/k2.7) 不可用 | 降级为本地 Claude Code 单 Agent 模式, 跳过交叉质疑 |
| 8G GPU OOM 频繁 | local_queue 自动 halve batch_size, 最多重试 3 次 |
| Runner 生成错误 | 生成后强制跑 sanity (1 epoch, 1 seed), 失败则阻断 |
| 审计发现 fake GT | 阻断发布, 要求人工检查, 记录到 `AUDIT_FAILURES.md` |
| 流水线状态损坏 | 每个阶段写 timestamped 备份, 支持从任意备份恢复 |

---

## 9. 下一步行动

1. **实现 `scripts/pipeline_start.py`**: 命令行入口 + 阶段路由
2. **实现 `scripts/pipeline_utils/agent_bridge.py`**: 子 Agent 通信协议
3. **实现 `scripts/pipeline_utils/runner_generator.py`**: 从 idea 生成 runner
4. **实现 `scripts/pipeline_utils/local_queue.py`**: 本地实验队列
5. **实现 `scripts/pipeline_utils/evidence_aggregator.py`**: 证据聚合
6. **集成 ARIS skills**: 安装 ARIS, 测试 `/research-lit` + `/idea-creator` + `/novelty-check`
7. **端到端测试**: 用一个简单 idea (如 "测试新的 KL 权重") 跑通全链

---

## 10. 附录: ARIS Skill 调用速查

```bash
# 阶段 1
/research-lit "频域特征在目标检测中的应用"
/idea-creator "基于 RLVR 的检测器后训练, 引入频域 verifier"
/novelty-check "用 FFT 相位一致性作为检测 RLVR reward"

# 阶段 3
/experiment-bridge "pipeline_state/stage_2_output/EXPERIMENT_PLAN.md"
/experiment-queue "pipeline_state/stage_3_output/manifest.json"
/analyze-results "pipeline_state/stage_3_output/"
/experiment-audit "pipeline_state/stage_3_output/"
/auto-review-loop "RLVR detection post-training"

# 阶段 4
/result-to-claim "pipeline_state/stage_3_output/EXPERIMENT_TRACKER.md"
/research-review "pipeline_state/stage_4_output/CLAIMS_FROM_RESULTS.md"
/ablation-planner "pipeline_state/stage_3_output/EXPERIMENT_TRACKER.md"
```
