"""RLIimage 自动化研究流水线启动脚本.

接收用户 idea, 顺序触发方向发现 -> 可行性验证 -> 实验执行 -> 结论迭代.
支持命令行、文件输入、状态恢复三种触发方式.

用法:
    python scripts/pipeline_start.py "idea: 用频域相位一致性作为 RLVR reward"
    python scripts/pipeline_start.py --from-file idea.txt
    python scripts/pipeline_start.py --resume pipeline_state/current_idea.json

环境要求:
    - Python 3.10+
    - ARIS skills 已安装 (可选, 用于 /research-lit 等)
    - 本地 GPU (8G) 可用
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("E:/CLIproject/RLimage")
PIPELINE_STATE_DIR = PROJECT_ROOT / "pipeline_state"
STAGE_DIRS = {
    1: PIPELINE_STATE_DIR / "stage_1_output",
    2: PIPELINE_STATE_DIR / "stage_2_output",
    3: PIPELINE_STATE_DIR / "stage_3_output",
    4: PIPELINE_STATE_DIR / "stage_4_output",
}
CURRENT_IDEA_FILE = PIPELINE_STATE_DIR / "current_idea.json"

# 子进程超时 (秒)
STAGE_TIMEOUTS = {
    1: 600,   # 方向发现: 文献搜索 + 脑暴
    2: 900,   # 可行性验证: 双 Agent 分析
    3: 3600,  # 实验执行: 单个 runner 可能跑 30-60 分钟
    4: 600,   # 结论迭代: 评估 + 生成新方向
}

# 最大重试次数
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class Idea:
    """用户输入的研究 idea."""
    text: str
    source: str = "cli"          # cli | file
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: f"idea_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "source": self.source, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Idea":
        return cls(text=d["text"], source=d.get("source", "cli"), timestamp=d["timestamp"], id=d["id"])


@dataclass
class PipelineState:
    """流水线全局状态, 支持序列化/恢复."""
    idea: Idea
    current_stage: int = 1
    completed_stages: list[int] = field(default_factory=list)
    stage_outputs: dict[int, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea.to_dict(),
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "stage_outputs": self.stage_outputs,
            "errors": self.errors,
            "last_updated": self.last_updated,
        }

    def save(self, path: Path | None = None) -> None:
        path = path or CURRENT_IDEA_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            idea=Idea.from_dict(d["idea"]),
            current_stage=d.get("current_stage", 1),
            completed_stages=d.get("completed_stages", []),
            stage_outputs=d.get("stage_outputs", {}),
            errors=d.get("errors", []),
            last_updated=d.get("last_updated", ""),
        )

    def mark_stage_complete(self, stage: int, output: dict[str, Any]) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.stage_outputs[stage] = output
        self.current_stage = stage + 1
        self.last_updated = datetime.now().isoformat()
        self.save()

    def add_error(self, msg: str) -> None:
        self.errors.append(f"[{datetime.now().isoformat()}] {msg}")
        self.last_updated = datetime.now().isoformat()
        self.save()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """打印带时间戳的日志."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def _run_skill(skill_name: str, args: str = "", timeout: int = 600) -> tuple[bool, str]:
    """调用 ARIS skill (通过 Claude Code 命令或本地 fallback).

    优先尝试 Claude Code 的 skill 调用; 若不可用, 使用本地 fallback.
    """
    # 尝试通过 claude 命令调用 skill (需要 Claude Code CLI)
    cmd = ["claude", "--skill", skill_name, args] if args else ["claude", "--skill", skill_name]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT))
        if r.returncode == 0:
            return True, r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: 本地 Python 实现 (简化版, 不依赖 ARIS)
    _log(f"ARIS skill '{skill_name}' 不可用, 使用本地 fallback")
    return _local_skill_fallback(skill_name, args)


def _local_skill_fallback(skill_name: str, args: str) -> tuple[bool, str]:
    """本地 skill fallback, 生成占位输出以便流水线继续."""
    # 这里可以接入本地实现的简化版技能
    # 例如: 本地文献搜索可用 arxiv API, 本地脑暴可用模板填充
    if skill_name == "research-lit":
        return True, _fallback_research_lit(args)
    if skill_name == "idea-creator":
        return True, _fallback_idea_creator(args)
    if skill_name == "novelty-check":
        return True, _fallback_novelty_check(args)
    if skill_name == "experiment-plan":
        return True, _fallback_experiment_plan(args)
    if skill_name == "analyze-results":
        return True, _fallback_analyze_results(args)
    if skill_name == "result-to-claim":
        return True, _fallback_result_to_claim(args)
    return False, f"Skill '{skill_name}' 无本地 fallback"


def _fallback_research_lit(query: str) -> str:
    """简化文献搜索: 尝试 arxiv API."""
    try:
        import urllib.request
        url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&max_results=5"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read().decode("utf-8")
        # 简单提取标题
        import re
        titles = re.findall(r"<title>([^<]+)</title>", data)
        return f"ArXiv 搜索结果 (前5): {titles[1:] if len(titles) > 1 else titles}"
    except Exception as e:
        return f"文献搜索失败: {e}"


def _fallback_idea_creator(context: str) -> str:
    """简化脑暴: 基于模板生成方向."""
    lines = [
        "## 脑暴方向 (fallback)",
        "",
        f"基于输入: {context}",
        "",
        "1. **方向 A**: 在现有 AFM 基础上引入可学习的相位门控, 替代硬编码 gate_strength",
        "2. **方向 B**: 将频域 verifier 从 ROI 特征移到 FPN 特征, 减少信息损失",
        "3. **方向 C**: 使用 DPO (Direct Preference Optimization) 替代 RLVR, 简化训练流程",
        "4. **方向 D**: 在大数据集 (NWPU/VOC) 上验证 mid06 的 AP75 增益是否泛化",
        "5. **方向 E**: 将 AFM 机制扩展到两阶段检测器的 RPN 阶段, 提升 proposal 质量",
        "",
        "(注: 此为本地 fallback 输出, 建议安装 ARIS 获取更丰富的脑暴结果)",
    ]
    return "\n".join(lines)


def _fallback_novelty_check(idea: str) -> str:
    return f"查新结果 (fallback): 方向 '{idea[:50]}...' 需要进一步搜索确认 novelty。建议搜索关键词: RLVR, detection, post-training, frequency domain。"


def _fallback_experiment_plan(proposal: str) -> str:
    lines = [
        "## 实验计划 (fallback)",
        "",
        f"基于提案: {proposal}",
        "",
        "### 里程碑",
        "1. **Sanity**: 1 seed, 1 epoch, 验证 runner 可运行",
        "2. **Baseline**: 3 seeds, 8 epochs, det_only_unf 模式",
        "3. **Main Method**: 3 seeds, 8 epochs, 新方法模式",
        "4. **Ablation**: 消融关键组件 (如去掉 phase, 改 gate_strength)",
        "",
        "### 配置",
        "- Dataset: Penn-Fudan (default) 或 NWPU (若指定)",
        "- Model: Faster R-CNN MobileNetV3-Large-FPN",
        "- Seeds: [42, 123, 456]",
        "- Metrics: AP50, AP75, precision, recall, ECE",
        "",
        "(注: 此为本地 fallback, 建议安装 ARIS 获取详细计划)",
    ]
    return "\n".join(lines)


def _fallback_analyze_results(results_dir: str) -> str:
    """简化结果分析: 读取 eval_metrics.json 文件."""
    p = Path(results_dir)
    if not p.exists():
        return f"结果目录不存在: {results_dir}"
    jsons = list(p.rglob("eval_metrics.json"))
    if not jsons:
        return "未找到 eval_metrics.json"
    summaries = []
    for j in jsons:
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
            rn = data.get("run_name", j.parent.name)
            ap50 = data.get("ap50", 0)
            ap75 = data.get("ap75", 0)
            summaries.append(f"  {rn}: AP50={ap50:.4f} AP75={ap75:.4f}")
        except Exception:
            continue
    return "## 结果汇总 (fallback)\n\n" + "\n".join(summaries)


def _fallback_result_to_claim(tracker_path: str) -> str:
    return "Claim 评估 (fallback): 请人工检查 EXPERIMENT_TRACKER.md 中的结果是否支持预期 claim。"


# ---------------------------------------------------------------------------
# 阶段实现
# ---------------------------------------------------------------------------

def stage_1_direction_discovery(state: PipelineState) -> dict[str, Any]:
    """阶段 1: 方向发现 — 文献调研 -> 脑暴 -> 查新 -> 排名.

    输出: IDEA_CANDIDATES.md
    """
    _log("=" * 60)
    _log("阶段 1: 方向发现")
    _log("=" * 60)

    idea = state.idea.text
    output_dir = STAGE_DIRS[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1.1 文献调研
    _log("[1.1] 文献调研...")
    ok, lit_result = _run_skill("research-lit", idea, timeout=STAGE_TIMEOUTS[1] // 3)
    if not ok:
        _log(f"文献调研失败: {lit_result}")
        state.add_error(f"research-lit failed: {lit_result}")
    else:
        _log("文献调研完成")

    # 1.2 脑暴方向
    _log("[1.2] 脑暴方向...")
    ok, ideas_result = _run_skill("idea-creator", idea, timeout=STAGE_TIMEOUTS[1] // 3)
    if not ok:
        _log(f"脑暴失败: {ideas_result}")
        state.add_error(f"idea-creator failed: {ideas_result}")
    else:
        _log("脑暴完成")

    # 1.3 查新 (对 top 3 方向)
    _log("[1.3] 查新...")
    ok, novelty_result = _run_skill("novelty-check", idea, timeout=STAGE_TIMEOUTS[1] // 3)
    if not ok:
        _log(f"查新失败: {novelty_result}")
        state.add_error(f"novelty-check failed: {novelty_result}")
    else:
        _log("查新完成")

    # 合并输出
    candidates_path = output_dir / "IDEA_CANDIDATES.md"
    content = f"""# Idea Candidates — {state.idea.id}

**输入**: {idea}
**时间**: {datetime.now().isoformat()}

## 文献调研
{lit_result}

## 脑暴方向
{ideas_result}

## 查新结果
{novelty_result}

## 排名 (待人工确认)
1. [方向 A] — 待填充
2. [方向 B] — 待填充
3. [方向 C] — 待填充

---
(注: 若 ARIS skills 可用, 输出会更丰富)
"""
    candidates_path.write_text(content, encoding="utf-8")
    _log(f"IDEA_CANDIDATES.md 已保存: {candidates_path}")

    return {"idea_candidates_path": str(candidates_path), "skills_used": ["research-lit", "idea-creator", "novelty-check"]}


def stage_2_feasibility_verification(state: PipelineState) -> dict[str, Any]:
    """阶段 2: 可行性验证 — 数学分析 + 论文调研 + 交叉质疑.

    输出: FEASIBILITY_REPORT.md
    """
    _log("=" * 60)
    _log("阶段 2: 可行性验证")
    _log("=" * 60)

    idea = state.idea.text
    output_dir = STAGE_DIRS[2]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 数学分析 (Codex/opus critic) — 通过本地文件模拟
    _log("[2.1] 数学分析 (Critic Agent)...")
    critic_input = output_dir / "critic_input.json"
    critic_output = output_dir / "critic_output.json"
    critic_input.write_text(json.dumps({"idea": idea, "task": "math_analysis"}, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Critic 输入已写: {critic_input}")
    _log("(提示: 在另一个 Claude Code 会话中运行 scripts/pipeline_utils/agent_bridge.py --role critic)")

    # 2.2 论文调研 (k2.7/sonnet explorer) — 通过本地文件模拟
    _log("[2.2] 论文调研 (Explorer Agent)...")
    explorer_input = output_dir / "explorer_input.json"
    explorer_output = output_dir / "explorer_output.json"
    explorer_input.write_text(json.dumps({"idea": idea, "task": "paper_survey"}, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Explorer 输入已写: {explorer_input}")
    _log("(提示: 在另一个 Claude Code 会话中运行 scripts/pipeline_utils/agent_bridge.py --role explorer)")

    # 2.3 交叉质疑 (主 Agent 读取两个输出后综合)
    _log("[2.3] 交叉质疑 (等待子 Agent 输出)...")
    _log("等待子 Agent 完成... (按 Ctrl+C 跳过, 使用本地 fallback)")

    # 简化: 若子 Agent 输出文件存在则读取, 否则用 fallback
    critic_result = ""
    explorer_result = ""
    if critic_output.exists():
        critic_result = critic_output.read_text(encoding="utf-8")
        _log("Critic 输出已读取")
    if explorer_output.exists():
        explorer_result = explorer_output.read_text(encoding="utf-8")
        _log("Explorer 输出已读取")

    if not critic_result or not explorer_result:
        _log("子 Agent 输出未找到, 使用本地 fallback")
        critic_result = f"数学分析 (fallback): Idea '{idea[:50]}...' 的奖励函数可微性需要验证。建议检查: 1) 梯度链是否完整 2) 方差是否可控 3) 是否存在拓扑死锁。"
        explorer_result = f"论文调研 (fallback): 找到 3 篇相关论文。1) Visual-RFT (RLVR for VLM) 2) MPLSeg (in-network FFT) 3) GRPO (group relative policy optimization)。关键教训: 外部 verifier 信号可能太弱, in-network 梯度更有效。"

    # 综合判断
    verdict = "PASS"  # 简化: 默认 PASS, 实际应由主 Agent 综合判断
    if "不可微" in critic_result or "方差太高" in critic_result:
        verdict = "NEEDS_RISK"
    if "拓扑死锁" in critic_result:
        verdict = "FAIL"

    report_path = output_dir / "FEASIBILITY_REPORT.md"
    content = f"""# Feasibility Report — {state.idea.id}

**输入**: {idea}
**时间**: {datetime.now().isoformat()}
**综合判断**: {verdict}

## 数学分析 (Critic)
{critic_result}

## 论文调研 (Explorer)
{explorer_result}

## 交叉质疑
- Q1: 梯度链是否完整? → 待回答
- Q2: 方差是否可控? → 待回答
- Q3: 与现有方法 (mid06) 的关系? → 待回答

## 建议
- 若 PASS: 进入阶段 3 实验执行
- 若 NEEDS_RISK: 在实验计划中增加 sanity check 和 early stopping
- 若 FAIL: 回到阶段 1 重新脑暴
"""
    report_path.write_text(content, encoding="utf-8")
    _log(f"FEASIBILITY_REPORT.md 已保存: {report_path}")

    return {"feasibility_report_path": str(report_path), "verdict": verdict}


def stage_3_experiment_execution(state: PipelineState) -> dict[str, Any]:
    """阶段 3: 实验执行 — 写 runner -> 检查 -> 排队跑 -> 聚合报告 -> bug 审计.

    输出: EXPERIMENT_TRACKER.md + 各 run 的 eval_metrics.json
    """
    _log("=" * 60)
    _log("阶段 3: 实验执行")
    _log("=" * 60)

    idea = state.idea.text
    output_dir = STAGE_DIRS[3]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3.1 生成实验计划
    _log("[3.1] 生成实验计划...")
    ok, plan_result = _run_skill("experiment-plan", idea, timeout=STAGE_TIMEOUTS[3] // 5)
    plan_path = output_dir / "EXPERIMENT_PLAN.md"
    plan_path.write_text(plan_result if ok else _fallback_experiment_plan(idea), encoding="utf-8")
    _log(f"EXPERIMENT_PLAN.md 已保存: {plan_path}")

    # 3.2 生成 runner
    _log("[3.2] 生成 runner...")
    runner_gen = PROJECT_ROOT / "scripts" / "pipeline_utils" / "runner_generator.py"
    if runner_gen.exists():
        # 调用 runner 生成器
        cmd = [sys.executable, str(runner_gen), "--plan", str(plan_path), "--output-dir", str(output_dir)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                _log("Runner 生成成功")
            else:
                _log(f"Runner 生成失败: {r.stderr}")
                state.add_error(f"runner_generator failed: {r.stderr}")
        except Exception as e:
            _log(f"Runner 生成异常: {e}")
            state.add_error(f"runner_generator exception: {e}")
    else:
        _log(f"Runner 生成器未找到: {runner_gen}")
        _log("请手动基于模板创建 runner, 或实现 runner_generator.py")

    # 3.3 本地队列执行
    _log("[3.3] 排队执行实验...")
    queue_script = PROJECT_ROOT / "scripts" / "pipeline_utils" / "local_queue.py"
    if queue_script.exists():
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            cmd = [sys.executable, str(queue_script), "--manifest", str(manifest_path)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=STAGE_TIMEOUTS[3])
                if r.returncode == 0:
                    _log("实验队列执行完成")
                else:
                    _log(f"队列执行失败: {r.stderr}")
                    state.add_error(f"local_queue failed: {r.stderr}")
            except subprocess.TimeoutExpired:
                _log("队列执行超时, 但可能部分完成")
                state.add_error("local_queue timeout")
        else:
            _log("manifest.json 未找到, 跳过队列执行")
    else:
        _log(f"本地队列未找到: {queue_script}")
        _log("请手动运行 runner, 或实现 local_queue.py")

    # 3.4 聚合报告
    _log("[3.4] 聚合报告...")
    ok, analysis_result = _run_skill("analyze-results", str(output_dir), timeout=STAGE_TIMEOUTS[3] // 5)
    tracker_path = output_dir / "EXPERIMENT_TRACKER.md"
    tracker_path.write_text(analysis_result if ok else _fallback_analyze_results(str(output_dir)), encoding="utf-8")
    _log(f"EXPERIMENT_TRACKER.md 已保存: {tracker_path}")

    # 3.5 Bug 审计
    _log("[3.5] Bug 审计...")
    audit_path = output_dir / "EXPERIMENT_AUDIT.md"
    audit_content = f"""# Experiment Audit — {state.idea.id}

**时间**: {datetime.now().isoformat()}

## 检查项
- [ ] GT 来源: 是否来自 dataset 而非 model output?
- [ ] AP75/AP50: 是否使用官方 COCO API 计算?
- [ ] 能量门控: 是否 per-group 而非 per-sample?
- [ ] 基线: checkpoint 是否 frozen?
- [ ] 死代码: 是否有未调用的 metric 函数?

## 结果
待人工检查或调用 /experiment-audit skill

## 建议
若审计通过, 进入阶段 4; 若有 FAIL, 修复后重跑
"""
    audit_path.write_text(audit_content, encoding="utf-8")
    _log(f"EXPERIMENT_AUDIT.md 已保存: {audit_path}")

    return {
        "experiment_plan_path": str(plan_path),
        "experiment_tracker_path": str(tracker_path),
        "experiment_audit_path": str(audit_path),
    }


def stage_4_conclusion_iteration(state: PipelineState) -> dict[str, Any]:
    """阶段 4: 结论+迭代 — 结构化证据 -> 跨模型评估 -> 生成下轮方向.

    输出: CLAIMS_FROM_RESULTS.md + 下轮 IDEA_CANDIDATES.md
    """
    _log("=" * 60)
    _log("阶段 4: 结论+迭代")
    _log("=" * 60)

    idea = state.idea.text
    output_dir = STAGE_DIRS[4]
    output_dir.mkdir(parents=True, exist_ok=True)
    stage3_dir = STAGE_DIRS[3]

    # 4.1 结构化证据
    _log("[4.1] 结构化证据...")
    tracker_path = stage3_dir / "EXPERIMENT_TRACKER.md"
    ok, claim_result = _run_skill("result-to-claim", str(tracker_path), timeout=STAGE_TIMEOUTS[4] // 3)
    claims_path = output_dir / "CLAIMS_FROM_RESULTS.md"
    claims_path.write_text(claim_result if ok else _fallback_result_to_claim(str(tracker_path)), encoding="utf-8")
    _log(f"CLAIMS_FROM_RESULTS.md 已保存: {claims_path}")

    # 4.2 跨模型评估
    _log("[4.2] 跨模型评估...")
    ok, review_result = _run_skill("research-review", str(claims_path), timeout=STAGE_TIMEOUTS[4] // 3)
    review_path = output_dir / "RESEARCH_REVIEW.md"
    review_path.write_text(review_result if ok else "评审结果 (fallback): 待人工评审", encoding="utf-8")
    _log(f"RESEARCH_REVIEW.md 已保存: {review_path}")

    # 4.3 生成下轮方向
    _log("[4.3] 生成下轮方向...")
    # 读取 claim 支持度
    claims_text = claims_path.read_text(encoding="utf-8")
    if "claim_supported: yes" in claims_text or "claim_supported: partial" in claims_text:
        _log("Claim 部分/完全支持, 建议设计消融或扩展数据集")
        next_action = "ablation_or_scale"
    else:
        _log("Claim 不支持, 建议重新脑暴方向")
        next_action = "re_ideate"

    # 生成下轮 IDEA_CANDIDATES.md (写入 stage_1_output 以便循环)
    next_ideas_path = STAGE_DIRS[1] / "NEXT_IDEA_CANDIDATES.md"
    next_ideas_content = f"""# Next Idea Candidates — {state.idea.id}

**上轮结论**: {next_action}
**时间**: {datetime.now().isoformat()}

## 建议方向
1. [基于上轮结果的新方向 A]
2. [基于上轮结果的新方向 B]
3. [基于上轮结果的新方向 C]

## 行动
- 若 {next_action} == ablation_or_scale: 调用 /ablation-planner
- 若 {next_action} == re_ideate: 调用 /idea-creator

---
(注: 请根据 CLAIMS_FROM_RESULTS.md 和 RESEARCH_REVIEW.md 人工细化)
"""
    next_ideas_path.write_text(next_ideas_content, encoding="utf-8")
    _log(f"NEXT_IDEA_CANDIDATES.md 已保存: {next_ideas_path}")

    return {
        "claims_path": str(claims_path),
        "review_path": str(review_path),
        "next_ideas_path": str(next_ideas_path),
        "next_action": next_action,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_pipeline(state: PipelineState) -> None:
    """执行流水线, 支持从任意阶段恢复."""
    _log(f"开始流水线: idea={state.idea.text[:60]}...")
    _log(f"当前阶段: {state.current_stage}, 已完成: {state.completed_stages}")

    stages = {
        1: stage_1_direction_discovery,
        2: stage_2_feasibility_verification,
        3: stage_3_experiment_execution,
        4: stage_4_conclusion_iteration,
    }

    for stage_num in range(state.current_stage, 5):
        _log(f"\n进入阶段 {stage_num}...")
        stage_fn = stages[stage_num]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                output = stage_fn(state)
                state.mark_stage_complete(stage_num, output)
                _log(f"阶段 {stage_num} 完成 (attempt {attempt})")
                break
            except Exception as e:
                state.add_error(f"Stage {stage_num} attempt {attempt} failed: {e}")
                _log(f"阶段 {stage_num} 失败 (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    _log(f"等待 10 秒后重试...")
                    time.sleep(10)
                else:
                    _log(f"阶段 {stage_num} 最终失败, 跳过")
                    # 标记为完成但记录错误, 继续下一阶段
                    state.mark_stage_complete(stage_num, {"error": str(e), "skipped": True})

    _log("\n" + "=" * 60)
    _log("流水线完成!")
    _log(f"状态文件: {CURRENT_IDEA_FILE}")
    _log(f"输出目录: {PIPELINE_STATE_DIR}")
    _log("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RLIimage 自动化研究流水线启动脚本")
    parser.add_argument("idea", nargs="?", help="研究 idea (一句话描述)")
    parser.add_argument("--from-file", type=str, help="从文件读取 idea")
    parser.add_argument("--resume", type=str, help="恢复已保存的流水线状态")
    args = parser.parse_args()

    # 初始化状态
    if args.resume:
        state_path = Path(args.resume)
        if not state_path.exists():
            print(f"错误: 状态文件不存在: {state_path}")
            sys.exit(1)
        state = PipelineState.load(state_path)
        _log(f"恢复流水线: {state.idea.id}")
    else:
        if args.from_file:
            file_path = Path(args.from_file)
            if not file_path.exists():
                print(f"错误: idea 文件不存在: {file_path}")
                sys.exit(1)
            idea_text = file_path.read_text(encoding="utf-8").strip()
            source = f"file:{file_path}"
        elif args.idea:
            idea_text = args.idea
            source = "cli"
        else:
            print("错误: 请提供 idea (命令行参数、--from-file 或 --resume)")
            parser.print_help()
            sys.exit(1)

        idea = Idea(text=idea_text, source=source)
        state = PipelineState(idea=idea)
        state.save()
        _log(f"新流水线: {idea.id}")

    # 执行
    try:
        run_pipeline(state)
    except KeyboardInterrupt:
        _log("\n用户中断, 状态已保存, 可稍后恢复")
        state.save()
        sys.exit(0)


if __name__ == "__main__":
    main()
