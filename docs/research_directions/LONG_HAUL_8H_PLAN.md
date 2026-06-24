# 8 小时长程自主实验计划

> 基于 `Auto-claude-code-research-in-sleep` 的实验智能体技能（experiment-plan / run-experiment / experiment-queue / monitor-experiment / analyze-results / auto-review-loop）设计。

## 核心原则

**每个新领域的任务开始前，先在网络上查找相关资料和开源实现，尽可能复用已有方法，减少“自己捏出来”的组件。**

具体执行方式：
1. 用 WebSearch / GitHub search 检索该领域的 survey、代表性论文、官方实现、主流工具箱（如 ART、Foolbox、Torchattacks 等）。
2. 优先复用或参考已有实现（算法、损失函数、评估协议）。
3. 在 plan 中记录调研结论、选择的 baseline、参考链接，再进入实现/实验阶段。
4. 只有确认现有方法不适用或无法直接集成时，才允许自定义新组件。

---

## 任务目标

在当前 6 个 Plan（A-F）已实现并通过单元测试的基础上，用 8 小时 wall-clock 完成：
1. **修复并验证 Plan B 对抗防御**（核心瓶颈）；
2. **若 Plan B 验证成功，启动 Plan C 分割训练**；
3. **自动分析结果并决定下一步**。

整个流程要求：
- 尽可能**无人值守**推进；
- 每个阶段有明确的**通过/失败判断标准**；
- 失败时自动降级到下一方案，不卡住；
- 最终产出：对比表格、可视化图、决策建议。

---

## 当前状态

- Plan A-F 模块 + 单元测试：✅ 97 passed
- Plan B smoke test：⚠️ 攻击无效（AP50 0.913 → 0.902），防御过度损伤干净样本（clean_drop = 0.10）
- Plan C/D/E/F：尚未端到端验证

---

## 8 小时阶段规划

### Phase 0：调研与 Baseline 选择（0.5h）

**目标**：确认 Plan B 应复用的主流攻击/防御方法，避免从零造轮子。

**调研结论（已提前完成）**：
- 检测器 patch 攻击主流方法：**DPatch**、**RP2+**、**EOT**、**TOG**、**LaVAN**。
- 主流损失：最小化 objectness + classification score + TV loss + NPS。
- 主流 patch 放置：覆盖 GT bbox 中心，大小按 bbox 短边比例缩放。
- 主流实现库：**ART (IBM)**、**Foolbox**、**Torchattacks**、**CleverHans**。
- 防御方向：对抗训练、检测+移除（SAR）、输入净化。

**选择**：
- 攻击：采用 **DPatch/RP2-style disappearance attack**（objectness + classification suppression + TV loss），在我们的代码中扩展 `AdversarialPatchAttack`。
- 防御：先验证我们自有的 **SpectralChordDefense**（频谱净化）作为输入净化方案；若效果不佳，再调研集成 SAR/检测移除。

---

### Phase 1：修复 Plan B 攻击（1.5h）

**目标**：让攻击在 Penn-Fudan 上产生可量化的下降（AP50_drop ≥ 0.15）。

**具体动作**：
1. 复用并增强现有 `AdversarialPatchAttack`，参考 DPatch/RP2 的放置策略：
   - 补丁大小从 48×48 提升到 **80×80**；
   - 补丁位置改为 **覆盖 GT 行人框中心**（按 bbox 中心放置）；
   - 优化步数从 100 提升到 **300**；
   - 学习率从 0.1 提升到 **0.5**（带 momentum 0.9）；
   - random init + smooth_sigma 保持补丁相对自然；
   - 损失函数继续使用“抑制目标类别最高置信度 + 阈值惩罚”。

2. 在 10 张图上跑攻击-only 实验，验证 AP50 是否明显下降。

**通过标准**：
- AP50_drop ≥ 0.15：进入 Phase 2；
- AP50_drop < 0.15：尝试 multiple patches、full-image PGD（标准基线）或调整 patch 位置到 torso/head；最多 3 次。

---

### Phase 2：修复 Plan B 防御（1.5h）

**目标**：在成功攻击的基础上，让 SpectralChordDefense 恢复性能，同时不过度损伤干净样本。

**具体动作**：
1. 修改防御配置（已验证 160 过大/慢，128 可运行）：
   - `defense_size` 设为 **128**；
   - `anomaly_threshold` 从 5.0 降到 **2.5**；
   - `lambda_step` 从 1.0 降到 **0.3**；
   - 保留 DC，通过 `preserve_dc=True` 保护低频；
   - 在 50 张干净图上拟合 `NaturalSpectrumModel`。

2. 在 10 张图上跑完整攻击+防御实验。

**通过标准**：
- `recovery_rate ≥ 0.5` 且 `clean_drop ≤ 0.05`：进入 Phase 3；
- `clean_drop > 0.05`：降低 lambda_step / 提高 anomaly_threshold；
- `recovery_rate < 0.5`：增强 anomaly detection 或训练更好的流形。

---

### Phase 3：扩展 Plan B 完整实验（2h）

**目标**：在完整测试集上验证 Plan B 的有效性，产出可信指标。

**具体动作**：
1. 在 **全部 ~150 张 Penn-Fudan 测试图像**上跑：
   - 干净图像 AP50；
   - 对抗图像 AP50；
   - 防御后 AP50；
   - 干净图像经防御后 AP50。
2. 做关键消融（只跑可行的配置，避免 OOM/挂起）：
   - patch_size ∈ {48, 80, 112}；
   - defense_size ∈ {96, 128}；
   - anomaly_threshold ∈ {2.0, 2.5, 3.0}。
3. 保存所有结果到 `runs/experiments/plan_b_defense/full/`。

**通过标准**：
- 完整实验 recovery_rate ≥ 0.5 且 clean_drop ≤ 0.05：标记 Plan B 成功，产出主表格；
- 否则：记录失败模式，进入 Phase 4 时只把 Plan B 作为 negative result。

---

### Phase 4：Plan C 语义分割 smoke test（1.5h）

**目标**：验证 ManifoldAFMBlock 在分割任务上能否训练且不崩溃。

**具体动作**：
1. 创建 `scripts/experiments/plan_c_smoke_test.py`：
   - 加载 FCN-ResNet50；
   - 在 layer4 后插入 `ManifoldAFMBlock`；
   - 在 Penn-Fudan 分割上训练 3 epoch；
   - 每 epoch 评估 mIoU。
2. 同时跑 baseline（无 AFM）作为对比。

**通过标准**：
- 训练不崩溃；
- ManifoldAFM mIoU ≥ baseline - 0.02：进入 Phase 5；
- 崩溃或 mIoU 严重下降：诊断后修复模块或降级到 Plan D。

---

### Phase 5：Plan D 分类 smoke test（1h）

**目标**：验证 SpectralClassifierHead 在 CIFAR-100 上可训练。

**具体动作**：
1. 创建 `scripts/experiments/plan_d_smoke_test.py`：
   - ResNet-18 + SpectralClassifierHead；
   - CIFAR-100 训练 10 epoch；
   - 对比标准线性头。

**通过标准**：
- 训练完成；
- SpectralHead Top-1 ≥ baseline - 0.01：进入 Phase 6；
- 否则：记录问题，进入 Phase 6 时聚焦 Plan B/C 结果。

---

### Phase 6：自动分析与决策（0.5h）

**目标**：汇总所有实验结果，自动生成对比表格、可视化图和下一步建议。

**具体动作**：
1. 读取所有 `metrics.json` 和训练日志；
2. 生成：
   - `runs/experiments/SUMMARY_8H.md`：对比表格 + 关键发现；
   - `runs/experiments/figures/`：趋势图、对比图；
3. 根据结果决定下一步（优先级排序）：
   - 若 Plan B 成功 → 扩展 Plan B 到遥感/多模态；
   - 若 Plan C 成功 → 跑完整分割实验；
   - 若都失败 → 回到模块设计层面修复。

---

## 自动化机制

### 1. 状态文件

每个 phase 完成后写入状态文件：

```
runs/experiments/LONG_HAUL_STATE.json
```

内容：
```json
{
  "phase": "P1",
  "status": "completed",
  "result": "AP50_adv=0.45",
  "next_phase": "P2",
  "timestamp": "2026-06-22T10:00:00Z"
}
```

### 2. 自动降级规则

| 阶段 | 失败条件 | 降级动作 |
|------|---------|---------|
| P1 | 攻击无法产生 AP50_drop ≥ 0.15 | 尝试 multiple patches / 更大 patch / torso 位置 / full-image PGD；最多 3 次 |
| P2 | clean_drop > 0.05 | 降低 lambda_step，提高 anomaly_threshold，只改高频；最多 3 次 |
| P3 | recovery < 0.5 | 只保留 96/128 中 best config，进入 P4 |
| P4 | 训练崩溃 | 修复 bug 后重试 1 次；仍失败则跳过到 P5 |
| P5 | 训练崩溃 | 跳过，不影响最终汇总 |

### 3. 监控与恢复

- 每个长时间运行的训练脚本通过 `tee` 保存日志；
- 每 10 分钟检查一次进程状态；
- 若进程异常退出，根据日志判断是否为 OOM，自动重试（降低 batch size）。

### 4. 通知

每完成一个 phase，在 `runs/experiments/LONG_HAUL_LOG.txt` 中追加一行：

```
[2026-06-22 10:00] P1 completed: AP50_adv=0.45 -> next P2
```

---

## 交付产物

1. `scripts/experiments/plan_b_smoke_test.py`（修复版）
2. `scripts/experiments/plan_b_full_experiment.py`
3. `scripts/experiments/plan_c_smoke_test.py`
4. `scripts/experiments/plan_d_smoke_test.py`
5. `runs/experiments/LONG_HAUL_STATE.json`
6. `runs/experiments/LONG_HAUL_LOG.txt`
7. `runs/experiments/SUMMARY_8H.md`
8. `runs/experiments/figures/*.png`

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 攻击仍无法有效 | 使用 ensemble attack、multiple patches、不同损失函数 |
| GPU OOM | 自动降低 batch size / defense_size |
| 训练时间过长 | 先 smoke test，成功后再扩展 |
| 代码 bug 导致崩溃 | 每个脚本先用 1-2 张图 dry-run |
| 8h 内完不成 | Phase 4-5 可并行：Plan C 和 Plan D 同时跑 |

---

## 启动命令

```bash
# 启动 8 小时长程任务（作为后台任务）
E:/anaconda/01/envs/RLimage/python.exe scripts/experiments/run_long_haul_8h.py \
  --output-dir runs/experiments/long_haul_8h \
  --device cuda \
  > runs/experiments/long_haul_8h.log 2>&1 &
```

---

## 下一步

确认此 plan 后，我将：
1. 创建 `scripts/experiments/run_long_haul_8h.py`（总控脚本）；
2. 先修复 `plan_b_smoke_test.py` 的攻击和防御；
3. 启动后台任务，让其自动推进 8 小时；
4. 建立监控机制，定时检查状态并汇报。
