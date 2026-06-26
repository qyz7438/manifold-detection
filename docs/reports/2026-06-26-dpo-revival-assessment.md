# DPO 线是否值得捡起来：评估报告

## 1. 历史结果

### 1.1 Round 2218-2220 短期扫描

在 NWPU 上跑 5 epoch 的 DPO 短期扫描：

| 配置 | AP75 | vs baseline |
|---|---:|---:|
| Baseline | 0.2939 | — |
| DPO-only w=0.1 | 0.2988 | **+0.0049** |
| DPO-only w=0.03 | 0.2986 | +0.0047 |
| DPO + rescue w=0.03 | 0.2989 | **+0.0050** |
| cls_score-only DPO | 0.2958 | +0.0019 |

最佳配置 `det.dpo.smoke.001`：
- `pre_nms_dpo_loss_weight=0.03`
- DPO + rescue
- `trainable_mode=predictor`
- 5 epoch AP75 +0.005

### 1.2 历史结果的问题

1. **单 seed，未验证**：round2218-2220 都是单 seed smoke，没有 cross-seed 验证。
2. **提升幅度小**：+0.005 AP75 在统计上可能不稳定，尤其在 NWPU 这种方差大的数据集上。
3. **LC-HI 覆盖率瓶颈**：verifier-positive 子集有提升，但整体 LC-HI pool 的 mean score shift 仍为负：
   - `verifier_positive_lchi_prob_delta_mean = +0.0095`
   - `lchi_prob_delta_mean = -0.0020`
4. **已知梯度 bug**：commit `734864c` 明确标注 2.50-2.60 的 DPO/RL 实验存在 **action 未 detach 的梯度 bug**，结果不可靠。虽然 round2218-2220 不在这个范围内，但它们与 bug 版本处于同一套代码体系，且整套 RLVR/DPO 代码已在 `73fd3be` 被 archive 到 `legacy/`。

## 2. 代码现状

### 2.1 当前分支缺少 DPO 基础设施

当前 `manifold-main` 分支：
- `spectral_detection_posttrain/rlvr/` 目录**已删除**
- `spectral_detection_posttrain/rlvr/action_verifier.py` **不存在**
- DPO 相关脚本（如 `round2129_nwpu_posttrain_smoke.py`）虽然还在 `scripts/`，但依赖的 `rlvr` 和 `models` 模块已不存在
- 相关代码被整体 archive 到 `legacy/methods/dpo/`

### 2.2 捡起来需要做什么

如果要重新跑 DPO，需要：

1. 从 `legacy/` 恢复 `action_verifier.py` 等核心模块
2. 适配到当前的 `core/models/build_detector.py` 结构
3. 适配到当前的 `round28_train_eval.py` 训练流程
4. 修复/验证 action detach 问题
5. 重新实现 rescue、confidence rescue、verifier score 等周边逻辑

工作量不小，相当于把一条已 archive 的实验线重新接回当前分支。

## 3. 历史数据是否说明 DPO 有意义？

### 3.1 支持继续的论据

- DPO-only 在 weight 0.01→0.03→0.1 上呈现**单调提升**，说明存在真实信号
- DPO + rescue 略优于 DPO-only，说明与 rescue 机制有协同
- predictor/adaptor 路径优于 cls_score-only，说明给网络足够的容量很重要
- verifier-positive 子集的 LC-HI score 确实在动，只是覆盖率低

### 3.2 反对继续的论据

- 提升幅度小（+0.005），且未 cross-seed 验证
- 整体 LC-HI pool 没有救回来，说明 verifier 本身不够强
- 代码已 archive，捡起来工程成本高
- 已知同一代码体系有 gradient bug，历史结果可信度打折
- 当前主线 LSG/AFM 还在验证中，分散精力不划算

### 3.3 结论

**DPO 历史数据说明它可能是一个小但是真实的信号，但不足以作为当前优先方向。**

根本瓶颈和 FSBG 一样：**verifier 信号不够强，覆盖率低**。在 verifier 本身没有改善之前，DPO 很难再上一层楼。

## 4. 建议策略

### 4.1 短期：不捡 DPO

当前优先级：
1. 完成 LSG v1 Phase 1（AFM / LSG no-radius / LSG radius）
2. 如果 LSG 有效，说明频域门控可以作为更强的特征/verifier 基础
3. 基于改进后的特征表示，再考虑 DPO

### 4.2 中期：用 LSG 增强 DPO verifier

如果 LSG 有效，DPO 的重新打开方式应该是：

- 用 LSG 输出的频域门控特征替代或增强 `fft_edge_truncation` / `phase_edge` / `phase_abs_high` 等手工 verifier 特征
- 在 LSG 改善的 base detector 上做 DPO + rescue
- 重新验证 action detach，跑 3 seeds

这样 DPO 不再是孤立的优化，而是建立在更强的频域表示之上。

### 4.3 长期：DPO 作为 policy 优化层

如果 LSG 把 base AP75 提上去，DPO 可以承担 policy 优化的角色：

- 利用 LSG 学到的频率选择性作为 preference signal
- DPO 优化 cls_score / bbox adapter 的决策边界
- 与 LSG 形成 "表示学习 + 策略优化" 的组合

## 5. 最终结论

| 问题 | 结论 |
|---|---|
| DPO 历史结果是否真实？ | 可能真实，但小且未充分验证 |
| 是否值得现在捡起来？ | **不建议** |
| 什么时候捡？ | LSG/AFM 验证有效后，用 LSG 特征增强 verifier 再捡 |
| 最大风险 | 工程成本高，且 verifier 瓶颈未解决 |

一句话：**DPO 不是死线，但应该先让频域表示这条线跑通，再把它接回去。**
