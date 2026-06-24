# Plan 2.12: MPLSeg-Style AFM — 修复梯度阻断

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 MicroAFM 中 `mag_scale`/`phase_scale` 梯度为严格零的架构缺陷，改为 MPLSeg 风格的硬编码门控（无 learnable scale、InstanceNorm、Sigmoid 门控），验证梯度恢复正常。

**Architecture:** 在 `micro_afm.py` 中新增 `MPLSegAFMBlock`（保持旧 `AFMBlock` 不变）。门控使用 MPLSeg 原始设计的 `1 - sigmoid(sigmoid(log(mag)))` 压制方案，相位使用 `pa(pha)` 直接残差。保留 `residual_scale` 作为唯一的可学习缩放参数，不改变训练模式。Penn-Fudan 小规模验证。

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN MobileNetV3 FPN, Penn-Fudan, pytest, 现有 `spectral_detection_posttrain` package。

---

## 根因分析摘要

2026-06-04 梯度诊断脚本 (`scripts/diagnose_afm_gradients.py`) 确认：

```
当前 AFMBlock 梯度范数:
  mag_scale:       0.0  ← 严格零
  phase_scale:     0.0  ← 严格零
  mag_gate conv:   0.0  ← 严格零
  phase_res conv:  0.0  ← 严格零
  residual_scale:  2.0  ← 正常非零

MPLSeg-style AFM 梯度范数:
  mp conv权重:     0.005 ~ 0.014  ← 正常非零
  pa conv权重:     0.002 ~ 0.007  ← 正常非零
```

**死锁机制**：

```
residual_scale = 0 (初始化)
  → ∂(output)/∂(freq_out) = residual_scale = 0
  → ∂(loss)/∂(mag_scale) = 0 × ∂(freq_out)/∂(mag) × ∂(mag)/∂(mag_scale) = 0
  → mag_scale 永远得不到梯度，mag_gate/phase_res 的 conv 权重也得不到梯度
```

即使训练后 `residual_scale` 从 0 变为 0.03，FFT 路径梯度被 `0.03 × iRFFT链路数值噪声` 双重衰减，无法与直接残差路径竞争。

**MPLSeg 为什么能工作**：不使用 learnable scale 门控 FFT 路径，门控始终激活。

---

## 7 个架构差异对照

| | 当前 AFMBlock | MPLSeg 原始 | Plan 2.12 新 AFM |
|---|---|---|---|
| 幅度门控 | `1 + mag_scale × tanh_gate(log1p(mag))` | `1 - sigmoid(sigmoid(log(mag)))` | 采用 MPLSeg 方案 |
| 相位残差 | `pha + phase_scale × tanh_res(pha)` | `pha + pa(pha)` | 采用 MPLSeg 方案 |
| 门控归一化 | 无 | InstanceNorm2d 每层 | 采用 MPLSeg 方案 |
| 门控激活 | Tanh → [-1,1] | Sigmoid → [0,1] | 采用 MPLSeg 方案 |
| 卷积初始化 | zeros_ | kaiming_normal_ | 采用 MPLSeg 方案 |
| 可学习 scale | mag_scale, phase_scale | 无 | 无 |
| 残差连接 | `x + residual_scale × residual` | 无残差 | 保留 `x + residual_scale × freq_out` |

**关键改动**：去掉 `mag_scale` 和 `phase_scale` 两个可学习参数，门控硬编码激活。保留 `residual_scale`（已验证可正常获得梯度）。保留残差连接形态但改用 MPLSeg 风格的门控内部实现。

---

## 文件结构

- **创建**: `tests/test_round212_mplseg_afm.py` — 新 AFM 的单元测试和梯度测试
- **修改**: `spectral_detection_posttrain/models/micro_afm.py` — 添加 `MPLSegAFMBlock`，更新 factory
- **修改**: `scripts/round28_train_eval.py` — 在 `_read_afm_scales` 中适配新 AFM（`MPLSegAFMBlock` 没有 mag_scale/phase_scale）
- **复用**: `scripts/diagnose_afm_gradients.py` — 验证新 AFM 梯度正常
- **复用**: `tests/test_round28_afm_variants.py` — 确保旧 AFM 测试仍然通过

---

## Task 1: MPLSegAFMBlock 实现 + 测试

**Files:**
- Create: `tests/test_round212_mplseg_afm.py`
- Modify: `spectral_detection_posttrain/models/micro_afm.py`

- [ ] **Step 1: 写测试（TDD RED）**

Create `tests/test_round212_mplseg_afm.py`:

```python
import torch

from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock, build_afm_block


def test_mplseg_afm_output_shape():
    afm = MPLSegAFMBlock(channels=64)
    x = torch.randn(2, 64, 24, 24)
    out = afm(x)
    assert out.shape == x.shape


def test_mplseg_afm_is_not_identity_at_init():
    """MPLSeg-style AFM should NOT be identity at init because gate is active."""
    afm = MPLSegAFMBlock(channels=64)
    x = torch.randn(2, 64, 24, 24)
    out = afm(x)
    diff = (out - x).abs().mean().item()
    assert diff > 1e-3, f"MPLSeg AFM should modify features, but diff={diff}"


def test_mplseg_afm_no_nan():
    afm = MPLSegAFMBlock(channels=64)
    x = torch.randn(2, 64, 24, 24)
    out = afm(x)
    assert torch.isfinite(out).all()


def test_mplseg_afm_mag_gate_gets_gradient():
    """The mag gate convs MUST receive non-zero gradient on backward."""
    afm = MPLSegAFMBlock(channels=16)
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    out = afm(x)
    target = torch.randn_like(out) * 0.1
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    # Check first conv in mp gate
    mp_conv = afm.mp[0]
    assert mp_conv.weight.grad is not None, "mp conv grad is None"
    assert mp_conv.weight.grad.abs().sum() > 0, "mp conv grad is zero"


def test_mplseg_afm_phase_res_gets_gradient():
    """The phase residual convs MUST receive non-zero gradient on backward."""
    afm = MPLSegAFMBlock(channels=16)
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    out = afm(x)
    target = torch.randn_like(out) * 0.1
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    pa_conv = afm.pa[0]
    assert pa_conv.weight.grad is not None, "pa conv grad is None"
    assert pa_conv.weight.grad.abs().sum() > 0, "pa conv grad is zero"


def test_mplseg_afm_residual_scale_gets_gradient():
    """residual_scale must get non-zero gradient."""
    afm = MPLSegAFMBlock(channels=16)
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    out = afm(x)
    target = torch.randn_like(out) * 0.1
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    assert afm.residual_scale.grad is not None
    assert abs(afm.residual_scale.grad.item()) > 0


def test_mplseg_afm_factory():
    afm = build_afm_block("mplseg", channels=32)
    assert isinstance(afm, MPLSegAFMBlock)
    assert build_afm_block("none", channels=32) is None


def test_mplseg_afm_output_nonnegative():
    """MPLSeg applies ReLU after iRFFT, so output should be >= 0 (mostly)."""
    afm = MPLSegAFMBlock(channels=16)
    x = torch.randn(2, 16, 24, 24)
    out = afm(x)
    # ReLU ensures output >= 0
    assert (out >= 0).all(), f"min={out.min().item()}"
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round212_mplseg_afm.py -v
```

Expected: FAIL — `MPLSegAFMBlock` not defined.

- [ ] **Step 3: 实现 MPLSegAFMBlock**

Modify `spectral_detection_posttrain/models/micro_afm.py`:

在 `AFMBlock` 类定义之后（line 97 后），添加：

```python
class MPLSegAFMBlock(nn.Module):
    """MPLSeg-style AFM: hard-coded active gate, InstanceNorm, no learnable scales.

    Gate design from MPLSeg (Yan et al., 2024):
      mag = mag * (1 - sigmoid(sigmoid(log(mag))))   // suppression gate
      pha = pha + pa(pha)                             // direct phase residual
      output = ReLU(iRFFT(mag * exp(j*pha)))          // post-iRFFT activation

    Residual connection: out = x + residual_scale * output
    Only residual_scale is a learnable parameter.
    """

    def __init__(self, in_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid = mid_ch or in_ch

        self.mp = nn.Sequential(
            nn.Conv2d(in_ch, mid // 4, 1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Sigmoid(),
            nn.Conv2d(mid // 4, mid // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Sigmoid(),
            nn.Conv2d(mid // 4, mid, 1, bias=False),
            nn.InstanceNorm2d(mid),
            nn.Sigmoid(),
        )
        self.pa = nn.Sequential(
            nn.Conv2d(in_ch, mid // 4, 1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Tanh(),
            nn.Conv2d(mid // 4, mid // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Tanh(),
            nn.Conv2d(mid // 4, mid, 1, bias=False),
            nn.InstanceNorm2d(mid),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self._eps = 1e-3

        for seq in [self.mp, self.pa]:
            for m in seq:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + self._eps)

        mag = mag * (1.0 - self.mp(torch.sigmoid(torch.log(mag + self._eps))))
        pha = pha + self.pa(pha)

        fr = mag * torch.exp(1j * pha)
        freq_out = torch.fft.irfft2(fr, norm="ortho")
        freq_out = F.relu(freq_out, inplace=False)

        return x + self.residual_scale * freq_out
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round212_mplseg_afm.py -v
```

Expected: 8 passed.

- [ ] **Step 5: 运行旧测试确认没有回归**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round28_afm_variants.py tests/test_micro_afm.py -v
```

Expected: 11 passed.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_round212_mplseg_afm.py spectral_detection_posttrain/models/micro_afm.py
git commit -m "feat: add MPLSegAFMBlock with active gate and InstanceNorm"
```

---

## Task 2: 更新 factory 和 build_detector

**Files:**
- Modify: `spectral_detection_posttrain/models/micro_afm.py` — `build_afm_block` factory
- Modify: `spectral_detection_posttrain/models/build_detector.py` — 支持 `afm_type="mplseg"`

- [ ] **Step 1: 更新 build_afm_block**

修改 `micro_afm.py` 中的 `build_afm_block` 函数：

```python
def build_afm_block(afm_type: str, channels: int, residual_mode: str = "current") -> nn.Module | None:
    if afm_type == "none":
        return None
    if afm_type == "old":
        return OldAFMBlock(channels=channels)
    if afm_type == "identity":
        return AFMBlock(channels=channels, residual_mode=residual_mode)
    if afm_type == "mplseg":
        return MPLSegAFMBlock(in_ch=channels)
    raise ValueError(f"Unknown afm_type: {afm_type}")
```

- [ ] **Step 2: 更新 build_detector 支持 mplseg afm_type**

`build_detector.py` 中 `build_detector` 函数需要处理 `afm_type="mplseg"`。检查当前代码是否已通过 `build_afm_block` 工厂自动适配——如果 `afm_channels` 配置不同需要调整。

确认 `build_detector.py` 中 AFM 创建是通过 `build_afm_block(afm_type, channels, residual_mode)` 调用。`MPLSegAFMBlock` 接受 `in_ch` 参数（与 `AFMBlock` 的 `channels` 参数语义相同），接口兼容。

- [ ] **Step 3: 更新 _read_afm_scales 适配 MPLSegAFMBlock**

`scripts/round28_train_eval.py` 的 `_read_afm_scales` 函数尝试读取 `mag_scale`/`phase_scale`/`residual_scale`。`MPLSegAFMBlock` 只有 `residual_scale`。修改函数使其检查 hasattr 后优雅降级（已有此逻辑，无需修改）。

- [ ] **Step 4: 运行梯度诊断脚本验证新 AFM**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe scripts/diagnose_afm_gradients.py
```

Expected: MPLSeg-style AFM 的 conv 权重梯度 > 0。

- [ ] **Step 5: Commit**

```powershell
git add spectral_detection_posttrain/models/micro_afm.py
git commit -m "feat: add mplseg afm_type to build_afm_block factory"
```

---

## Task 3: 训练对比实验（3 组）

**Files:**
- Reuse: `scripts/round28_train_eval.py` (no modifications needed)

- [ ] **Step 1: 运行 baseline（无 AFM）**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_train_eval.py `
  --run-name round212_baseline `
  --afm-type none `
  --trainable-mode full `
  --epochs 1 --seed 42
```

- [ ] **Step 2: 运行旧 identity AFM（对照组）**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_train_eval.py `
  --run-name round212_identity_afm `
  --afm-type identity `
  --afm-residual-mode current `
  --trainable-mode full `
  --epochs 1 --seed 42
```

- [ ] **Step 3: 运行 MPLSeg-style AFM**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_train_eval.py `
  --run-name round212_mplseg_afm `
  --afm-type mplseg `
  --trainable-mode full `
  --epochs 1 --seed 42
```

- [ ] **Step 4: 验证结果**

检查每个 group 的 `runs/<run_name>/eval_metrics.json`：
- `round212_baseline/eval_metrics.json` — AP50 参考基准
- `round212_identity_afm/eval_metrics.json` — mag_scale/phase_scale 应仍为 0（旧行为）
- `round212_mplseg_afm/eval_metrics.json` — residual_scale 应非零，AP50 应与 baseline 可比

```powershell
E:\anaconda\01\envs\RLimage\python.exe -c "
import json
for g in ['round212_baseline', 'round212_identity_afm', 'round212_mplseg_afm']:
    p = f'runs/{g}/eval_metrics.json'
    d = json.load(open(p))
    h = d.get('history', [{}])[-1] if d.get('history') else {}
    mag = h.get('mag_scale', 'N/A')
    pha = h.get('phase_scale', 'N/A')
    res = h.get('residual_scale', 'N/A')
    print(f'{g}: AP50={d.get(\"ap50\",\"?\")} mag_s={mag} pha_s={pha} res_s={res}')
"
```

- [ ] **Step 5: Commit 结果**

```powershell
git add runs/round212_baseline/eval_metrics.json runs/round212_identity_afm/eval_metrics.json runs/round212_mplseg_afm/eval_metrics.json
git commit -m "feat: Plan 2.12 3-group AFM comparison results"
```

---

## 成功标准

```text
1. test_mplseg_afm_mag_gate_gets_gradient PASS — mag gate conv 梯度非零
2. test_mplseg_afm_phase_res_gets_gradient PASS — phase residual conv 梯度非零
3. test_mplseg_afm_residual_scale_gets_gradient PASS — residual_scale 梯度非零
4. 旧 6 个 AFM 测试仍然通过（无回归）
5. round212_mplseg_afm 的 residual_scale 不为 0（FFT 路径确实在学习）
6. round212_mplseg_afm 的 AP50 不会比 baseline 差 >5%（门控不破坏特征）
```

## 判读规则

```text
如果 residual_scale > 0.01 且 AP50 ≈ baseline:
  → MPLSeg-style 门控在检测预训练特征上成功激活，梯度恢复正常。
    解释: 旧 AFM 的问题是架构拓扑缺陷（learnable scale 阻塞梯度），不是"预训练特征与 FFT 不兼容"。

如果 residual_scale ≈ 0 但 conv 梯度非零:
  → 门控在训练但 residual_scale 没移动，FFT 路径在学习但 optimizer 选择绕过它。
    可能是 1 epoch 不够，需要多 epoch 观察。

如果 residual_scale ≈ 0 且 AP50 >> baseline:
  → 门控没有贡献，但残余梯度不影响检测 loss 正常优化。

如果 AP50 崩溃:
  → MPLSeg-style 门控强度太大，预训练特征被 FFT 变换破坏。
    可能需要在残差前加 output_scale 或减小门控强度。
```
