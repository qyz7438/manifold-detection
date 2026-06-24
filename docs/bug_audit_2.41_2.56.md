# 2.41-2.56 Bug Audit Report
## 2026-06-06 - Data Flow & Implementation Issues

---

### BUG 1 [CRITICAL]: DPO log_probs_ref gradient leak (2.50, 2.51)

**File:** round250_runner.py:169, round251_runner.py:165+

**Bug:** `log_probs_ref = gaussian_log_prob(deltas, ref_mu, ref_sigma)` is OUTSIDE `torch.no_grad()`. 
`deltas` has grad (computed from `μ + σ·ε`). Even though ref_mu/ref_sigma are detached, 
the deltas dependency creates a gradient path through log_probs_ref back to μ.

**Impact:** The DPO ratio term `lp_c - lp_ref_c - lp_r + lp_ref_r` should only have gradient 
from `lp_c` and `lp_r` (current policy). The reference terms should be constant baselines. 
With this bug, the reference terms ALSO push μ, creating gradient conflict. 
This corrupts the entire DPO optimization direction.

**Fix:** `ref_deltas = deltas.detach()` before computing log_probs_ref.

---

### BUG 2 [HIGH]: DPO sigma passed as model's trainable sigma to ref (2.51)

**File:** round251_runner.py 

**Bug:** `log_probs_ref = gaussian_log_prob(deltas, ref_mu, sigma)` — `sigma` is the 
TRAINING model's sigma tensor (requires_grad=True), not a detached reference sigma.
This adds ANOTHER gradient path from sigma through log_probs_ref.

**Fix:** Use explicit `ref_sigma = torch.full_like(ref_mu, 0.1)` like 2.50 does.

---

### BUG 3 [HIGH]: All DPO/RFT experiments use pixel FFT instead of ROI FFT (2.41-2.51)

**File:** round241-251_runner.py (all)

**Bug:** The ONLY working approach (2.31, AP75 +27.4%) uses ROI-level FFT on 14×14 FPN features 
(semantic space). All subsequent DPO/RFT experiments (2.41-2.51) switched to pixel-level FFT
on 64×64 cropped image patches (texture space).

ROI FFT in semantic space naturally encodes class-aware information (CNN features know 
"person" vs "wall"). Pixel FFT only captures texture complexity, which correlates 
NEGATIVELY with IoU (r=-0.606, proven in diagnostic).

**Impact:** All experiments after 2.31 are using a fundamentally wrong quality signal.
The DPO framework might actually work — but with a quality metric that's anti-correlated 
with actual detection quality, no optimization algorithm can succeed.

---

### BUG 4 [MEDIUM]: 2.51 q_radial computed from proposal ROI (identical for both deltas)

**File:** round251_runner.py (v1, fixed in v2)

**Bug:** `q_radial` was computed from proposal-level ROI Align features, not from pixel patches.
Both sampled deltas share the same proposal → same ROI features → same q_radial.
The radial energy channel provides ZERO pairwise discrimination.

**Impact:** 3-channel Pareto reduced to 2 effective channels. 100% valid pair rate actually 
came from 2 working channels always producing a 2/3 winner, not from 3 independent channels.

**Fix:** Applied in v2 — q_radial now uses pixel patch FFT. But doesn't fix the pixel FFT 
fundamental problem (Bug 3).

---

### BUG 5 [MEDIUM]: 2.49 LearnedSpectralHead uses ROI features (256ch → complex) — 
semantic not physical

**File:** round249_runner.py

**Bug:** SpectralHead does `FFT(roi_features)` on (256, 14, 14) ROI features, then 
`mag.real.imag` mean over channels. The 256 channels are CNN features — each channel 
represents a different semantic feature map. Averaging over them loses ALL semantic 
information. The result is a "mean FFT" of 256 unrelated feature maps.

**Expected:** Should do FFT on EACH channel independently (256 separate FFTs) and 
concatenate the complex outputs, preserving per-channel frequency information.

**OR:** Should do FFT on the pooled feature (like 2.31 does — FFT on mean of channels, 
which already shows semantic correlation).

---

### BUG 6 [LOW]: Pixel patch cropping uses original CPU images — uint8 range [0,255]

**File:** round241-251_runner.py (all pixel-based)

**Bug:** All pixel patches are cropped from original CPU uint8 images. F.interpolate with 
.float() conversion returns values in [0, 255]. But FFT of large-valued inputs has 
different numerical properties than normalized [0,1] inputs.

**Impact:** Minor — FFT magnitude scales with input magnitude, but quality formula uses 
ratios (HF/total, normalized entropy) which are scale-invariant.

---

### BUG 7 [LOW]: sigma is recreated each batch (not a bug, but suboptimal)

**File:** round241-251_runner.py (all DPO/RFT)

**Observation:** sigma = torch.full_like(mu, 0.1) is created fresh each batch. This is 
correct behavior for a fixed sigma. But σ=0.1 is too small for pairwise discrimination.

**Not a bug, but the root cause of all DPO failures:** two deltas at σ=0.1 differ by only 
~1.1px in center position. No quality metric (pixel OR ROI) can reliably distinguish 
which delta is better at this resolution.

---

### VERIFIED OK:
- Hook mechanism: correct (pre_hook captures roi_features BEFORE box_head)
- Detach chain: δ_best correctly detached in RFT (MSE target)
- Indexing: proposals ↔ images mapping is correct (npi/ii alignment)
- N slicing: patched in all scripts (mu[:N], deltas[:N], sigma[:N])
- Reference model: deepcopy + freeze is correct
- GRPO: NOT implemented in any experiment (only REINFORCE and DPO tried)
- ARS: NOT implemented in any experiment (only diagnosed)
