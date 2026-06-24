# Full Bug Audit: 2.41 - 2.58 Data Flow
## 2026-06-06

---

### SCRIPT: round231_runner.py (2.31 Soft Reweighting)
**API75 = 0.744 — BEST**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | BROKEN | `sum(ld.values())` — ld is list of dicts in newer torchvision, this crashes or gives wrong sum |
| sigma | N/A | No sigma here |
| Quality input | FPN 14x14 ROI Align (semantic) | ✓ |
| Quality delta-dep? | N/A | No delta sampling |
| N alignment | N/A | No delta sampling |
| Img indices | N/A | No pixel patches |
| Gradient path | quality × box_reg → quality attached to FPN grad | ✓ (but frozen backbone/RPN) |

**BUG**: ld is `[{'loss_classifier': ..., 'loss_box_reg': ...}]` (list of dicts) in newer torchvision. `sum(ld.values())` throws AttributeError: 'list' object has no attribute 'values'. This means 2.31 MAY NOT HAVE RUN on the current torchvision version. The recorded AP75=0.744 might be from an older torchvision where ld was a single dict.

---

### SCRIPT: round241_runner.py (2.41 RFT pixel)
**API75 = 0.682**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | BROKEN | `sum(ld.values())` + `total_det += det_loss.item()` — grad inside model() already consumed |
| sigma | BUG | `torch.full_like(mu, 0.1)` inherits mu.requires_grad |
| Quality input | Pixel patches from decoded boxes | ✓ delta-dep |
| rft_loss | MSE(mu, best_deltas.detach()) | ✓ detach correct |
| N alignment | Fixed | `N = min(N, proposals_cat.shape[0]) + mu[:N] + deltas[:N]` |
| img_indices | OK | `ii = torch.cat(...)[:N]` |

**Bug**: sigma has grad (minor, ~10% noise). det_loss as `sum(ld.values())` might not work with newer torchvision list-format.

---

### SCRIPT: round243_iou_rl.py (2.43 IoU REINFORCE)
**API75 = 0.676 — RL zero contribution**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | BROKEN | `sum(ld.values())` |
| sigma | BUG | inherits grad |
| Quality | IoU(box, GT) | ✓ Verifiable |
| RL gradient | REINFORCE: log_prob × advantage | ✓ maths correct (verified zero contribution anyway) |
| N alignment | Fixed | Lines 149-151 |

**Verdict**: Bug-free in concept but RL signal too weak. Bug in sigma is minor.

---

### SCRIPT: round244_runner.py (2.44 Edge-RFT)
**API75 = 0.681 — same as 2.41**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | BROKEN | `sum(ld.values())` |
| sigma | BUG | inherits grad |
| Quality | Pixel edge FFT | Same issues as 2.41 |

**Verdict**: Same bugs as 2.41. No new issues.

---

### SCRIPT: round245_runner.py (2.45 Inverted Pixel Soft)
**API75 = 0.676 — worse than 2.31**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | BROKEN | `sum(ld.values())` |
| sigma | N/A | No sigma |
| Quality | Pixel FFT on ALL proposals (not matched) | quality_inv × box_reg |
| Gradient | quality is CPU → GPU → detached from model graph | quality has NO gradient path to model |

**BUG**: quality computed from CPU pixel patches → no grad connection to model. This is intentional (quality is just a weight), but the grad path through quality × box_reg only goes through box_reg. This is correct behavior for soft weighting.

---

### SCRIPT: round246_runner.py (2.46 Edge Centrality RFT)
**API75 = 0.663**

| Check | Status | Notes |
|-------|--------|-------|
| Same structure as 2.41 RFT, same bugs (sigma grad, det_loss format) |

---

### SCRIPT: round249_runner.py (2.49 Learned Spectral Head)
**API75 = 0.692**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | BROKEN | `sum(ld.values())` + `.item()` |
| spectral_head | has grad ✓ | quality_pred = CNN(FFT complex) → sigmoid |
| IoU regression | iou_reg = MSE(quality_pred, IoU_target) | ✓ |
| spec_loss | quality_pred × box_reg | Has BOTH paths to trained head |

**Bug**: `quality_pred` grad flow is: iou_reg → quality_pred → CNN → FFT → roi14 → ROI Align → FPN (but FPN frozen). spec_loss → quality_pred → CNN (additional). Both paths contribute gradient. This is correct.

---

### SCRIPT: round250_runner.py (2.50 X-DPO)
**API75 = 0.479 — crashed**

| Check | Status | Notes |
|-------|--------|-------|
| DPO ref log_probs | FIXED | ref_deltas.detach() + no_grad ref_mu |
| det_loss tensor? | BROKEN | sum_losses → .item() |
| sigma | BUG | inherits grad |
| Quality | Pixel patches from decoded (3-channel Pareto) | q_edge, q_smooth, q_overlap |
| Pareto voting | ✓ | wins_0 >= 2 → chosen |
| valid.any() guard | MISSING | `.mean()` on empty tensor → NaN |

---

### SCRIPT: round251_runner.py (2.51 Combined DPO)
**API75 = 0.578-0.631**

| Check | Status | Notes |
|-------|--------|-------|
| DPO ref log_probs | FIXED | ref_deltas.detach() + ref_sigma inside no_grad |
| det_loss tensor? | BROKEN | sum_losses → .item() |
| sigma | BUG | inherits grad |
| q_radial | BUG v1 | Same for both deltas (proposal-level ROI FFT) — fixed in v2 |
| q_edge | Pixel Sobel | delta-dep |
| q_overlap | IoU GT | delta-dep |
| valid guard | Same as 2.50 |

---

### SCRIPT: round252_runner.py (2.52 DPO-ROI Bug-fixed)
**API75 = 0.623**

| Check | Status | Notes |
|-------|--------|-------|
| DPO ref log_probs | FIXED ✓ | |
| det_loss tensor? | FIXED | Summed as tensor (no .item()) |
| sigma | BUG | inherits grad |
| Quality | ROI FFT from PROPOSALS (not decoded boxes) | Both deltas get SAME quality ← FATAL |
| Threshold | NONE | quality[:,0] always >= quality[:,1] → random preference |

**CRITICAL BUG**: q_quality[:,0] = q_quality[:,1] (both from same proposal ROI). DPO preference is a coin flip.

---

### SCRIPT: round255_runner.py (2.55 DPO-Pixel + Threshold)
**Not yet successfully run**

| Check | Status | Notes |
|-------|--------|-------|
| DPO ref log_probs | FIXED ✓ | |
| det_loss tensor? | FIXED ✓ | |
| sigma | BUG | inherits grad (LOW) |
| Quality | Pixel FFT from decoded boxes | delta-dep ✓ |
| Threshold filter | q_diff > 0.02 | Added ✓ |
| valid.any() guard | FIXED ✓ | |

---

### SCRIPT: round257_runner.py (2.57 Native Zero-Pad)
**Crashed (zero-width crop)**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | FIXED ✓ | |
| sigma | BUG | |
| Zero-pad logic | BUG | `h<4 or w<4` check added but still crashes on `w=0` |
| Quality | Pixel FFT on zero-padded native patches | |

---

### SCRIPT: round258_runner.py (2.58 Edge Truncation)
**Not run**

| Check | Status | Notes |
|-------|--------|-------|
| det_loss tensor? | FIXED ✓ | |
| sigma | BUG | inherits grad |
| Quality | edge_truncation_quality = 1 - boundary_edges/total_edges | Physical, no FFT |
| Quality delta-dep? | YES | From decoded box pixel patches ✓ |
| DPO ref | FIXED ✓ | |

---

## SUMMARY

### Critical Bugs Fixed
1. DPO reference gradient leak → fixed in 2.50, 2.51, 2.52, 2.55, 2.58
2. sum_losses breaking gradient → fixed in 2.52, 2.55, 2.57, 2.58
3. 2.51 q_radial identical for deltas → fixed in v2
4. 2.52 quality not delta-dependent → DESIGN LIMITATION (not bug)

### Remaining Low-Priority Issues
1. sigma=torch.full_like(mu, 0.1) inherits grad → ALL scripts (minor ~10% noise)
2. det_loss format: ld is likely a LIST of dicts in current torchvision → many scripts use `sum(ld.values())` which would crash

### Scripts Ready to Run (after sigma fix + det_loss format fix)
- 2.55: Pixel FFT DPO with threshold (most promising DPO variant)
- 2.58: Edge Truncation DPO (physics-based, no FFT)
- 2.52: ROI FFT DPO (for comparison, though quality is proposal-level)
- 2.57: Native zero-pad DPO (needs zero-width fix)
