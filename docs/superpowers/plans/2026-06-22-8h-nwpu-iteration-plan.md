# 8-Hour NWPU Iteration Plan (2026-06-22) — 短周期探测版

> **Goal:** Find a direction that improves NWPU val AP75 beyond `round2214_lr1e4_30ep_clean` best AP75 = 0.3066, using short 3-5 epoch runs first; scale up only on the winning direction.
> **Rule:** Each experiment is ≤ 5 epochs (~25-40 min). After each, read `eval_metrics.json` and decide the next step.

---

## Current state

| Run | best AP75 | final AP75 | notes |
|-----|-----------|------------|-------|
| round2211_relax_det05_lr1e4_clean_hd_fusion_15ep | 0.3026 | 0.3026 | 15-epoch rescue baseline |
| round2212_fft_ranking_global_clean_15ep | 0.3017 | 0.3013 | verifier-ranking w=0.01 |
| round2213_fft_ranking_w003_clean_15ep | 0.3020 | 0.3020 | verifier-ranking w=0.003 |
| **round2214_lr1e4_30ep_clean** | **0.3066** | **0.3039** | **current best**; best epoch 23 |
| round2215_policy003_clean_15ep | 0.3015 | 0.3012 | policy weight 0.003 |
| round2216_policy005_clean_15ep | 0.3016 | 0.3012 | policy weight 0.005 |
| round2217_softgate_clean_15ep | 0.3019 | 0.3018 | sigmoid verifier weight |
| round2218-2220 pre-NMS DPO sweeps | ~0.298 | ~0.298 | DPO weaker than rescue-only |

Offline interpretable diagnostics (`round2221`) on real NWPU cache:
- Best fusion: `score_edge_alignment` + `boundary_phase_coherence` + `interior_exterior_texture_contrast` + raw-iFFT recipe.
- Fixed-threshold transfer: P=0.812 / R=0.317, AUC=0.836, AP=0.451.

Real-data complementarity (`runs/nwpu_signal_complementarity_analysis`):
- Most complementary: `score_edge_alignment` ↔ `phase_edge_64` (|ρ|=0.001), `aspect_ratio_plausibility` ↔ `phase_abs_high_11` (|ρ|=0.004).
- Most redundant: `reference_raw_ifft_recipe` ↔ `phase_edge_64` (|ρ|=0.897).
- `score_edge_alignment` is the strongest standalone continuous-IoU predictor (Spearman=0.109).

---

## Short-cycle experiment queue (3-5 epochs each)

Each run uses `scripts/round2129_nwpu_posttrain_smoke.py` and targets ~25-40 minutes.

### Phase 1: Direction probing (3-5 epochs, first 3-4 hours)

#### Run 1: round2222_continuation_lr5e5_5ep
- **Base:** `round2214_lr1e4_30ep_clean/checkpoint_best.pth`
- **Changes:** lr=5e-5, adapter-lr=5e-5, predictor-lr=1.5e-5, cls-score-lr=7.5e-6, **epochs=5**.
- **Rationale:** Test whether lower LR can stabilize/improve from the best checkpoint without a long commitment.
- **Success criterion:** best AP75 > 0.3066.
- **If succeeds by ≥ +0.002:** run `round2222b_continuation_lr5e5_10ep` (scale to 10 epochs).
- **If mild gain (0.001-0.002):** try lr=2.5e-6, 5 epochs.
- **If flat/fails:** proceed to Run 2.

#### Run 2: round2223_rescue_weight_075_5ep
- **Base:** `round2214_lr1e4_30ep_clean/checkpoint_best.pth` (fresh start)
- **Changes:** `--rescue-loss-weight 0.075` (vs 0.05), lr=1e-4, **epochs=5**.
- **Rationale:** Fast check if stronger rescue weight improves learning dynamics.
- **Safety:** abort if `final_fp_rate` > 0.49 or `num_predictions` > 1450.
- **If succeeds:** run `round2223b_rescue_weight_100_5ep` next.
- **If fails:** proceed to Run 3.

#### Run 3: round2224_interpretable_fusion_verifier_ranking_5ep
- **Base:** `round2214_lr1e4_30ep_clean/checkpoint_best.pth`
- **Changes:** Use offline-winning interpretable fusion (`score_edge_alignment`, `boundary_phase_coherence`, `interior_exterior_texture_contrast`) as the verifier-ranking target/score, with `--verifier-ranking-loss-weight 0.003`, lr=1e-4, **epochs=5**.
- **Rationale:** Offline fusion has the best fixed-threshold transfer; expose it to the network as a ranking target.
- **Implementation note:** If adding the fusion score inside the training loop requires non-trivial code change, fall back to `round2224b_verifier_ranking_w003_from_best` (raw-iFFT only, from 2214 best).
- **If succeeds:** sweep weight 0.001 / 0.005.
- **If fails:** proceed to Run 4.

#### Run 4: round2225_trainable_adapter_predictor_5ep
- **Base:** `round2214_lr1e4_30ep_clean/checkpoint_best.pth`
- **Changes:** `--trainable-mode adapter_predictor` (unfreeze adapter + predictor), lr=1e-4, **epochs=3**.
- **Rationale:** Test if more trainable capacity helps, but keep it very short due to overfitting risk.
- **Safety:** stop if ECE rises > 0.095 or high-conf FP rate > 0.16.
- **If succeeds:** try with lr=5e-5 for 5 epochs.
- **If fails:** proceed to Run 5.

#### Run 5: round2226_kl_weight_05_5ep
- **Base:** `round2214_lr1e4_30ep_clean/checkpoint_best.pth`
- **Changes:** `--kl-weight 0.5` (vs 1.0), lr=1e-4, **epochs=5**.
- **Rationale:** Relax KL anchor to let policy explore better localization.
- **Safety:** stop if AP50 drops > 0.005.
- **If succeeds:** try kl-weight 0.25, 5 epochs.
- **If fails:** proceed to Run 6.

#### Run 6: round2227_rescue_high_iou_min_07_5ep
- **Base:** `round2214_lr1e4_30ep_clean/checkpoint_best.pth`
- **Changes:** `--rescue-high-iou-min 0.70` (vs 0.75), lr=1e-4, **epochs=5**.
- **Rationale:** Expand high-IoU rescue set; quick test of sample-definition sensitivity.
- **If succeeds:** combine with Run 2 weight increase for a 5-epoch follow-up.
- **If fails:** move to fallback.

---

## Phase 2: Scale-up (remaining 4-5 hours)

After Phase 1, pick the single best-performing direction and run longer:

- **Scale-up recipe:** 10 epochs at the winning hyperparameters.
- **If Phase 1 winner improved by ≥ 0.003:** run 15 epochs in Phase 2.
- **If Phase 1 winner improved by ≥ 0.005:** run 20 epochs in Phase 2.
- **If no Phase 1 run improves:** do a single conservative 10-epoch run at lr=5e-5 from 2214 best and consider rescuing a different detector baseline.

---

## Feedback loop rules

1. Record after every run into `runs/round2222_8h_iteration_summary.json`:
   - `run`, `best_ap75`, `final_ap75`, `ap50`, `fp_rate`, `high_conf_fp_rate`, `ece`, `num_predictions`, `best_epoch`, `delta_vs_2214`, `wall_time_min`.
2. Update this plan file in-place with results and the chosen next run.
3. **Promotion rule:** if a short run improves best AP75 by ≥ 0.0015, immediately run one deeper step on the same knob before moving to the next candidate.
4. **Dead-end rule:** if a direction fails twice in a row, abandon it.
5. **Safety rule:** any run with `fp_rate` > 0.49 or `high_conf_fp_rate` > 0.16 is disqualified from being the new best.

---

## First action

Launch **Run 1: round2222_continuation_lr5e5_5ep** now.
