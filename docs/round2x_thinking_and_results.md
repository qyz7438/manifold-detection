# 2.x Series Thinking And Results

Date: 2026-06-16

This document consolidates the 2.x experimental line from existing reports, plans, result JSON files, and runner names. It is a research log, not a new experiment. The goal is to make the sequence searchable and to separate verified results from plans and bug-audited dead ends.

## Evidence Tags

- verified: supported by an existing report and/or runs/*/eval_metrics.json.
- partial: some results exist, but the round is incomplete or not enough for a claim.
- audit-invalid: later audit found implementation issues; keep for lesson value, not as direct evidence.
- planned: design exists, but no reliable completed result was found in this pass.

## Executive Summary

The 2.x series has three major arcs.

First, 2.1-2.5 built a stable RLVR shell for detection. The stable recipe was KL=10 + policy=0.0003 + signed objective + frozen baseline + freeze-state control. It preserved AP50 and prediction count, but hand-built spectral rewards did not separate real from shuffled controls.

Second, 2.6-2.18 moved FFT inside the detector as AFM. The old learnable-scale AFM was partly dead because the FFT path had zero gradient. MPLSeg-style active amplitude/phase AFM fixed that topology and made in-network FFT useful for localization, with mid06 becoming the best candidate on Penn-Fudan.

Third, 2.21 onward tried freeze/data/phase/recipe ablations, then many RL/GRPO/DPO variants. The strongest durable result remained: phase/in-network AFM can help AP75 in the Penn-Fudan Faster R-CNN setup, while external verifier rewards remain weak or indistinguishable from controls.

## 2.1-2.5: Stable RLVR Shell, Weak Spectral Causality

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.1 | verified | Fix early engineering errors: post-NMS boxes used for training, shuffled control not actually active, temperature not propagated, z-score treated FP zeros badly, score thresholds desynced. | Conservative cls-only weighted CE still collapsed: clean AP50 about 0.623, precision about 0.21, prediction count about 300. | Positive weighted CE was not the right RLVR objective. |
| 2.2 | verified | Replace positive weighted CE with signed policy objective: high reward should increase action probability, low reward should decrease it. Add frozen baseline and KL. | Still had score distribution drift and AP75/precision issues. | The objective direction was better, but stability needed freeze-state and logging repair. |
| 2.3 | verified | Make the shell readable and stable: result schema, no-update path, BN/frozen eval state, initial ROI KL sanity, frozen baseline rollout. | signed_iou_0003_kl10 clean AP50 0.8726, AP75 0.6582, recall 0.8901, prediction count 125; baseline/null no-update AP50 about 0.8723, prediction count 124. | First stable RLVR shell. This is the useful engineering asset from external-verifier RLVR. |
| 2.4 | partial | Fill missing 2.3 result gaps before making R_amp causality claims. | Planned around repairing rows and missing evaluations. | Valuable as result hygiene; not a separate method result. |
| 2.5 | verified | Add structure/phase-inspired verifier components and real-vs-shuffled controls under the stable shell. | Stable groups clustered at AP50 0.884-0.886, prediction count 129-131, recall 0.9011. signed_amp AP75 0.6640 vs null 0.6435, but shuffled/structure controls were similar. | Hand-built spectral verifier had no causal separation on Penn-Fudan. Real vs shuffled gap was too small to claim success. |

## 2.6-2.12: From External Reward To In-Network FFT

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.6 | verified | Insert MicroAFM before ROI box head to test FFT as network module. | Baseline AP50 0.8850/AP75 0.6451; AFM full AP50 0.8527/AP75 0.4093, precision down to 0.4385, high-conf FP up to 14. | Naive AFM disturbed pretrained ROI feature statistics. |
| 2.7 | verified | Make AFM identity-preserving with residual/no-op initialization. | AP50 about 0.8761 vs baseline 0.8770, but AP75 dropped from 0.6524 to 0.5367 and prediction count rose. | Identity preserved AP50 but did not fix localization/score distribution. |
| 2.8 | verified | Diagnose AFM with frozen parity, scale tracking, localization stats, and AFM-only/box-head controls. | identity_current_afm_box_head AP50 0.8653/AP75 0.7378/ECE 0.0283; identity_delta_afm_box_head AP75 0.7374. But mag_scale=0, phase_scale=0; AFM-only AP50 0.0515. | Apparent AP75 gain came from residual/head adaptation, not active FFT gate learning. |
| 2.9 | partial | Ask whether AFM scales leave zero over more epochs, whether RPN can learn edge-patch robustness, and whether post-training from checkpoint is stable. | Some groups existed in runs/round29*; best AP75 around 0.7374 and RPN mixed AP50 around 0.8704. | Useful fairness/checkpoint sanity, but not enough to prove frequency causality. |
| 2.10 | verified | Fix checkpoint loading and edge-mix eval so post-training sanity is trustworthy. | round210_b1_ckpt_eval AP50 0.8198/AP75 0.6100; round210_b2_posttrain AP50 0.8149/AP75 0.5252, precision improved and ECE improved; round210_g9_rpn_mixed AP50 0.8707/AP75 0.6295. | Post-training and edge-mix can run stably, but improvements looked like supervised/RPN/head adaptation, not spectral reward. |
| 2.11 | verified | Move to VOC 3-class to test whether Penn-Fudan was too simple for spectral signal. | Baseline AP50 0.7084/AP75 0.3366. Detection-only posttrain AP50 0.7734. Spatial+spectral loggate AP50 0.7724/AP75 0.3751, while shuffled spectral AP50 0.7742/AP75 0.3765. | More complex VOC did not rescue hand-built spectral verifier; shuffled was slightly better. |
| 2.12 | verified | Fix the AFM gradient topology by replacing old scale-gated AFM with MPLSeg-style active gate/phase residual. | MPLSeg AFM AP50 0.8678/AP75 0.6534/ECE 0.0666; residual scale around 0.9720. Unit tests showed mag/phase/residual gradients. | In-network FFT path can be active and trainable. This does not validate external spectral reward. |

## 2.13-2.20: AFM Stability, Gate Strength, Post-Training Structure

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.13 | verified | Multi-seed compare baseline, identity AFM, and MPLSeg AFM. | Best round213_mplseg_s456: AP50 0.9432/AP75 0.7424; results had large seed variance. | MPLSeg-style AFM looked promising but required deterministic/matched reruns. |
| 2.14 | partial | Frozen/trained/no-tune MPLSeg controls. | Frozen AP75 0.6152, trained AP75 0.4992, no-tune AP75 0.3780. | Controls suggested architecture/initialization mattered, but single-seed style limited claim strength. |
| 2.15 | verified | Gate strength sweep: weak 0.3, mid 0.6, strong 1.0. | Weak AP50 0.8698/ECE 0.0473, mid AP75 0.6288, strong weaker AP75 0.5462. | Mid gate favored AP75; weak gate favored calibration. Strong gate over-suppressed. |
| 2.16 | verified | Deterministic rerun across seeds for gate strengths and controls. | AGENTS summary records mid06 as best: AP75 +12.7%, P@R=0.85 +2.8%, ECE -25.8%, predictions -17% vs baseline. Runs show round216p_mid06_s123 AP75 0.7944. | mplseg_mid/mid06 became the best detector AFM candidate. |
| 2.18 | verified | Validate post-training structures A/B/C from mid06. | A weak-gate AFM-only seed42 AP50 0.865/AP75 0.690; C feature-constraint AFM-only seed42 AP50 0.864/AP75 0.701; bbox correction head prototype not successful. | A and C are viable post-training recipes; C looked better for AP75. |
| 2.19 | partial | Cross-dataset/backbone small validation: PF+ResNet50 and VOC+MobV3. | Runs show high variance and probable backbone/config instability. AGENTS notes ResNet50 afm_channels crash remained to fix. | Generalization not settled; runner/schema hardening needed before using this as evidence. |
| 2.20 | partial | Follow-on cross-backbone/dataset matrix. | Runs exist with large AP50/AP75 spread, especially ResNet50 PF. | Treat as exploratory only. |

## 2.21-2.26: Focused AFM Ablations

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.21 | verified | Which components should be frozen during AFM post-training? | Five groups. freeze_rpn best AP50 0.8561; freeze_box best AP75 0.6696. | Freezing selected components helps post-training stability. |
| 2.22 | verified | Does post-training benefit depend on data volume? | Eight groups. A post-training AP75 stayed around 0.64-0.66 across data sizes; baseline fluctuated more, about 0.52-0.62. | Post-training provided data-efficiency/stability benefit, though still PF-specific. |
| 2.23 | partial | Visualize gate frequency response from mid06 checkpoint. | AGENTS says analysis script had a bug: inference hook did not trigger. | Result invalid until hook is fixed. |
| 2.24 | planned | Compare FFT against DCT/Wavelet transforms. | Roadmap exists; no reliable result found in this pass. | Still open. |
| 2.25 | verified | Test whether magnitude or phase branch drives AFM. | Nine groups. Phase-only had best single-seed AP50 0.9618 and best AP75 0.7826 on seed123. | Phase modulation is the critical AFM mechanism; magnitude-only is not the main source. |
| 2.26 | verified | Sweep post-training recipes A, C, AC and epoch counts from mid06 checkpoint. | Eighteen groups. C_5ep best AP75 0.6740; results were deterministic across seeds because same checkpoint/deterministic setup produced identical outputs. | Feature constraint C is the best post-training recipe among this set. |

## 2.27-2.36: Convergence And RL Post-Training Reopen

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.27 | partial | Establish longer baseline convergence so post-training claims are not compared to an undertrained baseline. | round227_v1_baseline_20ep AP50 0.8381/AP75 0.6569; smoke AP50 0.8126/AP75 0.6113. | Useful sanity, but not central. |
| 2.28 | partial | Frozen-control style follow-up. | Three groups, best AP50 0.9786 and AP75 0.8603. | High numbers need strict provenance checking before claim use. |
| 2.29 | partial | AFM refine. | Best AP50 0.8519/AP75 0.6395. | Did not clearly exceed earlier mid06/C results. |
| 2.30 | partial | AFM constraint. | Best AP50 0.8465/AP75 0.6395. | Did not become a new best. |
| 2.31 | audit-invalid | Per-pixel reward / soft reweighting style. | Runs show some AP75 around 0.6728, but bug audit warns 2.31 may have used incompatible torchvision loss format and direct claim is unsafe. | Keep as historical idea; do not cite as validated. |
| 2.32 | partial | Frequency consistency. | Nine groups; best AP75 0.6917. | Moderate result, no decisive advantage over det-only/AFM baselines. |
| 2.33 | partial | RLVR posttraining attempt. | Two groups; AP50 about 0.863-0.8725, AP75 about 0.606-0.6085. | No strong evidence of RLVR benefit. |
| 2.34 | partial | Bbox REINFORCE. | AP50 0.8638/AP75 0.6756. | Comparable to baseline-ish runs; not decisive. |
| 2.35 | partial | E2E PPO. | AP50 about 0.463, AP75 about 0.198. | PPO-style route collapsed. |
| 2.36 | partial | VOC/full RLVR direction. | Mixed: best AP50 0.8638/AP75 0.6756, but one group AP50 0.4358. | Unstable; not enough for a positive claim. |

## 2.38-2.58: DPO/RFT And Bug-Audited Dead Ends

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.38-2.40 | partial | Proposal refiner / spectral weighting / FPN quality variants. | AP50 ranged roughly 0.851-0.887; AP75 up to 0.6968. | Exploratory and not clearly superior. |
| 2.41 | audit-invalid | Pixel-level RFT. | AP50 about 0.861, AP75 up to 0.6912. | Later audit: switched away from working ROI feature FFT to pixel FFT and had DPO/RFT plumbing risks. |
| 2.43 | audit-invalid | IoU REINFORCE. | AP50 0.8638/AP75 0.6756. | Historical control; not evidence of frequency reward. |
| 2.44-2.46 | audit-invalid | Edge-RFT, inverted pixel soft, edge centrality. | AP75 roughly 0.662-0.692. | Audit says same structural issues as 2.41 family. |
| 2.49 | audit-invalid | Learned spectral head. | AP75 up to 0.6920. | Audit warns ROI feature use and signal definition were not reliable. |
| 2.50-2.51 | audit-invalid | X-DPO / combined DPO. | AP75 ranged 0.5448-0.6625. | Critical DPO reference gradient/sigma/q_radial issues in audit. |
| 2.52/2.55/2.57/2.58 | audit-invalid or partial | Bug-fixed DPO-ROI, pixel threshold, native zero-pad, edge truncation. | Many runs repeated AP50 0.8629/AP75 0.6625. Audit lists some fixes but remaining limitations. | Ready-to-run candidates after more fixes, but not validated evidence. |

## 2.59-2.71: Frequency Probes And Verifier Variants

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.59-2.60 | partial | FFT-edge and frequency-probe variants. | Runs largely repeated AP50 0.8629/AP75 0.6625. | No clear gain. |
| 2.61 | partial | IoU GRPO. | Best AP75 0.6786. | Slight but not decisive. |
| 2.62-2.64 | partial | ROI verifier, FFT14, shuffled FFT14. | 2.62 best AP75 0.7153; 2.63/2.64 best AP75 around 0.701. Shuffled controls similar. | Again, real-vs-control separation was weak. |
| 2.65-2.67 | partial | FFT-only, FFT28, per-channel FFT. | Best AP75 around 0.699-0.7119. | Per-channel features looked competitive but not causally distinct. |
| 2.68-2.71 | partial | Per-channel FFT, shuffled/band/q-only comparisons. | 2.71 best AP75 0.7446 for qonly_band_s42; AP50 best from shuffled q-only. | Promising diagnostic signal but still entangled with controls. |

## 2.72-2.90: GRPO, Energy, Geometry, Discrete Actions

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.72-2.74 | partial | Larger GRPO/reward variants, NMS-aware/AP75 event/local IoU. | Best AP75 around 0.7216 in 2.72 and 0.7153 in 2.74. | Useful controls, but det-only often remained competitive. |
| 2.75 | partial | Split localization vs selection rewards, FFT loc-only, early stop. | Best AP75 0.7117. | Did not beat det-only clearly. |
| 2.76 | partial | GRPO + KL anchor minimal decision experiment. | Best AP75 0.7267 for GRPO FFT on seed123; det-only/kl baselines close. | GRPO can be stable, but advantage over supervised continuation was small. |
| 2.77 | partial | Increase group size to G=8 for GRPO stability. | Best AP75 0.7030. | G=8 did not unlock a large gain. |
| 2.78 | partial | Direct FFT reward with energy/similarity/phase consistency. | Best AP75 0.6987. | Direct FFT reward remained weak. |
| 2.79 | partial | Unfreeze FPN/RPN/box_head to test whether representation layers help RL signal. | Best AP75 0.7192; det-only unfrozen was also strong. | More trainable capacity helped stability but not spectral causality. |
| 2.80 | partial | Unified 20-group comparison across RLVR, hybrid verifiers, AFM variants. | 63 eval rows. Best AP75 0.7369 from det-only unfrozen; per-channel FFT AP75 up to 0.7364 but AP50 lower. | Best result did not require spectral reward. |
| 2.81 | partial | Energy residual verifier: penalize high-energy proposals after FN/TP energy analysis. | Energy residual AP50 about 0.837-0.838/AP75 about 0.681-0.690, below det-only. | Global energy penalty created bad incentives. |
| 2.82-2.84 | partial | C-gated and nonlinear/asymmetric energy penalties. | Best AP75 around 0.7164 in shuffled/gated controls. | Gating improved safety, but controls still matched real signal. |
| 2.85 | partial | Cross-proposal GRPO so energy varies across proposals, not tiny within one proposal. | AP75 max about 0.6854. | Did not become stronger than earlier baselines. |
| 2.86 | partial | Geometry reward from area/cx/cy/aspect after high AUC diagnostic. | Best AP75 0.6695. | Geometry diagnostic did not translate into detector improvement. |
| 2.87 | partial | Energy-weighted detection loss instead of policy gradient. | Best AP75 0.7162. | Weighting is safer than PG, but not a major new best. |
| 2.88 | partial | Discrete action GRPO for bbox refinement. | Best AP75 0.7157, matching det-only. | Discrete actions fixed variance logic but not final performance. |
| 2.89 | partial | Multi-step feature-driven bbox refinement. | Best AP75 0.7157 from det-only; many action methods worse. | Multi-step dynamics did not unlock clear benefit. |
| 2.90 | partial | Raw image energy plus larger sigma so energy variance dominates IoU. | Raw energy/shuffle AP50 around 0.873-0.876; AP75 0.691-0.704. Det-only AP75 0.7157. | Stronger raw energy signal still failed to beat det-only. |

## 2.91-2.97: Cross-Dataset And Action Preference

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.91 | partial | NWPU VHR-10 baseline. | round2100_nwpu_baseline AP50 0.6483/AP75 0.2966. | Established harder aerial-detection baseline. |
| 2.92 | partial | NWPU PG/RLVR on converged baseline. | RLVR AP50 often below det-only; one seed collapsed to zero in earlier NWPU run. | PG did not improve NWPU baseline. |
| 2.93 | partial | NWPU AFM mid06 test. | Result files exist outside this summary pass, but no strong consolidated claim found. | Needs dedicated NWPU summary before citing. |
| 2.95 | partial | VisDrone baseline/resume on dense small objects. | Runner exists; no clean consolidated metrics included here. | Dataset expansion started but is not yet a 2.x conclusion. |
| 2.97b/2.97c | partial | Action-preference DPO reproducibility and quick optimizer sweep. | Writes .agent_reports/action_pref/*; intended to resolve DPO vs supervised MLP contradiction. | Separate offline preference-learning line; not direct detector AP evidence yet. |

## 2.100-2.119: Later RLVR/DPO Reopen

The numbering here appears as round2100 etc. in files, corresponding to later 2.x extensions rather than Plan 2100 in the original roadmap.

| Round | Tag | Thinking | Result | Conclusion |
|---|---|---|---|---|
| 2.100 | verified | NWPU baseline. | AP50 0.6483/AP75 0.2966. | Harder benchmark than Penn-Fudan. |
| 2.101 | partial | NWPU RLVR IoU cls. | Det-only AP50 about 0.632-0.643; RLVR one seed collapsed to 0, others AP50 0.635. | RLVR unstable or non-improving on NWPU. |
| 2.102 | partial | Return to PF with det-only vs RLVR IoU cls. | RLVR seed123 AP75 0.7120 vs det-only seed123 0.6578, but seed42 AP50 dropped to 0.8471. | Some AP75 gains, not robust enough. |
| 2.103 | partial | Geometry cls reward. | Geometry seed42 AP75 0.6897 vs det-only 0.6851. | Marginal effect. |
| 2.104-2.105 | partial | NMS/discrete RLVR. | Metrics nearly identical to det-only; best AP75 about 0.7022. | No clear improvement over supervised continuation. |
| 2.106 | partial | Large sweep with AP50-only style outputs. | Many AP75 fields are zero/missing, AP50 toggles around 0.8446/0.8553. | Logging incomplete; not comparable. |
| 2.107/2.109 | partial | NWPU discrete/DPO. | AP50 about 0.648, AP75 about 0.295. | No NWPU improvement. |
| 2.108 | partial | DPO on PF. | DPO seed42 AP75 0.7087 vs det-only seed42 0.7022, but other seeds similar. | Small, inconsistent gain. |
| 2.110 | partial | FFT process. | FFT process AP75 0.675-0.689 vs det-only 0.691-0.702. | FFT process underperformed det-only. |
| 2.111 | partial | RLVR on harder/NWPU-like setting. | RLVR AP50 0.421-0.448 vs det-only around 0.651. | Collapse. |
| 2.112-2.113 | partial | Manifold DPO / disagreement / percentile DPO. | Manifold DPO under det-only; disagreement DPO AP50 0.8768 but AP75 0.6875 vs det-only AP75 0.7019; percentile DPO AP75 0.6923. | Preference/DPO variants did not beat AP75 baseline. |
| 2.115-2.119 | partial | Edge DPO/RLVR, TPNN RLVR/recalib/DPO. | Edge/TPNN variants generally AP50/AP75 below det-only; tpnn_dpo AP50 0.8739 but AP75 0.6869 vs det-only 0.6995. | Later RL/DPO reopen did not overturn the earlier conclusion. |

## Durable Lessons

1. The stable ROI-policy shell is a real engineering asset, but it needs a stronger verifier than hand-built ROI FFT summaries.
2. External spectral reward repeatedly failed the real-vs-shuffled test.
3. In-network FFT is different: once the gradient topology was fixed, MPLSeg-style AFM improved localization in the Penn-Fudan two-stage detector setup.
4. Phase is more important than magnitude gate in the current AFM implementation.
5. Many later RL/GRPO/DPO variants produced small AP75 changes, but det-only continuation and shuffled controls often matched or beat them.
6. The project needs a canonical runner and stronger config/metadata validation before further large matrices.

## 2026-06-17: NWPU Manifold Rescue Notes

Current NWPU confidence-rescue manifold gate is a non-parametric verifier, not a trained manifold model. It builds TP/FP reference banks from baseline ROI features on the train split, then scores proposals by class/scale-aware kNN density ratio. The detector/adapters are trained; the manifold reference itself has no SGD-updated parameters.

The full-train A density-ratio run (`round2130_rescue_A_density_fulltrain_5ep`) did not improve final AP75:

| Run | Baseline AP75 | Final AP75 | Best Epoch AP75 | FP Rate Change | ECE Change |
|---|---:|---:|---:|---:|---:|
| round2130_rescue_A_density_fulltrain_5ep | 0.293908 | 0.293603 | 0.296559 at epoch 2 | +0.015621 | +0.018637 |

Main failure mode: the rescue objective is heavily positive-skewed. Each epoch saw about 1,940 rescue positives but only 10-15 rescue negatives, so the model learned to lift low-confidence candidates without enough hard-negative pressure. Prediction count and FP rate increased, and calibration worsened.

Optimization ideas to keep:

1. Learnable manifold verifier: replace pure kNN reference scoring with an ROI-feature projection head trained by supervised contrastive/triplet/BCE objectives. The target should explicitly pull LC-HI proposals toward TP anchors and push LC-LI/HC-LI proposals away from the TP manifold. This tests whether the baseline ROI feature geometry is the bottleneck.
2. Hard negative mining: add `low confidence + low IoU + high verifier score` proposals as negative rescue targets. These are the false-rescue cases that the current objective does not punish.
3. Best-checkpoint and safety selection: save `checkpoint_best.pth` by AP75 or a composite metric, not only `checkpoint_last.pth`. Reject best updates when prediction count, FP rate, high-confidence FP rate, or ECE exceed the baseline guardrail.
4. More conservative rescue weights: reduce rescue strength and policy pressure while increasing negative pressure, e.g. lower LR, lower rescue loss, higher negative weight, and higher supervised detection keep-loss.
5. Gate calibration after stability, not before: class-wise threshold calibration should be used only after the base density-ratio objective is stable. It optimizes coverage/precision but cannot fix an unsafe loss.
6. Two-stage verifier flow: train/evaluate the verifier offline first with AUC, LC-HI recall at precision targets, and false-rescue rate. Only attach it to post-training if the offline verifier beats simple score/IoU/geometry baselines and passes shuffled controls.
7. Feature-source ablation: compare ROI box-head features, FPN pooled features, geometry features, FFT ROI-crop features, and AFM-enhanced features for the same manifold verifier. The current kNN bank may be limited by using baseline ROI features only.
8. Dynamic reference refresh: either freeze the verifier/reference intentionally and log it, or periodically rebuild reference features from the current detector. A frozen baseline manifold can become stale after adapters change the feature/logit distribution.
9. Ranking loss instead of only BCE rescue: train on pairs within the same image/class where one candidate is LC-HI and another is LC-LI/HC-LI. Pairwise ranking may be safer than independently lifting all gated positives.
10. Inference-path alignment: if the training objective changes confidence, measure NMS survivor changes and AP75-localization changes directly. The verifier should improve final detector ranking, not only intermediate proposal scores.

## Pointers

- Main report: docs/reports/rlimage_full_experiment_report_2026-06-04.md
- AFM diagnostics: docs/round28_results.md
- VOC 2.11 results: docs/round211_results.md
- DPO/RFT bug audits: docs/bug_audit_2.41_2.56.md, docs/bug_audit_full_2.41-2.58.md
- Current project memory: AGENTS.md
