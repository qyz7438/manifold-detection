# Manifold Detection

This repository is a research codebase for **energy-guided ROI transport in object detection**.  The current active question is whether a detector can learn small, verifiable local actions that move uncertain proposals toward valid detection states without flooding NMS with false positives.

The project evolved through RLVR-style score rescue, DPO proposal preferences, FFT/raw-iFFT verifier signals, and prototype manifold losses.  The current refactor keeps those lessons but changes the main framing: the class-conditioned target manifold is unknown, so transport must be modeled as an **action-local, low-energy, constrained policy** rather than as direct attraction to a known class prototype.

## Research Status

**No experiment is currently authorized.** Research status is controlled by the registries under `spectral_detection_posttrain/configs/registry/`; the human-readable summary is generated at `docs/research/current_status.md`.

## Maintained Implementation Areas

- **Energy-Guided ROI Transport**: predicts bounded feature, score, bbox, and keep/reject actions for ROI states, with low-energy, threshold-preservation, and rescue-budget constraints.
- **Proposal-Aligned Verifier Diagnostics**: FFT/raw-iFFT, geometry, prototype, and high-dimensional ROI signals remain useful as proposal rankers and controls.
- **Prototype Manifold Diagnostics**: `PrototypeBank`, `SinkhornAssigner`, and historical `TransportHead` modules remain available, but prototype attraction alone is no longer treated as the main objective.
- **Multi-dataset training/eval**: maintained runners support Penn-Fudan, Pascal VOC 2007, NWPU VHR-10, and COCO 2017 for clean evaluation.

Historical RLVR / DPO / score-rescue, AFM, FPN-SM, and channel-attention lines remain in the codebase as baselines or reproducibility artifacts.  They are not the default active claim unless explicitly requested.

## Repository Layout

```text
spectral_detection_posttrain/
  core/                 detector builders, matching, shared model primitives
  datasets/             Penn-Fudan / VOC / NWPU / COCO loaders
  eval/                 detection metrics and diagnostics
  methods/
    energy_transport/   action-local low-energy ROI transport primitives
    rlvr/               ROI policy losses, confidence rescue, detector verifiers
    dpo/                action verifier and preference-learning helpers
    manifold/           prototype banks, Sinkhorn assignment, transport heads,
                        FPN spectral manifold, FPN real adapter,
                        FPN attention baselines (SE/FcaNet/ECA)
  signals/
    fft/                FFT/raw-iFFT features and spectral rewards
    geometry/           spatial and box-geometry signals
  trainers/             detection and segmentation training entry points
  experiments/          canonical runner, schema, metadata, version records
  utils/                seed, io, checkpoint helpers
scripts/                maintained runners and analysis tools
  round28_train_eval.py           standard detector train/eval runner (not the energy-transport runner)
  run_voc_matrix_v2.sh            active VOC 20-class matrix (GPU2)
  run_voc_comparison_matrix.sh    SE/FcaNet/ECA comparison on VOC
  run_nwpu_matrix.sh              NWPU VHR-10 matrix
  run_nwpu_mob_480x800_matrix.sh  NWPU mobile baseline matrix
  run_coco_smoke.sh               COCO 2017 smoke test
  aggregate_results.py            summarize eval_metrics.json across runs
  eval_per_size.py                AP by small/medium/large objects
  profile_model.py                params/FLOPs/FPS/VRAM profiling
  summarize_round.py              round-level markdown summaries
docs/reports/           active experiment reports (see docs/reports/index.md)
docs/energy_guided_roi_transport.md
obsidian/               human-readable project notes
```

Compatibility shims keep historical imports working:

- `spectral_detection_posttrain/spectral/*` forwards to `signals/fft/*`
- `spectral_detection_posttrain/models/*` forwards to `core/models/*` or method modules
- `spectral_detection_posttrain/rlvr/*` forwards to `methods/rlvr/*`
- `spectral_detection_posttrain/train/*` forwards to `trainers/detection/*`

## Environment

### Local (Windows)

```powershell
conda activate RLimage
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/ -q
```

### Remote Server (`ps@122.51.19.136`)

```bash
cd /home/ps/lzz/manifold-detection-energy-transport
source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export CUDA_VISIBLE_DEVICES=2
python scripts/round28_train_eval.py --help
```

Only **GPU2** is available on the remote server. Do not stop or interfere with processes already running on GPU2.

## Quick Start

### Run a single Penn-Fudan baseline

```bash
python scripts/round28_train_eval.py \
  --dataset penn_fudan \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 12 --seed 42 --batch-size 8 --lr 0.005 \
  --run-name pf_baseline_s42
```

### Run with FPN spectral manifold

FPN-SM is now a baseline/diagnostic path, not the active transport claim.

```bash
python scripts/round28_train_eval.py \
  --dataset voc --voc-full \
  --model-name fasterrcnn_resnet50_fpn \
  --epochs 12 --seed 42 --batch-size 4 --lr 0.005 \
  --min-size 800 --max-size 1333 \
  --fpn-spectral-manifold \
  --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 \
  --fpn-sm-init-alpha 0.01 \
  --run-name voc_resnet_fpn_sm_s42_12ep
```

### Run a channel-attention baseline

```bash
python scripts/round28_train_eval.py \
  --dataset voc --voc-full \
  --model-name fasterrcnn_resnet50_fpn \
  --epochs 12 --seed 42 --batch-size 4 --lr 0.005 \
  --fpn-attention-type se --fpn-attention-reduction 16 \
  --run-name voc_resnet_se_s42_12ep
```

### Aggregate results across runs

```bash
python scripts/aggregate_results.py --runs runs/ --out docs/reports/aggregated_results.md
python scripts/eval_per_size.py --run-dir runs/<run_name>
python scripts/profile_model.py --run-dir runs/<run_name>
```

## Useful Checks

Focused smoke tests:

```bash
python -m pytest \
  tests/test_energy_guided_transport.py \
  tests/test_canonical_runner.py \
  tests/test_experiment_schema.py \
  tests/test_experiment_metadata.py \
  tests/test_manifold_modules.py \
  tests/methods/test_pbg.py -q
```

Full test suite (run before claiming validation):

```bash
python -m pytest -q
```

## Current Important Artifacts

- `docs/reports/top_tier_experiment_roadmap.md`: 顶刊路线图与实验矩阵计划
- `docs/reports/voc_matrix_plan.md`: VOC 20-class 矩阵设计
- `docs/reports/voc_matrix_interim_results.md`: VOC 矩阵中期结果
- `docs/reports/nwpu_matrix_analysis.md` / `nwpu_matrix_summary.txt`: NWPU 矩阵分析
- `docs/reports/nwpu_mob_480x800_analysis.md` / `nwpu_mob_480x800_summary.txt`: NWPU mobile 480×800 矩阵
- `docs/reports/index.md`: active report index
- `obsidian/RLIimage Map.md`: human-readable project map

## Reproducibility Rules

- Use canonical experiment metadata for new clean runs.
- Record config hash, checkpoint hash, git commit, and dirty status.
- Treat `runs/`, `data/`, local agent state, and generated analysis caches as local artifacts.
- Promote an experiment to validated only after full clean eval and a fixed commit.
- Keep active run scripts and configs in `scripts/`; archive obsolete artifacts to `legacy/`.
