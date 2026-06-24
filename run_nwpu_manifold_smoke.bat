@echo off
setlocal

if "%PYTHON%"=="" set "PYTHON=python"

"%PYTHON%" -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain ^
  --config spectral_detection_posttrain/configs/manifold_nwpu.yaml ^
  --baseline runs/round2100_nwpu_baseline/checkpoint_best.pth ^
  --run-name nwpu_mglopt_active_smoke ^
  --limit-train 4 ^
  --limit-val 4 ^
  --epochs 1 ^
  --warmup-batches 1 ^
  --num-prototypes 4 ^
  --lambda-tr 0.01 ^
  --lambda-en 0.001 ^
  --lr 0.00001 ^
  --lr-manifold 0.0001 ^
  --active-manifold-correction ^
  --active-correction-gamma 0.05 ^
  --geometry-every 1 ^
  --eval-every 1 ^
  --early-stopping-patience 2

endlocal
