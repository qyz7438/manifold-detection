@echo off
setlocal enabledelayedexpansion

cd /d E:\CLIproject\RLimage
set PYTHON=E:\anaconda\01\envs\RLimage\python.exe
set CONFIG=spectral_detection_posttrain\configs\manifold_nwpu.yaml
set COMMON=--config %CONFIG% --baseline runs\nwpu_baseline_best.pth --lr 1e-5 --lr-manifold 1e-4 --lambda-tr 0.01 --lambda-en 0.001 --epochs 5 --early-stopping-patience 5

%PYTHON% -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain %COMMON% --run-name nwpu_head_original_e5 --box-head-type original
if errorlevel 1 exit /b 1

%PYTHON% -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain %COMMON% --run-name nwpu_head_bottleneck_r128_e5 --box-head-type bottleneck --box-head-rank 128
if errorlevel 1 exit /b 1

%PYTHON% -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain %COMMON% --run-name nwpu_head_bottlenecktwomlp_c64_e5 --box-head-type bottleneck_twomlp --box-head-conv-channels 64
if errorlevel 1 exit /b 1

%PYTHON% -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain %COMMON% --run-name nwpu_head_convlowdim_c128_e5 --box-head-type conv_lowdim --box-head-conv-channels 128
if errorlevel 1 exit /b 1

%PYTHON% -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain %COMMON% --run-name nwpu_head_attentionpool_e5 --box-head-type attention_pool --box-head-attention-channels 64
if errorlevel 1 exit /b 1

echo All box head ablations completed.
