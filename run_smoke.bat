@echo off
setlocal
cd /d %~dp0
set PYTHON=E:\anaconda\01\envs\RLimage\python.exe
set TEMP=E:\tmp
set TMP=E:\tmp
set MPLCONFIGDIR=E:\tmp\matplotlib
set TORCH_HOME=E:\tmp\torch
if not exist E:\tmp mkdir E:\tmp

%PYTHON% -m pytest tests -q
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.train.train_baseline --config spectral_detection_posttrain/configs/smoke.yaml --run-name det_smoke_baseline --limit-train 8 --limit-val 4 --epochs 1
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_spectral_reward --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_baseline/checkpoint_last.pth --run-name det_smoke_reward --limit-val 4
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.train.posttrain_reward_weighted --config spectral_detection_posttrain/configs/smoke.yaml --baseline runs/det_smoke_baseline/checkpoint_last.pth --run-name det_smoke_posttrain --limit-train 4 --limit-val 4 --epochs 1
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_posttrain/checkpoint_last.pth --run-name det_smoke_eval_clean --limit-val 4 --patch-mode none
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_posttrain/checkpoint_last.pth --run-name det_smoke_eval_patch --limit-val 4 --patch-mode random
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.visualization.visualize_roi_fft --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/det_smoke_posttrain/checkpoint_last.pth --run-name det_smoke_roi_fft --limit-val 4
if errorlevel 1 pause & exit /b 1
echo Detection smoke run finished. Check runs\det_smoke_*.
pause
