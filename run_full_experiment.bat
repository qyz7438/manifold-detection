@echo off
setlocal
cd /d %~dp0
set PYTHON=E:\anaconda\01\envs\RLimage\python.exe
set TEMP=E:\tmp
set TMP=E:\tmp
set MPLCONFIGDIR=E:\tmp\matplotlib
set TORCH_HOME=E:\tmp\torch
if not exist E:\tmp mkdir E:\tmp

%PYTHON% -m spectral_detection_posttrain.train.train_baseline --config spectral_detection_posttrain/configs/baseline.yaml --run-name det_baseline
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_spectral_reward --config spectral_detection_posttrain/configs/baseline.yaml --checkpoint runs/det_baseline/checkpoint_last.pth --run-name det_reward_diagnostic
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.train.posttrain_reward_weighted --config spectral_detection_posttrain/configs/posttrain.yaml --baseline runs/det_baseline/checkpoint_last.pth --run-name det_reward_posttrain
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_baseline/checkpoint_last.pth --run-name det_eval_baseline_clean --patch-mode none
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_baseline/checkpoint_last.pth --run-name det_eval_baseline_random_patch --patch-mode random --patch-type random
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_baseline/checkpoint_last.pth --run-name det_eval_baseline_checker_patch --patch-mode random --patch-type checkerboard
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_reward_posttrain/checkpoint_last.pth --run-name det_eval_posttrain_clean --patch-mode none
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_reward_posttrain/checkpoint_last.pth --run-name det_eval_posttrain_random_patch --patch-mode random --patch-type random
%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_reward_posttrain/checkpoint_last.pth --run-name det_eval_posttrain_checker_patch --patch-mode random --patch-type checkerboard
%PYTHON% -m spectral_detection_posttrain.eval.eval_spectral_reward --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_reward_posttrain/checkpoint_last.pth --run-name det_reward_posttrain_diagnostic
%PYTHON% -m spectral_detection_posttrain.visualization.visualize_roi_fft --config spectral_detection_posttrain/configs/posttrain.yaml --checkpoint runs/det_reward_posttrain/checkpoint_last.pth --run-name det_roi_fft
echo Full detection experiment finished. Check runs\det_*.
pause
