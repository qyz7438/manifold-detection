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

%PYTHON% -m spectral_detection_posttrain.train.train_baseline --config spectral_detection_posttrain/configs/mvp.yaml --run-name mvp_pf_baseline --epochs 1
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.eval.eval_spectral_reward --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --run-name mvp_pf_reward_baseline
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.train.posttrain_reward_weighted --config spectral_detection_posttrain/configs/mvp.yaml --baseline runs/mvp_pf_baseline/checkpoint_last.pth --run-name mvp_pf_posttrain --epochs 5
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --run-name mvp_pf_eval_baseline_clean --patch-mode none
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_eval_post_clean --patch-mode none
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_eval_post_random_patch --patch-mode random --patch-type random
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_eval_post_checker_patch --patch-mode random --patch-type checkerboard
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.eval.eval_spectral_reward --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_reward_posttrain
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.visualization.visualize_roi_fft --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_roi_fft
if errorlevel 1 pause & exit /b 1

echo MVP detection experiment finished. Check runs\mvp_pf_* and docs\mvp_detection_results_2026-06-03.md.
pause
