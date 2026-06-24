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

if not exist runs\mvp_pf_baseline\checkpoint_last.pth (
  %PYTHON% -m spectral_detection_posttrain.train.train_baseline --config spectral_detection_posttrain/configs/mvp.yaml --run-name mvp_pf_baseline --epochs 1
  if errorlevel 1 pause & exit /b 1
)

%PYTHON% -m spectral_detection_posttrain.spectral.roi_spectral_dataset --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --split train --run-name mvp_qh_candidates_train_clean --output runs/mvp_qh_candidates_train_clean/candidates.pt
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.spectral.roi_spectral_dataset --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --split val --run-name mvp_qh_candidates_val_clean --output runs/mvp_qh_candidates_val_clean/candidates.pt
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.spectral.roi_spectral_dataset --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --split val --run-name mvp_qh_candidates_val_random --output runs/mvp_qh_candidates_val_random/candidates.pt --patch-mode random --patch-type random
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.spectral.roi_spectral_dataset --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --split val --run-name mvp_qh_candidates_val_checker --output runs/mvp_qh_candidates_val_checker/candidates.pt --patch-mode random --patch-type checkerboard
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.train.train_quality_head --config spectral_detection_posttrain/configs/mvp.yaml --train-candidates runs/mvp_qh_candidates_train_clean/candidates.pt --val-candidates runs/mvp_qh_candidates_val_clean/candidates.pt --run-name mvp_qh_roi --feature-mode roi
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.train.train_quality_head --config spectral_detection_posttrain/configs/mvp.yaml --train-candidates runs/mvp_qh_candidates_train_clean/candidates.pt --val-candidates runs/mvp_qh_candidates_val_clean/candidates.pt --run-name mvp_qh_roi_amp_structure --feature-mode roi_amp_structure
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --run-name mvp_qh_eval_baseline_clean --method baseline
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_random/candidates.pt --run-name mvp_qh_eval_baseline_random --method baseline
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_checker/candidates.pt --run-name mvp_qh_eval_baseline_checker --method baseline
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --normalization-cache runs/mvp_qh_candidates_train_clean/candidates.pt --run-name mvp_qh_eval_oracle_clean --method oracle_ramp --combine blend --alpha 0.7
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_random/candidates.pt --normalization-cache runs/mvp_qh_candidates_train_clean/candidates.pt --run-name mvp_qh_eval_oracle_random --method oracle_ramp --combine blend --alpha 0.7
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_checker/candidates.pt --normalization-cache runs/mvp_qh_candidates_train_clean/candidates.pt --run-name mvp_qh_eval_oracle_checker --method oracle_ramp --combine blend --alpha 0.7
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --quality-checkpoint runs/mvp_qh_roi/quality_head_last.pth --run-name mvp_qh_eval_roi_clean --method learned --combine blend --alpha 0.7
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name mvp_qh_eval_full_clean --method learned --combine blend --alpha 0.7
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_random/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name mvp_qh_eval_full_random --method learned --combine blend --alpha 0.7
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_checker/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name mvp_qh_eval_full_checker --method learned --combine blend --alpha 0.7
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name mvp_qh_eval_full_clean_alpha09 --method learned --combine blend --alpha 0.9
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_random/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name mvp_qh_eval_full_random_alpha09 --method learned --combine blend --alpha 0.9
if errorlevel 1 pause & exit /b 1
%PYTHON% -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_checker/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name mvp_qh_eval_full_checker_alpha09 --method learned --combine blend --alpha 0.9
if errorlevel 1 pause & exit /b 1

%PYTHON% -m spectral_detection_posttrain.visualization.visualize_spectral_quality --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name mvp_qh_visual_quality
if errorlevel 1 pause & exit /b 1

echo Spectral Quality Head MVP finished. Check runs\mvp_qh_*.
pause
