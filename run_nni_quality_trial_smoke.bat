@echo off
setlocal
cd /d %~dp0
set PYTHON=E:\anaconda\01\envs\RLimage\python.exe
set TEMP=E:\tmp
set TMP=E:\tmp
set MPLCONFIGDIR=E:\tmp\matplotlib
set TORCH_HOME=E:\tmp\torch
if not exist E:\tmp mkdir E:\tmp

set PARAMS_FILE=E:\tmp\nni_quality_smoke_params.json
(
  echo {
  echo   "detector_epochs": 1,
  echo   "quality_head": "ROI+Amp+Struct",
  echo   "qh_epochs": 8,
  echo   "alpha": 0.9
  echo }
) > %PARAMS_FILE%

%PYTHON% -m spectral_detection_posttrain.nni_quality_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_quality_smoke --fixed-recall 0.85 --limit-train 4 --limit-val 4 --early-stopping-patience 2 --params-file %PARAMS_FILE%
if errorlevel 1 pause & exit /b 1
echo NNI trial smoke finished. Check runs\nni_quality_smoke\last_trial_result.json.
pause
