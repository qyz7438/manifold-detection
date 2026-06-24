@echo off
setlocal
cd /d %~dp0
set PYTHON=E:\anaconda\01\envs\RLimage\python.exe
set TEMP=E:\tmp
set TMP=E:\tmp
set MPLCONFIGDIR=E:\tmp\matplotlib
set TORCH_HOME=E:\tmp\torch
if not exist E:\tmp mkdir E:\tmp

set PARAMS_FILE=E:\tmp\nni_rlvr_smoke_params.json
(
  echo {
  echo   "signal": "ramp",
  echo   "unfreeze": "cls",
  echo   "optimizer": "adamw",
  echo   "reward_lambda": 0.3,
  echo   "alpha": 0.5,
  echo   "beta": 0.3
  echo }
) > %PARAMS_FILE%

%PYTHON% -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_smoke --limit-train 4 --limit-val 4 --rlvr-epochs 3 --early-stopping-patience 2 --params-file %PARAMS_FILE%
if errorlevel 1 pause & exit /b 1
echo RLVR trial smoke finished. Check runs\nni_rlvr_smoke\last_trial_result.json.
pause
