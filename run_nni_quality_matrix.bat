@echo off
setlocal
cd /d %~dp0
set PYTHON=E:\anaconda\01\envs\RLimage\python.exe
set NNICTL=E:\anaconda\01\envs\RLimage\Scripts\nnictl.exe
set TEMP=E:\tmp
set TMP=E:\tmp
set MPLCONFIGDIR=E:\tmp\matplotlib
set TORCH_HOME=E:\tmp\torch
if not exist E:\tmp mkdir E:\tmp
set NNI_LOCAL_HOME=%CD%\nni_home
set HOME=%NNI_LOCAL_HOME%
set USERPROFILE=%NNI_LOCAL_HOME%
set NNI_HOME=%NNI_LOCAL_HOME%
set APPDATA=%NNI_LOCAL_HOME%\AppData\Roaming
if not exist "%APPDATA%" mkdir "%APPDATA%"
if not exist "%CD%\nni_experiments" mkdir "%CD%\nni_experiments"

%PYTHON% -c "import nni"
if errorlevel 1 (
  echo NNI is not installed in RLimage. Install it with:
  echo %PYTHON% -m pip install nni==3.0
  pause
  exit /b 1
)

%NNICTL% create --config nni_configs\quality_matrix_config.yml --port 8080
if errorlevel 1 pause & exit /b 1
echo NNI quality matrix started at http://localhost:8080
pause
