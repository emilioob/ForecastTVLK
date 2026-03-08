@echo off
setlocal

set "APP_DIR=%~dp0"
set "PACKAGED_EXE=%APP_DIR%TVLKForecastingTool.exe"
set "DIST_EXE=%APP_DIR%dist\TVLKForecastingTool\TVLKForecastingTool.exe"
set "VENV_PYTHON=%APP_DIR%.venv\Scripts\python.exe"
set "APP_FILE=%APP_DIR%app.py"

if exist "%PACKAGED_EXE%" (
    start "" "%PACKAGED_EXE%"
    exit /b 0
)

if exist "%DIST_EXE%" (
    start "" "%DIST_EXE%"
    exit /b 0
)

if exist "%VENV_PYTHON%" (
    start "" "%VENV_PYTHON%" -m streamlit run "%APP_FILE%"
    exit /b 0
)

echo Unable to find the packaged app or the local virtual environment.
echo Expected one of:
echo   %PACKAGED_EXE%
echo   %DIST_EXE%
echo   %VENV_PYTHON%
pause
