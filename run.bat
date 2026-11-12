@echo off
title GridPulse AI - Electricity Consumption & Peak Prediction System
echo =================================================================
echo    ⚡ Starting GridPulse AI Intelligent Prediction Dashboard   
echo =================================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    echo [INFO] Launching using virtual environment...
    .venv\Scripts\streamlit.exe run app.py --server.port 8501
    goto end
)

where streamlit >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Launching using system streamlit...
    streamlit run app.py --server.port 8501
    goto end
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Launching using py -m streamlit...
    py -m streamlit run app.py --server.port 8501
    goto end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Launching using python -m streamlit...
    python -m streamlit run app.py --server.port 8501
    goto end
)

echo [ERROR] Python or Streamlit not found in PATH.
pause

:end
