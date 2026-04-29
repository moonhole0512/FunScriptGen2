@echo off
setlocal enabledelayedexpansion

echo [1/4] Checking Python Environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo [2/4] Activating Local Virtual Environment...
set VENV_PATH=%~dp0.venv
if not exist "%VENV_PATH%" (
    echo [ERROR] Virtual environment not found at %VENV_PATH%
    pause
    exit /b
)
call "%VENV_PATH%\Scripts\activate"

echo [3/4] Syncing Dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [4/4] Launching Video-to-Funscript Tool...
python main.py

pause
