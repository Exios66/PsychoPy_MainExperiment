@echo off
REM Installation script for PsychoPy WebGazer Integration on Windows

echo ===== PsychoPy WebGazer Integration Setup =====

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.9 or later.
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version') do set pyver=%%i
echo Detected Python version: %pyver%

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install setuptools and wheel
echo Installing setuptools and wheel...
pip install setuptools wheel

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Install the package in development mode
echo Installing the package in development mode...
pip install -e .

echo.
echo Installation complete!
echo.
echo To activate the virtual environment, run:
echo venv\Scripts\activate.bat
echo.
echo To run the demo experiment, run:
echo python -m PsychoPy.experiments.webgazer_demo
echo.
echo To run the main application, run:
echo python PsychoPy\run_application.py
echo.

pause 