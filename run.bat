@echo off
REM OpenVINO Object Detection Demo Runner for Windows
REM This script sets up the environment and runs the demo

echo 🚀 OpenVINO Object Detection Demo
echo =================================

REM Check if uv is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv is not installed. Installing...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ❌ Failed to install uv. Please install manually.
        pause
        exit /b 1
    )
)

REM Check if virtual environment exists
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    uv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Run the detector
echo 🎯 Starting object detection...
python openvino_detector.py

pause