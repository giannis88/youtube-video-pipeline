@echo off
echo Setting up YouTube Video Processing Pipeline

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install project dependencies
echo Installing project dependencies...
pip install -r requirements.txt

REM Install the project in editable mode
echo Installing project in editable mode...
pip install -e .

REM Create necessary directories
echo Creating input and output directories...
mkdir input 2>nul
mkdir output 2>nul

REM Generate default configuration
echo Generating default configuration...
python -m src.cli generate-config

echo.
echo Setup complete! 
echo Activate the virtual environment with: venv\Scripts\activate
echo Run the CLI with: vidpipe --help
pause
