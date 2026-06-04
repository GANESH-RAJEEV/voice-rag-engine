@echo off
REM Installation script for RAG QA System
REM This script installs all required Python dependencies

echo.
echo ========================================
echo RAG QA System - Dependency Installation
echo ========================================
echo.

REM Check if pip is available
python -m pip --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python and pip are not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Installing dependencies from requirements.txt...
echo.

REM Install requirements
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed. Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo You can now run the Streamlit app:
echo.
echo   streamlit run streamlit_app.py
echo.
pause
