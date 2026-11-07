@echo off
REM ============================================================================
REM Task 6: Single-Click Web Application Launcher
REM Launches local aiohttp server using CUDA-enabled Python 3.12 environment
REM ============================================================================

set PYTHON_PATH=C:\Users\fahad\AppData\Local\Programs\Python\Python312\python.exe

if not exist "%PYTHON_PATH%" (
    echo [ERROR] CUDA-enabled Python 3.12 was not found at:
    echo %PYTHON_PATH%
    echo Falling back to system python...
    set PYTHON_PATH=python
)

echo ============================================================================
echo   Launching Clinical Trial Eligibility Screening & Benchmarking System
echo   Target Model: Qwen/Qwen2.5-3B-Instruct (4-bit NF4 + LoRA Adapter)
echo   Draft Model:  Qwen/Qwen2.5-0.5B-Instruct (FP16 Speculative Engine)
echo   Local URL:    http://127.0.0.1:8000
echo ============================================================================

start http://127.0.0.1:8000
"%PYTHON_PATH%" task6\app.py

pause

