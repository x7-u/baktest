@echo off
title Trading Backtester (Pine Script + MQL5)
cd /d "%~dp0"
echo ============================================================
echo   Trading Backtester - Pine Script + MQL5
echo ============================================================
echo.

:: Build Cython if .pyd files don't exist
if not exist "pine_fast*.pyd" (
    echo   Building Cython Pine Script engine...
    python setup.py build_ext --inplace >nul 2>&1
)
if not exist "mql5_fast*.pyd" (
    echo   Building Cython MQL5 engine...
    python setup.py build_ext --inplace >nul 2>&1
)
if exist "pine_fast*.pyd" (
    echo   Pine Script engine: Cython
) else (
    echo   Pine Script engine: Python (fallback)
)
if exist "mql5_fast*.pyd" (
    echo   MQL5 engine: Cython
) else (
    echo   MQL5 engine: Python (fallback)
)
echo.
start http://localhost:1234
python app.py
pause
