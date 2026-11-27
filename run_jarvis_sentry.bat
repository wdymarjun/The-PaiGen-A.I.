@echo off
title PaiGen AI - Jarvis Bluetooth Proximity Sentry
echo ========================================================
echo   PaiGen AI: Jarvis Bluetooth Proximity Sentry
echo   Architect: Arjun Singh
echo   Target Device: CH (41:42:35:D0:DF:57)
echo   Mode: Home Sentinel Node ^(Connected to Omen 16 RTX 5050^)
echo ========================================================
echo.

cd /d "%~dp0"

set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not exist "%PY_EXE%" (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_EXE=python"
    ) else (
        echo [ERROR] Python 3.11 not found at %PY_EXE%
        pause
        exit /b 1
    )
)

echo Starting Jarvis Proximity Monitor...
"%PY_EXE%" -m paigen.senses.bluetooth_sentry
pause
