@echo off
title PaiGen AI - Mobile Gateway Server
echo ========================================================
echo   PaiGen AI: Mobile Gateway Server ^& Personal Assistant
echo   Architect: Arjun Singh
echo ========================================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH!
    echo Please install Python 3.10+ and add it to your PATH.
    pause
    exit /b 1
)

echo Starting PaiGen Mobile Gateway Server on port 8765 ...
echo Connect your phone on the same Wi-Fi network.
echo.

python -m paigen.server.gateway

pause
