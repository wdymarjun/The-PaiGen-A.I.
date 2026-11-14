@echo off
title PaiGen AI — Interactive Terminal Companion
cd /d "%~dp0"

echo ========================================================
echo          PaiGen AI Interactive CLI Companion
echo ========================================================
echo.
python interactive_companion.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [PaiGen CLI exited with error code %ERRORLEVEL%]
    pause
)
