@echo off
title PaiGen AI — Living Neon Companion
cd /d "%~dp0"

echo ========================================================
echo               PaiGen AI Living Desktop Companion
echo ========================================================
echo.
echo Launching PaiGen GUI with animated Neon Face...
echo.
echo Controls:
echo   - Type in the input box at the bottom and hit Enter
echo   - Press 'C' or click the top-right button to toggle Camera
echo   - Press 'M' or click the top-right button to toggle Mic
echo.

python paigen_face.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [PaiGen exited with error code %ERRORLEVEL%]
    pause
)

