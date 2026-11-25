@echo off
title PaiGen AI - Render Parent Bridge & PC Uplink
echo ====================================================================
echo   PaiGen AI: Render Parent Bridge ^& PC Workstation Uplink
echo   Connects your Windows PC ^<--^> Render (https://paigen-ai-cloud.onrender.com)
echo   Control your PC and PaiGen Brain from both PC and Vivo V29 APK.
echo ====================================================================
echo.

cd /d "%~dp0"
python -m paigen.server.cloud_uplink
pause
