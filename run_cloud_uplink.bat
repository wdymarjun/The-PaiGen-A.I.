@echo off
title PaiGen AI - PC Cloud Uplink (Connects PC to Vercel)
echo ========================================================
echo   PaiGen AI: PC Cloud Uplink Daemon
echo   Connects your PC securely to your Vercel Cloud Server
echo ========================================================
echo.

cd /d "%~dp0"
python -m paigen.server.cloud_uplink
pause
