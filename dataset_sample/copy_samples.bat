@echo off
REM Double-click to copy the 6 representative samples into .\samples\
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0copy_samples.ps1"
echo.
pause
