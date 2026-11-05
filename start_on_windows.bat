@echo off
setlocal
where pwsh.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] PowerShell 7 ^(pwsh^) is required to launch ADSMOD.
  exit /b 1
)
pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_maintenance.ps1" -Launch
exit /b %ERRORLEVEL%
