@echo off
setlocal

set "ROOT=%~dp0"
set "PY=%ROOT%app\server\.venv\Scripts\python.exe"
set "NPM=%ROOT%runtimes\nodejs\npm.cmd"
if not exist "%NPM%" set "NPM=npm"

if not exist "%PY%" (
  echo [ERROR] Missing backend runtime at "%PY%"
  exit /b 1
)

start "ADSMOD Core API" /D "%ROOT%app\server" "%PY%" -m uvicorn core_service.app:app --host 127.0.0.1 --port 8000
start "ADSMOD Core UI" /D "%ROOT%app\client" "%NPM%" run dev

endlocal
