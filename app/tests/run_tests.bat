@echo off
setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
set "APP_DIR=%PROJECT_ROOT%\app"
set "BACKEND_DIR=%APP_DIR%\server"
set "CLIENT_DIR=%APP_DIR%\client"
set "TESTS_DIR=%APP_DIR%\tests"
set "PYTEST_CONFIG=%TESTS_DIR%\pytest.ini"
set "CANONICAL_CONFIG=%APP_DIR%\resources\adsmod.json"
set "VENV_PYTHON=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "RUNTIME_NPM=%PROJECT_ROOT%\runtimes\nodejs\npm.cmd"
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "(Get-Content -Raw '%CANONICAL_CONFIG%' | ConvertFrom-Json).runtime.host"`) do set "BACKEND_HOST=%%A"& set "UI_HOST=%%A"
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "(Get-Content -Raw '%CANONICAL_CONFIG%' | ConvertFrom-Json).runtime.backend_port"`) do set "BACKEND_PORT=%%A"
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "(Get-Content -Raw '%CANONICAL_CONFIG%' | ConvertFrom-Json).runtime.frontend_port"`) do set "UI_PORT=%%A"
set "TEST_BACKEND_HOST=%BACKEND_HOST%"
set "TEST_UI_HOST=%UI_HOST%"
if /i "%TEST_BACKEND_HOST%"=="0.0.0.0" set "TEST_BACKEND_HOST=127.0.0.1"
if /i "%TEST_BACKEND_HOST%"=="::" set "TEST_BACKEND_HOST=127.0.0.1"
if /i "%TEST_BACKEND_HOST%"=="[::]" set "TEST_BACKEND_HOST=127.0.0.1"
if /i "%TEST_UI_HOST%"=="0.0.0.0" set "TEST_UI_HOST=127.0.0.1"
if /i "%TEST_UI_HOST%"=="::" set "TEST_UI_HOST=127.0.0.1"
if /i "%TEST_UI_HOST%"=="[::]" set "TEST_UI_HOST=127.0.0.1"
set "APP_TEST_BACKEND_URL=http://%TEST_BACKEND_HOST%:%BACKEND_PORT%"
set "APP_TEST_FRONTEND_URL=http://%TEST_UI_HOST%:%UI_PORT%"
if "!STANDARD_TEST_SKIP_LIVE_SERVERS!"=="" set "STANDARD_TEST_SKIP_LIVE_SERVERS=false"
if "!STANDARD_TEST_SKIP_FRONTEND!"=="" set "STANDARD_TEST_SKIP_FRONTEND=false"
set "TEST_RESULT=0"
set "STARTED_BACKEND=0"
if not exist "%VENV_PYTHON%" (echo [ERROR] Missing backend venv: "%VENV_PYTHON%"& exit /b 1)
set "PYTHON_CMD=%VENV_PYTHON%"
if exist "%RUNTIME_NPM%" (set "NPM_CMD=%RUNTIME_NPM%") else (set "NPM_CMD=npm")
set "PYTHONPATH=%PROJECT_ROOT%\app\server\common\src;%PROJECT_ROOT%\app\server\core\src;%PROJECT_ROOT%\app\server\ml\src;%PROJECT_ROOT%"
"%PYTHON_CMD%" -c "import adsmod_common.config; import adsmod_core.app" || exit /b 1
if /i "!STANDARD_TEST_SKIP_LIVE_SERVERS!"=="false" (
  curl -s --max-time 2 "%APP_TEST_BACKEND_URL%/health/ready" >nul 2>&1
  if errorlevel 1 (
    echo [INFO] Starting unified backend server...
    start "" /B /D "%PROJECT_ROOT%" "%PYTHON_CMD%" -m adsmod_core.cli --config "%CANONICAL_CONFIG%"
    set "STARTED_BACKEND=1"
  )
)
echo [STEP] Running Python tests...
if /i "!STANDARD_TEST_SKIP_LIVE_SERVERS!"=="true" (
  "%PYTHON_CMD%" -m pytest -c "%PYTEST_CONFIG%" "%TESTS_DIR%" --ignore "%TESTS_DIR%\e2e" -k "not performance" -v --tb=short %*
) else (
  "%PYTHON_CMD%" -m pytest -c "%PYTEST_CONFIG%" "%TESTS_DIR%" -k "not performance" -v --tb=short %*
)
if errorlevel 1 set "TEST_RESULT=1"
if /i "!STANDARD_TEST_SKIP_FRONTEND!"=="false" if exist "%CLIENT_DIR%\package.json" (
  echo [STEP] Running frontend validation...
  call "%NPM_CMD%" --prefix "%CLIENT_DIR%" run test:unit --if-present
  if errorlevel 1 set "TEST_RESULT=1"
  call "%NPM_CMD%" --prefix "%CLIENT_DIR%" run build
  if errorlevel 1 set "TEST_RESULT=1"
)
if "%STARTED_BACKEND%"=="1" (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R "LISTENING" ^| findstr /R ":%BACKEND_PORT% "') do taskkill /PID %%P /F >nul 2>&1
)
exit /b %TEST_RESULT%
