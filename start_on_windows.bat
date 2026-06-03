@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0.") do set "repo_root=%%~fI"
set "app_dir=%repo_root%\app"
set "server_dir=%app_dir%\server"
set "client_dir=%app_dir%\client"
set "ml_client_dir=%app_dir%\ml_client"
set "scripts_dir=%app_dir%\scripts"
set "tests_dir=%app_dir%\tests"
set "log_dir=%app_dir%\resources\logs"
set "settings_dir=%repo_root%\settings"
set "runtimes_dir=%repo_root%\runtimes"
set "python_dir=%runtimes_dir%\python"
set "uv_dir=%runtimes_dir%\uv"
set "nodejs_dir=%runtimes_dir%\nodejs"

set "python_exe=%python_dir%\python.exe"
set "python_pth_file=%python_dir%\python314._pth"
set "uv_exe=%uv_dir%\uv.exe"
set "node_exe=%nodejs_dir%\node.exe"
set "npm_cmd=%nodejs_dir%\npm.cmd"
set "venv_python=%server_dir%\.venv\Scripts\python.exe"
set "pyproject=%server_dir%\pyproject.toml"
set "init_db_script=%scripts_dir%\initialize_database.py"
set "tauri_clean_script=%repo_root%\release\tauri\scripts\clean-tauri-build.ps1"
set "dotenv=%settings_dir%\.env"
set "uv_cache_dir=%server_dir%\.uv-cache"

set "py_version=3.14.2"
set "python_zip_filename=python-%py_version%-embed-amd64.zip"
set "python_zip_url=https://www.python.org/ftp/python/%py_version%/%python_zip_filename%"
set "python_zip_path=%python_dir%\%python_zip_filename%"

set "UV_CHANNEL=latest"
set "UV_ZIP_AMD=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP_ARM=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-aarch64-pc-windows-msvc.zip"
set "uv_zip_path=%uv_dir%\uv.zip"

set "nodejs_version=22.12.0"
set "nodejs_zip_filename=node-v%nodejs_version%-win-x64.zip"
set "nodejs_zip_url=https://nodejs.org/dist/v%nodejs_version%/%nodejs_zip_filename%"
set "nodejs_zip_path=%nodejs_dir%\%nodejs_zip_filename%"

call :load_env
if /i "%~1"=="maintenance" goto :maintenance_menu

:main_menu
cls
echo ==========================================================================
echo                               ADSMOD Menu
echo ==========================================================================
echo 1. Launch core webapp
echo 2. Launch ML webapp
echo 3. Launch both webapps
echo 4. Setup and maintenance
echo 5. Exit
echo.
choice /c 12345 /n /m "Select an option (1-5): "
set "main_choice=%ERRORLEVEL%"

if "%main_choice%"=="1" (
  call :launch_scope core
  goto :main_menu
)
if "%main_choice%"=="2" (
  call :launch_scope ml
  goto :main_menu
)
if "%main_choice%"=="3" (
  call :launch_scope both
  goto :main_menu
)
if "%main_choice%"=="4" goto :maintenance_menu
if "%main_choice%"=="5" goto :exit
goto :main_menu

:maintenance_menu
cls
echo ==========================================================================
echo                         ADSMOD Setup And Maintenance
echo ==========================================================================
echo 1. Install or update core webapp
echo 2. Install or update ML webapp
echo 3. Install or update both webapps
echo 4. Initialize database
echo 5. Run test suite
echo 6. Remove logs
echo 7. Clean desktop packages
echo 8. Uninstall core webapp artifacts
echo 9. Uninstall ML webapp artifacts
echo A. Uninstall everything
echo 0. Back
echo.
choice /c 123456789A0 /n /m "Select an option (1-9, A, 0): "
set "maintenance_choice=%ERRORLEVEL%"

if "%maintenance_choice%"=="1" (
  call :install_or_update_scope core
  goto :maintenance_menu
)
if "%maintenance_choice%"=="2" (
  call :install_or_update_scope ml
  goto :maintenance_menu
)
if "%maintenance_choice%"=="3" (
  call :install_or_update_scope both
  goto :maintenance_menu
)
if "%maintenance_choice%"=="4" (
  call :run_init_db
  goto :maintenance_menu
)
if "%maintenance_choice%"=="5" (
  call :run_tests
  goto :maintenance_menu
)
if "%maintenance_choice%"=="6" (
  call :remove_logs
  goto :maintenance_menu
)
if "%maintenance_choice%"=="7" (
  call :remove_desktop
  goto :maintenance_menu
)
if "%maintenance_choice%"=="8" (
  call :uninstall_scope core
  goto :maintenance_menu
)
if "%maintenance_choice%"=="9" (
  call :uninstall_scope ml
  goto :maintenance_menu
)
if "%maintenance_choice%"=="10" (
  call :uninstall_scope both
  goto :maintenance_menu
)
if "%maintenance_choice%"=="11" goto :main_menu
goto :maintenance_menu

:load_env
set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=6045"
set "CORE_SERVICE_HOST=127.0.0.1"
set "CORE_SERVICE_PORT=8000"
set "ML_SERVICE_HOST=127.0.0.1"
set "ML_SERVICE_PORT=8001"
set "UI_HOST=127.0.0.1"
set "UI_PORT=5173"
set "ML_UI_HOST=127.0.0.1"
set "ML_UI_PORT=5174"
set "OPTIONAL_DEPENDENCIES=false"

if exist "%dotenv%" (
  for /f "usebackq tokens=* delims=" %%L in ("%dotenv%") do (
    set "line=%%L"
    if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
      for /f "tokens=1,* delims==" %%A in ("!line!") do (
        set "k=%%A"
        set "v=%%B"
        if defined v (
          for /f "tokens=* delims= " %%Q in ("!v!") do set "v=%%Q"
          set "v=!v:"=!"
          if "!v:~0,1!"=="'" if "!v:~-1!"=="'" set "v=!v:~1,-1!"
        )
        if /i "!k!"=="FASTAPI_HOST" set "FASTAPI_HOST=!v!"
        if /i "!k!"=="FASTAPI_PORT" set "FASTAPI_PORT=!v!"
        if /i "!k!"=="CORE_SERVICE_HOST" set "CORE_SERVICE_HOST=!v!"
        if /i "!k!"=="CORE_SERVICE_PORT" set "CORE_SERVICE_PORT=!v!"
        if /i "!k!"=="ML_SERVICE_HOST" set "ML_SERVICE_HOST=!v!"
        if /i "!k!"=="ML_SERVICE_PORT" set "ML_SERVICE_PORT=!v!"
        if /i "!k!"=="UI_HOST" set "UI_HOST=!v!"
        if /i "!k!"=="UI_PORT" set "UI_PORT=!v!"
        if /i "!k!"=="ML_UI_HOST" set "ML_UI_HOST=!v!"
        if /i "!k!"=="ML_UI_PORT" set "ML_UI_PORT=!v!"
        if /i "!k!"=="OPTIONAL_DEPENDENCIES" set "OPTIONAL_DEPENDENCIES=!v!"
      )
    )
  )
)

if not defined CORE_SERVICE_HOST set "CORE_SERVICE_HOST=%FASTAPI_HOST%"
if not defined CORE_SERVICE_PORT set "CORE_SERVICE_PORT=%FASTAPI_PORT%"
if not defined ML_SERVICE_HOST set "ML_SERVICE_HOST=127.0.0.1"
if not defined ML_SERVICE_PORT set "ML_SERVICE_PORT=8001"
if not defined UI_HOST set "UI_HOST=127.0.0.1"
if not defined UI_PORT set "UI_PORT=5173"
if not defined ML_UI_HOST set "ML_UI_HOST=%UI_HOST%"
if not defined ML_UI_PORT set "ML_UI_PORT=5174"
exit /b 0

:launch_scope
set "scope=%~1"
call :load_env
echo.
echo [INFO] Preparing %scope% webapp launch...
call :prepare_scope "%scope%" "false"
if errorlevel 1 (
  echo [ERROR] Launch preparation failed.
  pause
  exit /b 1
)

if /i "%scope%"=="core" (
  call :assert_port_available "%CORE_SERVICE_PORT%" "Core API"
  if errorlevel 1 goto :launch_failed
  call :assert_port_available "%UI_PORT%" "Core UI"
  if errorlevel 1 goto :launch_failed

  start "ADSMOD Core API" /D "%server_dir%" "%venv_python%" -m uvicorn core_service.app:app --host %CORE_SERVICE_HOST% --port %CORE_SERVICE_PORT%
  start "ADSMOD Core UI" /D "%client_dir%" "%npm_cmd%" run dev
  start "" "http://%UI_HOST%:%UI_PORT%"
  echo [SUCCESS] Core webapp started.
  echo [INFO] Core API: http://%CORE_SERVICE_HOST%:%CORE_SERVICE_PORT%
  echo [INFO] Core UI : http://%UI_HOST%:%UI_PORT%
  pause
  exit /b 0
)

if /i "%scope%"=="ml" (
  call :assert_port_available "%ML_SERVICE_PORT%" "ML API"
  if errorlevel 1 goto :launch_failed
  call :assert_port_available "%ML_UI_PORT%" "ML UI"
  if errorlevel 1 goto :launch_failed

  start "ADSMOD ML API" /D "%server_dir%" "%venv_python%" -m uvicorn ml_service.app:app --host %ML_SERVICE_HOST% --port %ML_SERVICE_PORT%
  start "ADSMOD ML UI" /D "%ml_client_dir%" "%npm_cmd%" run dev
  start "" "http://%ML_UI_HOST%:%ML_UI_PORT%"
  echo [SUCCESS] ML webapp started.
  echo [INFO] ML API: http://%ML_SERVICE_HOST%:%ML_SERVICE_PORT%
  echo [INFO] ML UI : http://%ML_UI_HOST%:%ML_UI_PORT%
  pause
  exit /b 0
)

if /i "%scope%"=="both" (
  call :assert_port_available "%CORE_SERVICE_PORT%" "Core API"
  if errorlevel 1 goto :launch_failed
  call :assert_port_available "%ML_SERVICE_PORT%" "ML API"
  if errorlevel 1 goto :launch_failed
  call :assert_port_available "%UI_PORT%" "Core UI"
  if errorlevel 1 goto :launch_failed
  call :assert_port_available "%ML_UI_PORT%" "ML UI"
  if errorlevel 1 goto :launch_failed

  start "ADSMOD Core API" /D "%server_dir%" "%venv_python%" -m uvicorn core_service.app:app --host %CORE_SERVICE_HOST% --port %CORE_SERVICE_PORT%
  start "ADSMOD ML API" /D "%server_dir%" "%venv_python%" -m uvicorn ml_service.app:app --host %ML_SERVICE_HOST% --port %ML_SERVICE_PORT%
  start "ADSMOD Core UI" /D "%client_dir%" "%npm_cmd%" run dev
  start "ADSMOD ML UI" /D "%ml_client_dir%" "%npm_cmd%" run dev
  start "" "http://%UI_HOST%:%UI_PORT%"
  start "" "http://%ML_UI_HOST%:%ML_UI_PORT%"
  echo [SUCCESS] Core and ML webapps started.
  echo [INFO] Core API: http://%CORE_SERVICE_HOST%:%CORE_SERVICE_PORT%
  echo [INFO] ML API  : http://%ML_SERVICE_HOST%:%ML_SERVICE_PORT%
  echo [INFO] Core UI : http://%UI_HOST%:%UI_PORT%
  echo [INFO] ML UI   : http://%ML_UI_HOST%:%ML_UI_PORT%
  pause
  exit /b 0
)

:launch_failed
echo [ERROR] Launch aborted.
pause
exit /b 1

:install_or_update_scope
set "scope=%~1"
echo.
echo [INFO] Installing or updating %scope% webapp...
call :prepare_scope "%scope%" "true"
if errorlevel 1 (
  echo [ERROR] Install or update failed.
  pause
  exit /b 1
)
echo [SUCCESS] %scope% webapp is ready.
pause
exit /b 0

:prepare_scope
set "scope=%~1"
set "build_frontends=%~2"

call :ensure_runtime_dirs
call :ensure_python_runtime
if errorlevel 1 exit /b 1
call :ensure_uv_runtime
if errorlevel 1 exit /b 1
call :ensure_node_runtime
if errorlevel 1 exit /b 1
call :sync_backend_dependencies
if errorlevel 1 exit /b 1
call :install_frontend_scope "%scope%" "%build_frontends%"
if errorlevel 1 exit /b 1

if exist "%uv_cache_dir%" rd /s /q "%uv_cache_dir%" >nul 2>&1
exit /b 0

:ensure_runtime_dirs
if not exist "%runtimes_dir%" md "%runtimes_dir%" >nul 2>&1
if not exist "%python_dir%" md "%python_dir%" >nul 2>&1
if not exist "%uv_dir%" md "%uv_dir%" >nul 2>&1
if not exist "%nodejs_dir%" md "%nodejs_dir%" >nul 2>&1
exit /b 0

:ensure_python_runtime
echo [STEP 1/4] Ensuring portable Python runtime
if not exist "%python_exe%" (
  echo [DL] %python_zip_url%
  call :download_file "%python_zip_url%" "%python_zip_path%"
  if errorlevel 1 exit /b 1
  call :expand_zip "%python_zip_path%" "%python_dir%"
  if errorlevel 1 exit /b 1
  del /q "%python_zip_path%" >nul 2>&1
)

if exist "%python_pth_file%" (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $p='%python_pth_file%'; if(Test-Path -LiteralPath $p){ $content=Get-Content -LiteralPath $p -Raw; if($content -match '#import site'){ $content=$content -replace '#import site','import site'; Set-Content -LiteralPath $p -Value $content -Force } }" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Failed to update "%python_pth_file%".
    exit /b 1
  )
)

for /f "delims=" %%V in ('"%python_exe%" -c "import platform; print(platform.python_version())"') do set "found_py=%%V"
echo [OK] Python ready: !found_py!
exit /b 0

:ensure_uv_runtime
echo [STEP 2/4] Ensuring portable uv runtime
set "uv_zip_url=%UV_ZIP_AMD%"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "uv_zip_url=%UV_ZIP_ARM%"

if not exist "%uv_exe%" (
  echo [DL] %uv_zip_url%
  call :download_file "%uv_zip_url%" "%uv_zip_path%"
  if errorlevel 1 exit /b 1
  call :expand_zip "%uv_zip_path%" "%uv_dir%"
  if errorlevel 1 exit /b 1
  del /q "%uv_zip_path%" >nul 2>&1

  for /f "usebackq delims=" %%F in (`powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; (Get-ChildItem -LiteralPath '%uv_dir%' -Recurse -Filter 'uv.exe' | Select-Object -First 1 -ExpandProperty FullName)"`) do set "found_uv=%%F"
  if not defined found_uv (
    echo [ERROR] uv.exe not found after extraction.
    exit /b 1
  )
  if /i not "!found_uv!"=="%uv_exe%" copy /y "!found_uv!" "%uv_exe%" >nul
)

for /f "delims=" %%V in ('"%uv_exe%" --version') do echo [OK] %%V
exit /b 0

:ensure_node_runtime
echo [STEP 3/4] Ensuring portable Node.js runtime
if not exist "%node_exe%" (
  echo [DL] %nodejs_zip_url%
  call :download_file "%nodejs_zip_url%" "%nodejs_zip_path%"
  if errorlevel 1 exit /b 1
  call :expand_zip "%nodejs_zip_path%" "%nodejs_dir%"
  if errorlevel 1 exit /b 1
  del /q "%nodejs_zip_path%" >nul 2>&1
)

set "node_archive_dir=%nodejs_dir%\node-v%nodejs_version%-win-x64"
if exist "%node_archive_dir%\node.exe" (
  call :promote_node_runtime "%node_archive_dir%"
  if errorlevel 1 exit /b 1
)

if not exist "%node_exe%" (
  echo [ERROR] node.exe not found in "%nodejs_dir%".
  exit /b 1
)
if not exist "%npm_cmd%" (
  echo [ERROR] npm.cmd not found in "%nodejs_dir%".
  exit /b 1
)

for /f "delims=" %%V in ('"%node_exe%" --version') do echo [OK] Node.js ready: %%V
exit /b 0

:sync_backend_dependencies
echo [STEP 4/4] Syncing backend dependencies
if not exist "%pyproject%" (
  echo [ERROR] Missing pyproject: "%pyproject%"
  exit /b 1
)

set "PYTHONHOME=%python_dir%"
set "PYTHONNOUSERSITE=1"
pushd "%server_dir%" >nul
"%uv_exe%" sync --all-packages --group dev --python "%python_exe%"
set "sync_ec=%ERRORLEVEL%"
if not "%sync_ec%"=="0" (
  "%uv_exe%" sync --all-packages --group dev
  set "sync_ec=%ERRORLEVEL%"
)
popd >nul
set "PYTHONHOME="
set "PYTHONNOUSERSITE="

if not "%sync_ec%"=="0" (
  echo [ERROR] uv sync failed with code %sync_ec%.
  exit /b 1
)

if not exist "%venv_python%" (
  echo [ERROR] Backend venv python not found at "%venv_python%".
  exit /b 1
)
echo [OK] Backend environment ready.
exit /b 0

:install_frontend_scope
set "scope=%~1"
set "build_frontends=%~2"

if /i "%scope%"=="core" (
  call :install_frontend "%client_dir%" "Core UI" "%build_frontends%"
  exit /b %ERRORLEVEL%
)
if /i "%scope%"=="ml" (
  call :install_frontend "%ml_client_dir%" "ML UI" "%build_frontends%"
  exit /b %ERRORLEVEL%
)
if /i "%scope%"=="both" (
  call :install_frontend "%client_dir%" "Core UI" "%build_frontends%"
  if errorlevel 1 exit /b 1
  call :install_frontend "%ml_client_dir%" "ML UI" "%build_frontends%"
  exit /b %ERRORLEVEL%
)

echo [ERROR] Unknown scope "%scope%".
exit /b 1

:install_frontend
set "frontend_dir=%~1"
set "frontend_name=%~2"
set "build_frontends=%~3"
set "frontend_lockfile=%frontend_dir%\package-lock.json"
set "frontend_modules=%frontend_dir%\node_modules"

if /i "%build_frontends%"=="true" (
  echo [STEP] Installing dependencies for %frontend_name%
  pushd "%frontend_dir%" >nul
  if exist "%frontend_lockfile%" (
    call "%npm_cmd%" ci
  ) else (
    call "%npm_cmd%" install
  )
  set "npm_ec=%ERRORLEVEL%"
  popd >nul
  if not "%npm_ec%"=="0" (
    echo [ERROR] Dependency install failed for %frontend_name% with code %npm_ec%.
    exit /b 1
  )

  echo [STEP] Building %frontend_name%
  pushd "%frontend_dir%" >nul
  call "%npm_cmd%" run build
  set "build_ec=%ERRORLEVEL%"
  popd >nul
  if not "%build_ec%"=="0" (
    echo [ERROR] Build failed for %frontend_name% with code %build_ec%.
    exit /b 1
  )

  echo [OK] %frontend_name% updated.
  exit /b 0
)

if exist "%frontend_modules%" (
  echo [OK] %frontend_name% dependencies already present.
  exit /b 0
)

echo [STEP] Installing dependencies for %frontend_name%
pushd "%frontend_dir%" >nul
if exist "%frontend_lockfile%" (
  call "%npm_cmd%" ci
) else (
  call "%npm_cmd%" install
)
set "npm_ec=%ERRORLEVEL%"
popd >nul
if not "%npm_ec%"=="0" (
  echo [ERROR] Dependency install failed for %frontend_name% with code %npm_ec%.
  exit /b 1
)

echo [OK] %frontend_name% dependencies installed.
exit /b 0

:run_init_db
echo.
if not exist "%init_db_script%" (
  echo [ERROR] Missing database script: "%init_db_script%".
  pause
  exit /b 1
)

call :prepare_scope core false
if errorlevel 1 (
  echo [ERROR] Database initialization prerequisites failed.
  pause
  exit /b 1
)

pushd "%server_dir%" >nul
"%uv_exe%" run --project "%server_dir%" --python "%python_exe%" python "%init_db_script%"
set "run_ec=%ERRORLEVEL%"
popd >nul
if "%run_ec%"=="0" (
  echo [SUCCESS] Database initialization completed.
  pause
  exit /b 0
)

echo [ERROR] Database initialization failed with exit code %run_ec%.
pause
exit /b 1

:run_tests
set "test_script=%tests_dir%\run_tests.bat"
if not exist "%test_script%" (
  echo [ERROR] Missing test script: "%test_script%".
  pause
  exit /b 1
)

echo [RUN] Executing test suite: "%test_script%"
pushd "%tests_dir%" >nul
call "%test_script%"
set "test_ec=%ERRORLEVEL%"
popd >nul
if "%test_ec%"=="0" (
  echo [SUCCESS] Test suite completed.
  pause
  exit /b 0
)

echo [ERROR] Test suite failed with exit code %test_ec%.
pause
exit /b 1

:remove_logs
if not exist "%log_dir%\" (
  echo [INFO] Log directory not found at "%log_dir%".
  pause
  exit /b 0
)
if exist "%log_dir%\*.log" (
  del /q "%log_dir%\*.log"
  echo [SUCCESS] Log files deleted.
  pause
  exit /b 0
)

echo [INFO] No log files found.
pause
exit /b 0

:remove_desktop
if exist "%tauri_clean_script%" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%tauri_clean_script%"
  set "clean_ec=%ERRORLEVEL%"
  if not "%clean_ec%"=="0" (
    echo [ERROR] Desktop package cleanup failed with exit code %clean_ec%.
    pause
    exit /b 1
  )
  echo [SUCCESS] Desktop package cleanup completed.
  pause
  exit /b 0
)

if exist "%client_dir%\src-tauri\target\release" rd /s /q "%client_dir%\src-tauri\target\release"
if exist "%client_dir%\src-tauri\target" rd /s /q "%client_dir%\src-tauri\target"
if exist "%repo_root%\release\windows" rd /s /q "%repo_root%\release\windows"
echo [SUCCESS] Desktop package cleanup completed.
pause
exit /b 0

:uninstall_scope
set "scope=%~1"
echo.
echo [UNINSTALL] Removing %scope% artifacts...

if /i "%scope%"=="core" (
  call :remove_frontend_artifacts "%client_dir%"
  echo [SUCCESS] Core webapp artifacts removed.
  pause
  exit /b 0
)

if /i "%scope%"=="ml" (
  call :remove_frontend_artifacts "%ml_client_dir%"
  echo [SUCCESS] ML webapp artifacts removed.
  pause
  exit /b 0
)

if /i "%scope%"=="both" (
  call :remove_frontend_artifacts "%client_dir%"
  call :remove_frontend_artifacts "%ml_client_dir%"
  if exist "%server_dir%\.venv" rd /s /q "%server_dir%\.venv"
  if exist "%server_dir%\uv.lock" del /q "%server_dir%\uv.lock"
  if exist "%repo_root%\uv.lock" del /q "%repo_root%\uv.lock"
  if exist "%runtimes_dir%\uv" rd /s /q "%runtimes_dir%\uv"
  if exist "%runtimes_dir%\.uv-cache" rd /s /q "%runtimes_dir%\.uv-cache"
  if exist "%runtimes_dir%\uv_cache" rd /s /q "%runtimes_dir%\uv_cache"
  if exist "%runtimes_dir%\python" rd /s /q "%runtimes_dir%\python"
  if exist "%runtimes_dir%\nodejs" rd /s /q "%runtimes_dir%\nodejs"
  echo [SUCCESS] All app artifacts removed.
  pause
  exit /b 0
)

echo [ERROR] Unknown uninstall scope "%scope%".
pause
exit /b 1

:remove_frontend_artifacts
set "frontend_dir=%~1"
if exist "%frontend_dir%\node_modules" rd /s /q "%frontend_dir%\node_modules"
if exist "%frontend_dir%\.angular" rd /s /q "%frontend_dir%\.angular"
if exist "%frontend_dir%\dist" rd /s /q "%frontend_dir%\dist"
if exist "%frontend_dir%\package-lock.json" del /q "%frontend_dir%\package-lock.json"
exit /b 0

:assert_port_available
set "target_port=%~1"
set "target_name=%~2"
if "%target_port%"=="" exit /b 1

netstat -ano | findstr /R /C:":%target_port% .*LISTENING" >nul
if errorlevel 1 exit /b 0

echo [ERROR] %target_name% port %target_port% is already in use.
exit /b 1

:download_file
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%~1' -OutFile '%~2'"
exit /b %ERRORLEVEL%

:expand_zip
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath '%~1' -DestinationPath '%~2' -Force"
exit /b %ERRORLEVEL%

:promote_node_runtime
set "node_source_dir=%~1"
if not defined node_source_dir exit /b 1
for %%D in ("%~1") do set "node_source_dir=%%~fD"
if /i "%node_source_dir%"=="%nodejs_dir%" exit /b 0

robocopy "%node_source_dir%" "%nodejs_dir%" /MOVE /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NC /NS >nul
set "node_move_ec=%ERRORLEVEL%"
if %node_move_ec% geq 8 (
  echo [ERROR] Failed to flatten portable Node.js runtime from "%node_source_dir%".
  exit /b %node_move_ec%
)

if exist "%node_source_dir%" rd /s /q "%node_source_dir%" >nul 2>&1
exit /b 0

:exit
endlocal
