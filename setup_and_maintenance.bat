@echo off
setlocal EnableExtensions

if /i "%~1"=="core" goto :install_core
if /i "%~1"=="ml" goto :install_ml
if /i "%~1"=="both" goto :install_both
if /i "%~1"=="init-db" goto :init_db
if /i "%~1"=="test" goto :run_tests
if /i "%~1"=="remove-logs" goto :remove_logs
if /i "%~1"=="clean-desktop" goto :clean_desktop
if /i "%~1"=="uninstall-core" goto :uninstall_core
if /i "%~1"=="uninstall-ml" goto :uninstall_ml
if /i "%~1"=="uninstall-both" goto :uninstall_both

:menu
cls
echo ==========================================================================
echo                         ADSMOD Setup And Maintenance
echo ==========================================================================
echo 1. Install or update core frontend + core service
echo 2. Install or update ML frontend + ML service
echo 3. Install or update both frontend + service stacks
echo 4. Initialize database
echo 5. Run test suite
echo 6. Remove logs
echo 7. Clean desktop packages
echo 8. Uninstall core webapp artifacts
echo 9. Uninstall ML webapp artifacts
echo A. Uninstall everything
echo 0. Exit
echo.
choice /c 123456789A0 /n /m "Select an option (1-9, A, 0): "
set "maintenance_choice=%ERRORLEVEL%"

if "%maintenance_choice%"=="1" goto :install_core
if "%maintenance_choice%"=="2" goto :install_ml
if "%maintenance_choice%"=="3" goto :install_both
if "%maintenance_choice%"=="4" goto :init_db
if "%maintenance_choice%"=="5" goto :run_tests
if "%maintenance_choice%"=="6" goto :remove_logs
if "%maintenance_choice%"=="7" goto :clean_desktop
if "%maintenance_choice%"=="8" goto :uninstall_core
if "%maintenance_choice%"=="9" goto :uninstall_ml
if "%maintenance_choice%"=="10" goto :uninstall_both
if "%maintenance_choice%"=="11" goto :exit
goto :menu

:install_core
call "%~dp0start_on_windows.bat" install core
goto :return_or_exit

:install_ml
call "%~dp0start_on_windows.bat" install ml
goto :return_or_exit

:install_both
call "%~dp0start_on_windows.bat" install both
goto :return_or_exit

:init_db
call "%~dp0start_on_windows.bat" init-db
goto :return_or_exit

:run_tests
call "%~dp0start_on_windows.bat" test
goto :return_or_exit

:remove_logs
call "%~dp0start_on_windows.bat" remove-logs
goto :return_or_exit

:clean_desktop
call "%~dp0start_on_windows.bat" clean-desktop
goto :return_or_exit

:uninstall_core
call "%~dp0start_on_windows.bat" uninstall core
goto :return_or_exit

:uninstall_ml
call "%~dp0start_on_windows.bat" uninstall ml
goto :return_or_exit

:uninstall_both
call "%~dp0start_on_windows.bat" uninstall both
goto :return_or_exit

:return_or_exit
set "script_ec=%ERRORLEVEL%"
if "%~1"=="" goto :menu
goto :exit

:exit
if not defined script_ec set "script_ec=0"
endlocal & exit /b %script_ec%
