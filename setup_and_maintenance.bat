@echo off
setlocal
call "%~dp0start_on_windows.bat" maintenance %*
set "script_ec=%ERRORLEVEL%"
endlocal & exit /b %script_ec%
