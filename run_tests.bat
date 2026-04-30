@echo off
setlocal
call "%~dp0app\tests\run_tests.bat" %*
exit /b %ERRORLEVEL%