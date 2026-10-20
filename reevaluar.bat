@echo off








setlocal
set "BASE=%~dp0"
if exist "%BASE%venv\Scripts\python.exe" (
    set "PY=%BASE%venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" "%BASE%entrenamiento\reevaluar.py" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
