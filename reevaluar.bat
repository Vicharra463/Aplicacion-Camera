@echo off
REM ============================================================
REM  reevaluar - REEVALUA TODAS las imagenes de TODOS los dias
REM  (carpetas de imagenes_procesadas\placa\AAAA-MM-DD)
REM
REM  Con etiqueta (gt + etiquetas_reales.csv): OK/FALLO y % total.
REM  Sin etiqueta: muestra la lectura del motor, sin veredicto.
REM  Uso: reevaluar.bat [--dia AAAA-MM-DD] [--modo auto^|txt^|modelo]
REM ============================================================
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
