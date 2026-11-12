@echo off
REM Comandos del servidor de placas CYTV Ancon.
REM Uso: cytv estado | cytv placas | cytv log | ... (cytv ayuda para el listado)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cytv.ps1" %*
