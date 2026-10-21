@echo off
title Depurador de placas CYTV
cd /d "%~dp0"
set "PY=C:\CytvAncon\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Abriendo el depurador en el navegador...
start "" http://127.0.0.1:8791/
"%PY%" depurador.py
pause
