@echo off
REM Consola interactiva del servidor de placas CYTV Ancon.
REM Doble clic para el menu; doble clic + argumento para un comando.
cd /d "%~dp0"
"C:\CytvAncon\venv\Scripts\python.exe" "%~dp0consola.py" %*
pause
