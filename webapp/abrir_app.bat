@echo off
REM  Abre la app web del servidor CYTV Ancon (sube fotos por el FTP real).
cd /d "%~dp0"
echo.
echo   App web CYTV Ancon - http://localhost:3000
echo   (cierra esta ventana para detener la app; el SERVIDOR FTP NO se toca)
echo.
start "" http://localhost:3000
node server.js
