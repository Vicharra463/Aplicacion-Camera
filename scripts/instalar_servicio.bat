@echo off
setlocal EnableExtensions

















set "SERVICIO=CamaraCytvAncon"
set "RUTA=%~dp0"



if exist "%~dp0..\ftp_server.py" set "RUTA=%~dp0.."




if exist "C:\CytvAncon\config.ini" if /i not "%RUTA%"=="C:\CytvAncon\" set "RUTA=C:\CytvAncon\"

if /i "%~1"=="--check" goto :check
if /i "%~1"=="config" goto :config
if /i "%~1"=="" goto :instalar
goto :uso





:instalar
fltmc >nul 2>&1
if errorlevel 1 goto :pedir_uac
goto :instalar_elevado

:pedir_uac
echo Se necesita permiso de administrador para instalar el servicio.
echo.
echo Se abrira el cuadro de dialogo de Windows - elija "Si".
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%ComSpec%' -ArgumentList '/c','""%~f0""' -Verb RunAs"
if errorlevel 1 goto :uac_fallo
exit /b 0

:uac_fallo
echo.
echo [ERROR] No se obtuvo el permiso de administrador.
pause
exit /b 1

:instalar_elevado
if not exist "%~dp0instalar.ps1" goto :sin_ps1
echo Instalando - el detalle sale en esta misma ventana ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalar.ps1"
set "RC=%ERRORLEVEL%"
goto :instalar_fin

:instalar_fin
echo.
if "%RC%"=="0" goto :fin_ok
goto :fin_mal

:fin_ok
echo Instalacion terminada OK.
pause
exit /b 0

:fin_mal
echo.
echo [ERROR] La instalacion no termino bien - el paso que fallo
echo         esta impreso arriba en esta misma ventana.
pause
exit /b 1

:sin_ps1
echo [ERROR] Falta instalar.ps1 junto a este archivo.
echo         Copie la carpeta COMPLETA del proyecto y vuelva a correrlo.
pause
exit /b 1





:check
if not exist "%~dp0instalar.ps1" goto :sin_ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalar.ps1" -Check
exit /b 0





:config
if /i "%~2"=="" goto :config_mostrar
if /i "%~2"=="set" goto :config_set
if /i "%~2"=="editar" goto :config_editar
goto :uso

:config_mostrar
if not exist "%RUTA%config.ini" goto :config_no_existe
echo ============================================================
echo   config.ini  -  %RUTA%
echo ============================================================
findstr /n "^" "%RUTA%config.ini"
exit /b 0

:config_set
if not exist "%RUTA%config.ini" goto :config_no_existe
if "%~3"=="" goto :uso
if "%~4"=="" goto :uso
set "CFG_INI=%RUTA%config.ini"
set "CFG_CLAVE=%~3"
set "CFG_VALOR=%~4"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=$env:CFG_INI; $k=$env:CFG_CLAVE; $v=$env:CFG_VALOR; if ($k -notmatch '^[a-z_]+$') { Write-Host ('Clave invalida: [' + $k + '] - solo letras y guiones bajos'); exit 1 }; $lines=@(Get-Content -LiteralPath $p); $pat='^\s*' + [regex]::Escape($k) + '\s*='; if (@($lines | Where-Object { $_ -match $pat }).Count -eq 0) { Write-Host ('La clave [' + $k + '] no existe. Vea las claves con: instalar_servicio.bat config'); exit 1 }; if ($k -eq 'puerto' -and $v -notmatch '^[0-9]{1,5}$') { Write-Host 'ERROR: el puerto debe ser numerico'; exit 1 }; try { $out=$lines | ForEach-Object { if ($_ -match $pat) { $k + ' = ' + $v } else { $_ } }; [IO.File]::WriteAllLines($p, $out, (New-Object System.Text.UTF8Encoding($false))) } catch { Write-Host ('ERROR al guardar: ' + $_.Exception.Message); exit 1 }; Write-Host ('OK  ' + $k + ' = ' + $v)"
if errorlevel 1 exit /b 1
goto :config_servicio

:config_editar
if not exist "%RUTA%config.ini" goto :config_no_existe
start "" notepad "%RUTA%config.ini"
exit /b 0

:config_servicio
sc query %SERVICIO% >nul 2>&1
if errorlevel 1 goto :config_sin_servicio
fltmc >nul 2>&1
if errorlevel 1 goto :config_elevar_reinicio
echo Reiniciando el servicio para aplicar el cambio...
net stop %SERVICIO% >nul 2>&1
net start %SERVICIO% >nul 2>&1
sc query %SERVICIO% 2>nul | findstr /i "RUNNING" >nul
if errorlevel 1 goto :config_no_corrio
echo Servicio reiniciado.
exit /b 0

:config_elevar_reinicio
echo Cambio guardado, pero reiniciar el servicio requiere administrador.
echo Reinicielo usted mismo desde services.msc (clic derecho ^> Reiniciar)
echo o desde una consola de administrador:  net stop %SERVICIO% ^& net start %SERVICIO%
exit /b 0

:config_sin_servicio
echo El servicio aun no esta instalado - el cambio se aplicara al instalar.
exit /b 0

:config_no_corrio
echo [AVISO] El servicio no volvio a arrancar - mire C:\CytvAncon\logs\stderr.log
exit /b 1

:config_no_existe
echo [ERROR] No existe config.ini en %RUTA%
exit /b 1





:uso
echo.
echo Uso:
echo   instalar_servicio.bat                 instala todo - pide permiso de admin
echo   instalar_servicio.bat --check         diagnostico de solo lectura
echo   instalar_servicio.bat config          muestra config.ini completo
echo   instalar_servicio.bat config set CLAVE VALOR
echo   instalar_servicio.bat config editar   abre config.ini en el Bloc de notas
echo.
echo Requisitos: Windows 8.1 / Server 2012 R2 o superior, 64 bits.
echo El Python correcto para su SO se instala solo desde python.org.
echo La instalacion queda siempre en C:\CytvAncon.
echo.
echo Ejemplos:
echo   instalar_servicio.bat config set usuario camara
echo   instalar_servicio.bat config set password MiClave123
exit /b 1
