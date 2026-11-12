# Reinstala el servicio NSSM CamaraCytvAncon apuntando a C:\CytvAncon.
# Ejecutar en una PowerShell COMO ADMINISTRADOR.
#   powershell -ExecutionPolicy Bypass -File C:\CytvAncon\reinstalar_servicio.ps1

$ErrorActionPreference = 'Stop'
$Servicio  = 'CamaraCytvAncon'
$RutaBase  = 'C:\CytvAncon'
$Python    = Join-Path $RutaBase 'venv\Scripts\python.exe'
$Servidor  = Join-Path $RutaBase 'ftp_server.py'
$Nssm      = Join-Path $RutaBase 'nssm.exe'
$LogsDir   = Join-Path $RutaBase 'logs'

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'ERROR: ejecute esto en una PowerShell como administrador.' -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Python))  { Write-Host "Falta: $Python"  -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Servidor)){ Write-Host "Falta: $Servidor"-ForegroundColor Red; exit 1 }
if (-not (Test-Path $Nssm))    { Write-Host "Falta: $Nssm"    -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

Write-Host "1/5 Deteniendo servicio $Servicio (si existe)..." -ForegroundColor Cyan
& $Nssm stop $Servicio 2>$null | Out-Null
Start-Sleep -Seconds 2

Write-Host "2/5 Eliminando registro del servicio..." -ForegroundColor Cyan
& $Nssm remove $Servicio confirm | Out-Null
Start-Sleep -Seconds 1

Write-Host "3/5 Instalando servicio nuevo en $RutaBase..." -ForegroundColor Cyan
& $Nssm install $Servicio $Python
if ($LASTEXITCODE -ne 0) { Write-Host "nssm install fallo (codigo $LASTEXITCODE)" -ForegroundColor Red; exit 1 }

& $Nssm set $Servicio AppParameters 'ftp_server.py'
& $Nssm set $Servicio AppDirectory $RutaBase
& $Nssm set $Servicio AppStdout    (Join-Path $LogsDir 'stdout.log')
& $Nssm set $Servicio AppStderr    (Join-Path $LogsDir 'stderr.log')
& $Nssm set $Servicio Start        SERVICE_AUTO_START
& $Nssm set $Servicio AppExit      Default Restart
& $Nssm set $Servicio AppRotateFiles 1
& $Nssm set $Servicio AppRotateOnline 1
& $Nssm set $Servicio AppRotateBytes 10485760

Write-Host "4/5 Arrancando servicio..." -ForegroundColor Cyan
& $Nssm start $Servicio
Start-Sleep -Seconds 4

Write-Host "5/5 Verificando..." -ForegroundColor Cyan
$estado = (Get-Service -Name $Servicio -ErrorAction SilentlyContinue).Status
if ($estado -eq 'Running') {
    Write-Host "  Servicio: Running" -ForegroundColor Green
} else {
    Write-Host "  Servicio: $estado" -ForegroundColor Yellow
    Write-Host "  Revise $LogsDir\stderr.log" -ForegroundColor Yellow
}

$ftpOk = $false
try {
    $c = New-Object System.Net.Sockets.TcpClient
    $a = $c.BeginConnect('127.0.0.1', 2121, $null, $null)
    $ftpOk = $a.AsyncWaitHandle.WaitOne(2000) -and $c.Connected
    $c.Close()
} catch { }
if ($ftpOk) {
    Write-Host '  FTP 127.0.0.1:2121: OK' -ForegroundColor Green
} else {
    Write-Host '  FTP 127.0.0.1:2121: todavia no responde (espere 5 s mas o revise el log)' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Listo. Compruebe con:  cytv estado' -ForegroundColor Cyan
