<#
.SYNOPSIS
  Comandos del servidor de placas CYTV Ancon (estilo artisan).

.DESCRIPTION
  Un comando por accion. Sin menu, sin TUI: se escribe y se ejecuta.

.EXAMPLE
  cytv estado
  cytv placas
  cytv historial
  cytv log -Lineas 50
  cytv fotos C:\Users\felip\foto.jpg -Canal 02
  cytv servicios reiniciar
  cytv servicios parar
  cytv servicios arrancar
  cytv config -Abrir
  cytv web
  cytv ayuda
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('estado', 'placas', 'historial', 'log', 'fotos',
                 'servicios', 'config', 'web', 'ayuda', 'reiniciar', 'limpiar')]
    [string]$Comando = 'ayuda',

    [Parameter(Position = 1)]
    [string]$Ruta,

    [ValidateSet('01', '02')]
    [string]$Canal = '01',

    [int]$Lineas = 30,

    [switch]$Abrir,

    [switch]$Fuerza
)

$ErrorActionPreference = 'Stop'
$RutaBase = 'C:\CytvAncon'
$Servicio = 'CamaraCytvAncon'
$ConfigIni = Join-Path $RutaBase 'config.ini'

# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
function Write-Titulo {
    Write-Host ''
    Write-Host 'CYTV Ancon - servidor de placas' -ForegroundColor Cyan
    Write-Host ('-' * 44) -ForegroundColor DarkGray
}

function Get-IniValor {
    param([string]$Seccion, [string]$Clave, [string]$Defecto = '')
    $enSeccion = $false
    foreach ($linea in Get-Content $ConfigIni -Encoding UTF8) {
        $t = $linea.Trim()
        if ($t -match '^\[(.+)\]$') {
            $enSeccion = ($Matches[1] -eq $Seccion)
            continue
        }
        if ($enSeccion -and $t -match "^\s*$Clave\s*=\s*(.+)$") {
            return $Matches[1].Trim()
        }
    }
    return $Defecto
}

function Test-Tcp {
    param([int]$Puerto, [int]$EsperaMs = 1000)
    $cliente = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $cliente.BeginConnect('127.0.0.1', $Puerto, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne($EsperaMs)
        if ($ok -and $cliente.Connected) { return $true }
        return $false
    } catch {
        return $false
    } finally {
        $cliente.Close()
    }
}

function Get-EstadoServicio {
    try {
        (Get-Service -Name $Servicio -ErrorAction Stop).Status.ToString()
    } catch {
        'desconocido'
    }
}

function Get-RutaReporteHoy {
    $hoy = Get-Date -Format 'yyyy-MM-dd'
    $carpeta = Get-IniValor 'rutas' 'reportes' (Join-Path $RutaBase 'reportes')
    Join-Path $carpeta "placas-$hoy.json"
}

# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------
function Invoke-Estado {
    $puertoFtp   = [int](Get-IniValor 'ftp' 'puerto' '2121')
    $servicio    = Get-EstadoServicio
    $ftpOk       = Test-Tcp $puertoFtp
    $webOk       = Test-Tcp 3000
    $onnx        = Join-Path $RutaBase 'modelo\placas.onnx'
    $hoy         = Get-Date -Format 'yyyy-MM-dd'
    $total       = '?'
    try {
        $rep = Get-Content (Get-RutaReporteHoy) -Raw -Encoding UTF8 | ConvertFrom-Json
        $total = $rep.total_placas
    } catch { }

    Write-Titulo
    $svc = if ($servicio -eq 'Running') { 'OK' } else { 'FALLA' }
    $svcColor = if ($servicio -eq 'Running') { 'Green' } else { 'Red' }
    Write-Host ('  Servicio {0,-18} ' -f $Servicio) -NoNewline
    Write-Host ('{0,-8}' -f $svc) -ForegroundColor $svcColor -NoNewline
    Write-Host (' ({0})' -f $servicio)

    Write-Host ('  FTP 127.0.0.1:{0,-10} ' -f $puertoFtp) -NoNewline
    if ($ftpOk) { Write-Host 'OK' -ForegroundColor Green } else { Write-Host 'FALLA' -ForegroundColor Red }

    Write-Host '  App web :3000             ' -NoNewline
    if ($webOk) { Write-Host 'OK' -ForegroundColor Green } else { Write-Host 'FALLA' -ForegroundColor Red }

    if (Test-Path $onnx) {
        $mb = [math]::Round((Get-Item $onnx).Length / 1MB, 1)
        Write-Host '  Modelo ONNX               OK  ' -NoNewline -ForegroundColor Green
        Write-Host ('({0} MB)' -f $mb)
    } else {
        Write-Host "  Modelo ONNX               FALLA (no existe $onnx)" -ForegroundColor Red
    }
    Write-Host ''
    Write-Host ('  Placas hoy ({0}): {1}' -f $hoy, $total)
}

function Get-Levenshtein {
    # Distancia de edicion (mismo criterio que el worker: <= 2 = misma placa).
    # Index LINEAL (sin comas en $d[...]): Windows PowerShell 5.1 no parsea
    # $d[$i - 1, $j] (la coma del indice tiene prioridad sobre la resta).
    param([string]$a, [string]$b)
    $m = $a.Length; $n = $b.Length
    if ($m -eq 0) { return $n }
    if ($n -eq 0) { return $m }
    $w = $n + 1                      # ancho de cada fila en la matriz lineal
    $d = New-Object 'int[]' (($m + 1) * $w)
    for ($i = 0; $i -le $m; $i++) { $d[$i * $w] = $i }
    for ($j = 0; $j -le $n; $j++) { $d[$j] = $j }
    for ($i = 1; $i -le $m; $i++) {
        for ($j = 1; $j -le $n; $j++) {
            $cost = if ($a[$i - 1] -eq $b[$j - 1]) { 0 } else { 1 }
            $arriba    = $d[($i - 1) * $w + $j] + 1
            $izquierda = $d[$i * $w + ($j - 1)] + 1
            $diagonal  = $d[($i - 1) * $w + ($j - 1)] + $cost
            $d[$i * $w + $j] = [Math]::Min([Math]::Min($arriba, $izquierda), $diagonal)
        }
    }
    return $d[$m * $w + $n]
}

function Invoke-Placas {
    $ruta = Get-RutaReporteHoy
    Write-Titulo
    if (-not (Test-Path $ruta)) {
        Write-Host "  No hay reporte de hoy todavia: $ruta"
        return
    }
    $rep = Get-Content $ruta -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $rep.placas -or $rep.placas.Count -eq 0) {
        Write-Host '  Todavia ninguna placa guardada hoy.'
        return
    }
    # Agrupa variantes de la MISMA placa (B1Y-756 / BIY-756 / B1V-756):
    # mismo criterio que el worker (Levenshtein <= 2).
    $grupos = @()
    foreach ($p in $rep.placas) {
        $texto = $p.placa
        $npl  = if ($p.imagenes -and $p.imagenes.placa)  { @($p.imagenes.placa).Count }  else { 0 }
        $npan = if ($p.imagenes -and $p.imagenes.panoramica) { @($p.imagenes.panoramica).Count } else { 0 }
        $g = $grupos | Where-Object { (Get-Levenshtein $_.placa $texto) -le 2 } | Select-Object -First 1
        if ($g) {
            $g.detecciones += [int]$p.detecciones
            if ($p.confianza_max -gt $g.confianza_max) { $g.confianza_max = $p.confianza_max }
            if ($p.primera_vez -lt $g.primera_vez) { $g.primera_vez = $p.primera_vez }
            if ($p.ultima_vez -gt $g.ultima_vez) { $g.ultima_vez = $p.ultima_vez }
            $g.npl += $npl; $g.npan += $npan
        } else {
            $grupos += [pscustomobject]@{
                placa = $texto; detecciones = [int]$p.detecciones
                confianza_max = $p.confianza_max
                primera_vez = $p.primera_vez; ultima_vez = $p.ultima_vez
                npl = $npl; npan = $npan
            }
        }
    }
    $grupos = @($grupos | Sort-Object primera_vez)
    $totalDet = ($grupos | Measure-Object detecciones -Sum).Sum
    Write-Host ('  Placas del dia {0}  ({1} placas, {2} detecciones)' -f
        $rep.fecha, $grupos.Count, $totalDet)
    Write-Host ''
    Write-Host ('  {0,-12} {1,9} {2,9} {3,10}  {4}' -f
        'PLACA', 'PRIMERA', 'ULTIMA', 'CONFIANZA', 'FOTOS') -ForegroundColor Yellow
    foreach ($g in $grupos) {
        $pv = if ($g.primera_vez) { $g.primera_vez.Substring(11, 8) } else { '' }
        $uv = if ($g.ultima_vez)  { $g.ultima_vez.Substring(11, 8)  } else { '' }
        $conf = if ($g.confianza_max) { '{0:P0}' -f $g.confianza_max } else { '-' }
        Write-Host ('  {0,-12} {1,9} {2,9} {3,10}  {4} placa / {5} pano' -f
            $g.placa, $pv, $uv, $conf, $g.npl, $g.npan)
    }
}

function Invoke-Historial {
    $py = Join-Path $RutaBase 'venv\Scripts\python.exe'
    if (-not (Test-Path $py)) {
        Write-Host "  No se encontro el python del venv: $py" -ForegroundColor Red
        return
    }
    $db = Join-Path $RutaBase 'registro.db'
    $codigo = @"
import sqlite3
con = sqlite3.connect(r'$db')
filas = con.execute(
    'SELECT fecha_hora, placa, confianza, estado, archivo '
    'FROM registros ORDER BY id DESC LIMIT 15').fetchall()
for fh, placa, conf, estado, archivo in filas:
    c = f'{conf:.0%}' if conf is not None else '-'
    p = placa or '-'
    print(f'{fh:<20} {p:<12} {c:>6}  {estado:<16} {archivo}')
"@
    Write-Titulo
    Write-Host '  Ultimos 15 registros (aceptados y descartados):'
    Write-Host ''
    Write-Host ('  {0,-20} {1,-12} {2,6}  {3,-16} {4}' -f
        'FECHA/HORA', 'PLACA', 'CONF', 'ESTADO', 'ARCHIVO') -ForegroundColor Yellow
    $codigo | & $py -
}

function Invoke-Log {
    $carpeta = Get-IniValor 'rutas' 'logs' (Join-Path $RutaBase 'logs')
    $ruta = Join-Path $carpeta 'servidor.log'
    Write-Titulo
    if (-not (Test-Path $ruta)) {
        Write-Host "  No existe el log: $ruta"
        return
    }
    Write-Host "  Ultimas $Lineas lineas de ${ruta}:"
    Write-Host ''
    Get-Content $ruta -Tail $Lineas -Encoding UTF8 | ForEach-Object { Write-Host "  $_" }
}

function Invoke-Fotos {
    if (-not $Ruta -or -not (Test-Path $Ruta)) {
        Write-Host '  Indique la ruta de la foto:' -ForegroundColor Yellow
        Write-Host '    cytv fotos C:\ruta\foto.jpg -Canal 02'
        return
    }
    $puerto = [int](Get-IniValor 'ftp' 'puerto' '2121')
    $usuario = Get-IniValor 'ftp' 'usuario' 'admin'
    $clave   = Get-IniValor 'ftp' 'password' ''
    $ts = Get-Date -Format 'yyyyMMddHHmmssfff'
    $nombre = "web_${Canal}_${ts}_MD_WITH_TARGET.jpg"

    Write-Titulo
    Write-Host "  Subiendo $Ruta como $nombre (canal $Canal)..."
    try {
        $web = New-Object System.Net.WebClient
        $web.Credentials = New-Object System.Net.NetworkCredential($usuario, $clave)
        $destino = "ftp://127.0.0.1:$puerto/$nombre"
        $web.UploadFile($destino, 'STOR', $Ruta)
        Write-Host '  Enviada. El worker la evalua en 1-3 s.' -ForegroundColor Green
    } catch {
        Write-Host "  Fallo la subida por FTP: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host '  Revise que el servicio este Running (cytv estado).'
        return
    }
    Start-Sleep -Seconds 3.5
    $carpetaLog = Get-IniValor 'rutas' 'logs' (Join-Path $RutaBase 'logs')
    $log = Join-Path $carpetaLog 'servidor.log'
    Write-Host ''
    $coincidencias = Get-Content $log -Tail 40 -Encoding UTF8 | Select-String -SimpleMatch $nombre
    if ($coincidencias) {
        $coincidencias | ForEach-Object { Write-Host "  $($_.Line)" }
    } else {
        Write-Host '  Todavia no aparece en el log; espere unos segundos y use "cytv log".'
    }
}

function Invoke-Servicios {
    $accion = if ($Ruta) { $Ruta.ToLower() } else { 'estado' }
    $estado = Get-EstadoServicio
    Write-Titulo
    Write-Host ('  Servicio {0}: ' -f $Servicio) -NoNewline
    $color = if ($estado -eq 'Running') { 'Green' } else { 'Yellow' }
    Write-Host $estado -ForegroundColor $color

    function Test-Admin {
        ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    function Confirmar([string]$Texto) {
        if ($Fuerza) { return $true }
        $resp = Read-Host ('  ' + $Texto + ' (s/N)')
        return ($resp -in @('s', 'si', 'sí'))
    }
    function EsperarEstado([string]$Deseado) {
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Seconds 1
            $estado = Get-EstadoServicio
            if ($estado -eq $Deseado) {
                Write-Host ('  Servicio ' + $Deseado + '.') -ForegroundColor Green
                return $true
            }
        }
        Write-Host ('  Estado actual: ' + $estado + '. Si no cambio, revise el log.') -ForegroundColor Yellow
        return $false
    }

    switch ($accion) {
        'parar' {
            if (-not (Confirmar 'Parar el servicio ahora?')) { Write-Host '  Cancelado.'; return }
            if (-not (Test-Admin)) {
                Write-Host '  Requiere elevacion (UAC); abra PowerShell como administrador y ejecute:'
                Write-Host '    cytv servicios parar' -ForegroundColor Cyan
                return
            }
            try { Stop-Service -Name $Servicio -Force }
            catch { Write-Host ('  Error al parar: ' + $_.Exception.Message) -ForegroundColor Red; return }
            [void](EsperarEstado 'Stopped')
        }
        'arrancar' {
            if (-not (Test-Admin)) {
                Write-Host '  Requiere elevacion (UAC); abra PowerShell como administrador y ejecute:'
                Write-Host '    cytv servicios arrancar' -ForegroundColor Cyan
                return
            }
            try { Start-Service -Name $Servicio }
            catch { Write-Host ('  Error al arrancar: ' + $_.Exception.Message) -ForegroundColor Red; return }
            [void](EsperarEstado 'Running')
        }
        'reiniciar' {
            if (-not (Confirmar 'Reiniciar el servicio ahora?')) { Write-Host '  Cancelado.'; return }
            if (-not (Test-Admin)) {
                Write-Host '  Requiere elevacion (UAC); abra PowerShell como administrador y ejecute:'
                Write-Host '    cytv servicios reiniciar' -ForegroundColor Cyan
                return
            }
            try { Restart-Service -Name $Servicio -Force }
            catch { Write-Host ('  Error al reiniciar: ' + $_.Exception.Message) -ForegroundColor Red; return }
            [void](EsperarEstado 'Running')
        }
        default {
            Write-Host '  Acciones: cytv servicios parar | arrancar | reiniciar [-Fuerza]'
        }
    }
}

function Invoke-Config {
    Write-Titulo
    Write-Host "  ${ConfigIni}:"
    Write-Host ''
    Get-Content $ConfigIni -Encoding UTF8 | ForEach-Object { Write-Host "  $_" }
    if ($Abrir) {
        Start-Process notepad.exe $ConfigIni
    }
}

function Invoke-Web {
    Write-Titulo
    if (Test-Tcp 3000) {
        Start-Process 'http://localhost:3000'
        Write-Host '  Abriendo http://localhost:3000 ...'
        return
    }
    Write-Host '  La app web no esta corriendo en el puerto 3000.'
    $resp = Read-Host '  ¿Iniciarla ahora? (s/N)'
    if ($resp -in @('s', 'si', 'sí')) {
        Start-Process node -ArgumentList 'server.js' -WorkingDirectory (Join-Path $RutaBase 'webapp') -WindowStyle Hidden
        Start-Sleep -Seconds 2
        if (Test-Tcp 3000) {
            Start-Process 'http://localhost:3000'
            Write-Host '  App web iniciada; abriendo el navegador.' -ForegroundColor Green
        } else {
            Write-Host '  No arranco; revise que exista webapp\server.js y node.' -ForegroundColor Red
        }
    } else {
        Write-Host '  Iniciela despues con webapp\abrir_app.bat'
    }
}

function Invoke-Limpiar {
    $script = Join-Path $RutaBase 'scripts\limpiar_almacenamiento.ps1'
    Write-Titulo
    if (-not (Test-Path $script)) {
        Write-Host "  No existe el script de limpieza: $script" -ForegroundColor Red
        return
    }
    Write-Host '  Ejecutando limpieza manual (retencion 14 dias)...'
    Write-Host ''
    & powershell -NoProfile -ExecutionPolicy Bypass -File $script
    Write-Host ''
    Write-Host '  Listo. Revise el detalle en C:\CytvAncon\logs\limpieza.log'
    Write-Host '  La tarea automatica corre cada 2 semanas (domingo 3:00 AM).'
}

function Invoke-Ayuda {
    Write-Titulo
    Write-Host @'
  Comandos del servidor de placas CYTV Ancon
  -------------------------------------------
  cytv estado                 Servicio, FTP, web, modelo y placas de hoy
  cytv placas                 Placas detectadas hoy (reporte JSON)
  cytv historial              Ultimos 15 registros de la base SQLite
  cytv log [-Lineas 50]       Ultimas lineas del log del servidor
  cytv fotos RUTA [-Canal]    Sube una foto de prueba (01 pano / 02 placa)
  cytv servicios [reiniciar]  Estado o reinicio del servicio (Fuerza = sin preguntar)
  cytv config [-Abrir]        Ver el config.ini (o abrirlo en el bloc)
  cytv web                    Abrir/iniciar la app web (localhost:3000)
  cytv limpiar                Borrar imagenes con mas de 14 dias (manual)
  cytv ayuda                  Esta ayuda

  La limpieza automatica corre cada 2 semanas (domingo 3:00 AM) via
  Task Scheduler; "cytv limpiar" la ejecuta a mano cuando quiera.

  Las fotos rechazadas por el worker se borran; para revisar por que una
  placa no guardo, busque su nombre en "cytv log" o "cytv historial".
'@
}

# --------------------------------------------------------------------------
# Despacho
# --------------------------------------------------------------------------
switch ($Comando) {
    'estado'    { Invoke-Estado }
    'placas'    { Invoke-Placas }
    'historial' { Invoke-Historial }
    'log'       { Invoke-Log }
    'fotos'     { Invoke-Fotos }
    'servicios' { Invoke-Servicios }
    'reiniciar' { $Ruta = 'reiniciar'; Invoke-Servicios }
    'config'    { Invoke-Config }
    'web'       { Invoke-Web }
    'limpiar'   { Invoke-Limpiar }
    'ayuda'     { Invoke-Ayuda }
}
Write-Host ''
