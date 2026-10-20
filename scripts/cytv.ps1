  
         
                                                              

            
                                                                    

        
             
             
                
                     
                                              
                          
                      
                         
                    
                          
                                       
              
            
  
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('estado', 'placas', 'historial', 'log', 'fotos',
                 'servicios', 'config', 'ayuda', 'reiniciar', 'limpiar',
                 'entrenador', 'replay', 'heatmap')]
    [string]$Comando = 'ayuda',

    [Parameter(Position = 1)]
    [string]$Ruta,

    [ValidateSet('01', '02')]
    [string]$Canal = '01',

    [int]$Lineas = 30,

    [switch]$Abrir,

    [switch]$Fuerza,

    [string]$Fecha,
    [string]$Motivo,
    [string]$Salida,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$RutaBase = 'C:\CytvAncon'
$Servicio = 'CamaraCytvAncon'
$ConfigIni = Join-Path $RutaBase 'config.ini'

                                                                            
            
                                                                            
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

                                                                            
          
                                                                            
function Invoke-Estado {
    $puertoFtp   = [int](Get-IniValor 'ftp' 'puerto' '2121')
    $servicio    = Get-EstadoServicio
    $ftpOk       = Test-Tcp $puertoFtp
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
                                                                              
                                                                           
                                                                         
    param([string]$a, [string]$b)
    $m = $a.Length; $n = $b.Length
    if ($m -eq 0) { return $n }
    if ($n -eq 0) { return $m }
    $w = $n + 1                                                              
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

function Invoke-Entrenador {
    $py = Join-Path $RutaBase 'venv\Scripts\python.exe'
    $script = Join-Path $RutaBase 'entrenamiento\entrenador_continuo.py'
    $log = Join-Path $RutaBase 'cola_entrenamiento\entrenador.log'
    if (Test-Path $ConfigIni) {
        $cfgTxt = Get-Content -LiteralPath $ConfigIni -Raw
        if ($cfgTxt -match '(?m)^\s*entrenador_python\s*=\s*(\S[^\r\n]*)') {
            $pyEntrenador = $Matches[1].Trim()
            if ($pyEntrenador) { $py = $pyEntrenador }
        }
    }
    Write-Titulo
    $accion = if ($Ruta) { $Ruta.ToLower() } else { 'estado' }
    if ($accion -eq 'arrancar') {
        if (-not (Test-Path $py)) {
            Write-Host "  No se encontro el python del venv: $py" -ForegroundColor Red
            return
        }
        if (-not (Test-Path $script)) {
            Write-Host "  No existe el script: $script" -ForegroundColor Red
            return
        }
        $enEjecucion = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like '*entrenador_continuo.py*' }
        if ($enEjecucion) {
            Write-Host "  Ya esta corriendo (PID $($enEjecucion.ProcessId))."
            return
        }
        Start-Process -FilePath $py -ArgumentList "`"$script`"" -WindowStyle Hidden
        Start-Sleep -Seconds 2
        Write-Host '  Entrenador continuo arrancado.'
        Write-Host '  Entrena solo en el reposo (cola estable y sin entrada nueva)'
        Write-Host '  y solo publica pesos si el examen GT no baja.'
        return
    }
    if (-not (Test-Path $log)) {
        Write-Host "  Sin log todavia: $log"
        return
    }
    Write-Host "  Entrenador continuo (ultimas lineas de $log):"
    Write-Host ''
    Get-Content $log -Tail 15 | ForEach-Object { Write-Host "  $_" }
}

function Invoke-Replay {
                                                                      
    $py = Join-Path $RutaBase 'venv\Scripts\python.exe'
    $script = Join-Path $RutaBase 'scripts\replay_descartes.py'
    Write-Titulo
    if (-not (Test-Path $py)) {
        Write-Host "  No se encontro el python del venv: $py" -ForegroundColor Red
        return
    }
    if (-not (Test-Path $script)) {
        Write-Host "  No existe el script: $script" -ForegroundColor Red
        return
    }
    $argumentos = @($script)
    if ($Fecha)  { $argumentos += @('--fecha', $Fecha) }
    if ($Motivo) { $argumentos += @('--motivo', $Motivo) }
    if ($DryRun) { $argumentos += '--dry-run' }
    Write-Host ''
    & $py @argumentos
}

function Invoke-Heatmap {
                                                                  
    $py = Join-Path $RutaBase 'venv\Scripts\python.exe'
    $script = Join-Path $RutaBase 'scripts\heatmap_placas.py'
    Write-Titulo
    if (-not (Test-Path $py)) {
        Write-Host "  No se encontro el python del venv: $py" -ForegroundColor Red
        return
    }
    if (-not (Test-Path $script)) {
        Write-Host "  No existe el script: $script" -ForegroundColor Red
        return
    }
    $argumentos = @($script)
    if ($Salida) { $argumentos += @('--salida', $Salida) }
    Write-Host ''
    & $py @argumentos
}

function Invoke-Ayuda {
    Write-Titulo
    Write-Host @'
  Comandos del servidor de placas CYTV Ancon
  -------------------------------------------
  cytv estado                 Servicio, FTP, modelo y placas de hoy
  cytv placas                 Placas detectadas hoy (reporte JSON)
  cytv historial              Ultimos 15 registros de la base SQLite
  cytv log [-Lineas 50]       Ultimas lineas del log del servidor
  cytv fotos RUTA [-Canal]    Sube una foto de prueba (01 pano / 02 placa)
  cytv replay [-Fecha dia] [-Motivo motivo] [-DryRun]
                              Devuelve a la cola las fotos de descartadas
  cytv entrenador [arrancar]  Entrenamiento continuo en el reposo: con
                              "arrancar" lo pone a correr; sin argumento
                              muestra el log (estado actual)
  cytv heatmap [-Salida a.csv]  Banda espacial tipica de las placas
  cytv servicios [reiniciar]  Estado o reinicio del servicio (Fuerza = sin preguntar)
  cytv config [-Abrir]        Ver el config.ini (o abrirlo en el bloc)
  cytv limpiar                Borrar imagenes con mas de 14 dias (manual)
  cytv ayuda                  Esta ayuda

  La limpieza automatica corre cada 2 semanas (domingo 3:00 AM) via
  Task Scheduler; "cytv limpiar" la ejecuta a mano cuando quiera.

  descartadas\<dia>\ SOLO guarda duplicados de placa (purga diaria
  7:00 PM); lo demas se reevalua hasta 10 veces y se borra si no
  se lee. Los rescatados entran como PLACA (DUDOSA) con recuadro
  verde fino.
  Cada registro nuevo trae evidencia (0-100) y semaforo (verde/amarillo/
  rojo); revise "cytv historial".
'@
}

                                                                            
          
                                                                            
switch ($Comando) {
    'estado'     { Invoke-Estado }
    'placas'     { Invoke-Placas }
    'entrenador' { Invoke-Entrenador }
    'historial'  { Invoke-Historial }
    'log'        { Invoke-Log }
    'fotos'      { Invoke-Fotos }
    'servicios'  { Invoke-Servicios }
    'reiniciar'  { $Ruta = 'reiniciar'; Invoke-Servicios }
    'config'     { Invoke-Config }
    'replay'     { Invoke-Replay }
    'heatmap'    { Invoke-Heatmap }
    'limpiar'    { Invoke-Limpiar }
    'ayuda'      { Invoke-Ayuda }
}
Write-Host ''
