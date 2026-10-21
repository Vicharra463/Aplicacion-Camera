# diag_recursos.ps1 - Diagnostico de recursos del servidor CamaraCytvAncon
# Compatible con Windows Server 2012 R2 (PowerShell 4.0).
# Ejecutar en ELEVADO (Run as administrator). No escribe nada, solo imprime.
#
# USO:
#   1) Correlo con el servicio DETENIDO, apunta la salida
#   2) Start-Service CamaraCytvAncon
#   3) Correlo de nuevo inmediatamente (mientras arranca) y compara

$ErrorActionPreference = 'SilentlyContinue'
$svc  = 'CamaraCytvAncon'
$raiz = 'C:\CytvAncon'

Write-Output ('=' * 70)
Write-Output " DIAGNOSTICO  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output ('=' * 70)

# ---------------------------------------------------------------- 1. HARDWARE
Write-Output "`n--- 1. HARDWARE ---"
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$cores = [int]$cs.NumberOfLogicalProcessors
Write-Output ("CPU           : {0}" -f $cpu.Name)
Write-Output ("Nucleos logicos: {0}  |  carga actual: {1}%" -f $cores, $cpu.LoadPercentage)
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("RAM total     : {0:N1} GB | libre: {1:N1} GB" -f ($os.TotalVisibleMemorySize/1MB), ($os.FreePhysicalMemory/1MB))
Write-Output ("Mem. virtual  : libre {0:N1} GB de {1:N1} GB (page file)" -f (($os.FreeVirtualMemory - $os.FreePhysicalMemory)/1MB), ($os.TotalVirtualMemorySize/1MB))
Get-CimInstance Win32_DiskDrive | ForEach-Object {
    Write-Output ("Disco         : {0} [{1}] {2:N0} GB" -f $_.Model, $_.InterfaceType, ($_.Size/1GB))
}
try {
    Get-PhysicalDisk | ForEach-Object {
        Write-Output ("  tipo medio : {0} ({1})" -f $_.MediaType, $_.FriendlyName)
    }
} catch { Write-Output "  tipo medio : no disponible (Get-PhysicalDisk)" }

# ------------------------------------------------------- 2. ESTADO DEL SERVICIO
Write-Output "`n--- 2. SERVICIO ---"
$s = Get-Service -Name $svc -ErrorAction SilentlyContinue
if ($s) { Write-Output ("Estado       : {0}" -f $s.Status) } else { Write-Output "Estado       : NO INSTALADO" }
$sig = Get-CimInstance Win32_Service -Filter "Name='$svc'"
if ($sig) {
    Write-Output ("PID actual   : {0}" -f $sig.ProcessId)
    $p = Get-Process -Id $sig.ProcessId -ErrorAction SilentlyContinue
    if ($p) { Write-Output ("RAM proceso  : {0:N0} MB | CPU acumulada {1:N0}s | hilos {2}" -f ($p.WorkingSet64/1MB), $p.CPU, $p.Threads.Count) }
}
$log = Join-Path $raiz 'logs\servidor.log'
if (Test-Path $log) {
    $arranques = @(Select-String -Path $log -Pattern 'starting FTP server' -ErrorAction SilentlyContinue)
    Write-Output ("Arranques en servidor.log: {0}" -f $arranques.Count)
    if ($arranques.Count -gt 0) {
        $ultimo = $arranques[-1].Line
        Write-Output ("Ultimo      : {0}" -f $ultimo)
        if ($ultimo -match 'pid=(\d+)') {
            $pidLog = $matches[1]
            if ($sig -and $pidLog -ne [string]$sig.ProcessId) {
                Write-Output "  *** ALERTA: el PID del log NO coincide con el PID actual -> REINICIOS EN LOOP (NSSM AppExit Restart) ***"
            }
        }
    }
}
Write-Output ("stderr.log   :")
Get-Content (Join-Path $raiz 'logs\stderr.log') -Tail 15 -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Output ("    {0}" -f $_) }

# ---------------------------------------------------- 3. EVENTOS DEL SERVICIO
Write-Output "`n--- 3. EVENTOS RECIENTES DEL SERVICIO (System) ---"
Get-EventLog -LogName System -Newest 300 -ErrorAction SilentlyContinue |
    Where-Object { $_.Source -eq 'Service Control Manager' -and $_.Message -like "*$svc*" } |
    Select-Object -Last 10 |
    ForEach-Object { Write-Output ("  {0}  ev{1}  {2}" -f $_.TimeGenerated, $_.EventID, ($_.Message -split "`n")[0]) }

# --------------------------------------------------------- 4. CARGA DEL PIPELINE
Write-Output "`n--- 4. DATA DEL PIPELINE ---"
function Contar-Fotos($ruta) {
    if (-not (Test-Path $ruta)) { return -1 }
    @(Get-ChildItem $ruta -Recurse -File -Include *.jpg,*.jpeg,*.png -ErrorAction SilentlyContinue).Count
}
$desc = Join-Path $raiz 'descartadas'
$nDesc = Contar-Fotos $desc
if ($nDesc -lt 0) {
    Write-Output "descartadas  : NO EXISTE (modo_descarte = eliminar)"
} else {
    Write-Output ("descartadas  : {0} fotos  <-- backlog del REPLAY AL ARRANCAR" -f $nDesc)
    Get-ChildItem $desc -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $n = Contar-Fotos $_.FullName
        Write-Output ("    {0}: {1}" -f $_.Name, $n)
    }
}
$nProc = Contar-Fotos (Join-Path $raiz 'imagenes_procesadas')
Write-Output ("procesadas   : {0} fotos" -f $nProc)
$nRec = Contar-Fotos (Join-Path $raiz 'ftp_recibidos')
Write-Output ("recibidos    : {0} fotos pendientes en cola de entrada" -f $nRec)
$db = Get-Item (Join-Path $raiz 'registro.db') -ErrorAction SilentlyContinue
if ($db) {
    Write-Output ("registro.db  : {0:N1} MB" -f ($db.Length/1MB))
    try {
        $nFilas = (Get-Item $db).Length
        Write-Output ("  (tamanio en disco - crece sin poda)" -f $nFilas)
    } catch {}
}
$ini = Join-Path $raiz 'config.ini'
if (Test-Path $ini) {
    Write-Output "config.ini   :"
    Select-String -Path $ini -Pattern 'modo_descarte|replay_descartes|replay_pausa|nivel' -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Output ("    {0}" -f $_.Line.Trim()) }
}

# ------------------------------------------------------- 5. CPU EN VIVO (15 s)
Write-Output "`n--- 5. CPU POR PROCESO (muestreo 15 s) ---"
Write-Output "    (NucleosUsados = nucleos logicos ocupados; 1.00 = un nucleo al 100%)"
$antes = @{}
Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
    try { $antes[[int]$_.Id] = $_.CPU } catch {}
}
$qDisco1 = $null
try { $qDisco1 = (Get-Counter '\PhysicalDisk(_Total)\Current Disk Queue Length' -ErrorAction Stop).CounterSamples.CookedValue } catch {}
Start-Sleep -Seconds 15
$qDisco2 = $null
try { $qDisco2 = (Get-Counter '\PhysicalDisk(_Total)\Current Disk Queue Length' -ErrorAction Stop).CounterSamples.CookedValue } catch {}

$filas = @()
Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
    $id = [int]$_.Id
    if ($antes.ContainsKey($id)) {
        try {
            $delta = $_.CPU - $antes[$id]
            if ($delta -gt 0.1) {
                $filas += [pscustomobject]@{
                    Proceso       = $_.Name
                    PID           = $id
                    NucleosUsados = [math]::Round($delta / 15, 2)
                    CPU_Pct       = [math]::Round(($delta / 15) * 100 / [math]::Max($cores,1), 1)
                    RAM_MB        = [math]::Round($_.WorkingSet64 / 1MB, 0)
                }
            }
        } catch {}
    }
}
$filas | Sort-Object NucleosUsados -Descending | Select-Object -First 12 |
    Format-Table -AutoSize

$python = $filas | Where-Object { $_.Proceso -like 'python*' } |
    Measure-Object -Property NucleosUsados -Sum
Write-Output ("PYTHON en total: {0} nucleos ocupados de {1} ({2:P0} de la CPU)" -f `
    [math]::Round($python.Sum,2), $cores, (($python.Sum)/[math]::Max($cores,1)))
if ($qDisco2 -ne $null) {
    Write-Output ("Cola de disco promedio: {0:N2}  (>= 2 = disco saturado, TODA la maquina frena)" -f (($qDisco1 + $qDisco2)/2))
}

# ------------------------------------------------- 6. SI HAY TORCH/ENTRENAMIENTO
Write-Output "`n--- 6. PROCESOS SOSPECHOSOS (torch/entrenamiento/otras IAs) ---"
Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'python|torch|yolo|train|ollama|nvidia' } |
    ForEach-Object { Write-Output ("    {0}  pid={1}  RAM={2:N0} MB" -f $_.Name, $_.Id, ($_.WorkingSet64/1MB)) }

Write-Output "`n--- 7. TAREAS PROGRAMADAS RELACIONADAS ---"
schtasks /query /fo LIST 2>$null | Select-String -Pattern 'cytv|camara|camera|entrena' -Context 1,2 |
    ForEach-Object { Write-Output ($_.ToString()) }

Write-Output "`n('=' * 70)"
Write-Output " FIN - pega TODA esta salida tal cual"
Write-Output ('=' * 70)
