# medir_recursos.ps1 v2 - snapshot A/B + TOP10 de consumidores por delta de CPU
# Uso (PowerShell COMO ADMINISTRADOR):
#   .\medir_recursos.ps1 -Etiqueta "A-parado" -Segundos30
#   .\medir_recursos.ps1 -Etiqueta "B-trafico" -Segundos30
param(
    [string]$Etiqueta = "medicion",
    [int]$Segundos =30
)
$ErrorActionPreference = "SilentlyContinue"

# ---- muestra INICIAL: CPU acumulado por PID de TODOS los procesos
$ini = @{}
foreach ($p in (Get-Process)) { $ini[$p.Id] = @{ C = $p.CPU; N = $p.Name } }

# ---- ventana: maxima cola del scheduler
$maxCola = 0
$t0 = Get-Date
do {
    Start-Sleep -Seconds5
    $cola = (Get-CimInstance Win32_PerfFormattedData_PerfOS_System).ProcessorQueueLength
    if ($null -ne $cola -and $cola -gt $maxCola) { $maxCola = $cola }
} while (((Get-Date) - $t0).TotalSeconds -lt $Segundos)
$segReales = ((Get-Date) - $t0).TotalSeconds

# ---- muestra FINAL
$fin = @{}
foreach ($p in (Get-Process)) { $fin[$p.Id] = $p }
$cpuSysIni = ($ini.Values | ForEach-Object { $_.C } | Measure-Object -Sum).Sum
$cpuSysFin = ($fin.Values | ForEach-Object { $_.CPU } | Measure-Object -Sum).Sum

$nucleos = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$os = Get-CimInstance Win32_OperatingSystem
$ramLibre = [int]($os.FreePhysicalMemory / 1KB)
$ramTotal = [int]($os.TotalVisibleMemorySize / 1KB)
$pctSistema = [math]::Round((($cpuSysFin - $cpuSysIni) / ($nucleos * $segReales)) * 100, 1)

Write-Output ("=" * 78)
Write-Output ("ETIQUETA : {0}   (ventana {1:N0}s, {2} nucleos)" -f $Etiqueta, $segReales, $nucleos)
Write-Output ("HORA     : " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Output ("RAM      : {0} MB libres de {1} MB" -f $ramLibre, $ramTotal)
Write-Output ("CPU SISTEMA: {0}% (max {1}%)   COLA CPU MAX: {2}  (> {3} = saturado)" -f $pctSistema, ($nucleos * 100), $maxCola, $nucleos)

# ---- TOP10 por CPU consumida EN LA VENTANA (delta en segundos de CPU / ventana)
$filas = @()
foreach ($id in $fin.Keys) {
    $p = $fin[$id]
    $c0 = $null
    if ($ini.ContainsKey($id)) { $c0 = $ini[$id].C }
    if ($null -eq $c0 -or $null -eq $p.CPU) {
        if ($id -notin $ini.Keys) { $c0 = 0 } else { $c0 = 0 }
    }
    $delta = [double]$p.CPU - [double]$c0
    if ($delta -lt 0) { $delta = 0 }
    $pct1 = [math]::Round(($delta / $segReales) * 100, 1)   # % de UN nucleo
    $ram = [int]($p.WorkingSet64 / 1MB)
    $filas += [PSCustomObject]@{
        NOMBRE = $p.Name
        PID    = $p.Id
        CPU_PCT_UN_NUCLEO = $pct1
        RAM_MB = $ram
        PRIORIDAD = "$($p.PriorityClass)"
    }
}
Write-Output ("-" * 78)
Write-Output "TOP10 PROCESOS POR CPU EN LA VENTANA (CPU_PCT es % de UN nucleo;400 = 4 nucleos enteros):"
$filas | Sort-Object CPU_PCT_UN_NUCLEO -Descending | Select-Object -First10 |
    Format-Table -AutoSize | Out-String -Width100 | ForEach-Object { $_.TrimEnd() } | Write-Output

# ---- foco: los procesos que nos interesan (donde esten y como se llamen)
Write-Output ("-" * 78)
Write-Output "FOCO (sqlservr / python / w3wp / LocalServiceControl):"
$filas | Where-Object { $_.NOMBRE -match 'sqlservr|python|w3wp|LocalService' } |
    Format-Table -AutoSize | Out-String -Width100 | ForEach-Object { $_.TrimEnd() } | Write-Output

# ---- E/S de disco reciente
$io = Get-CimInstance Win32_PerfFormattedData_PerfProc_Process |
    Where-Object { $_.Name -match 'sqlservr|python|w3wp|LocalService' -and ($_.IOReadBytesPerSec -gt 0 -or $_.IOWriteBytesPerSec -gt 0) }
if ($io) {
    Write-Output ("-" * 78)
    Write-Output "E/S DISCO (KB/s promedio reciente):"
    foreach ($x in $io) {
        Write-Output ("{0,-22} leidos {1} KB/s, escritos {2} KB/s" -f $x.Name, [int]($x.IOReadBytesPerSec / 1KB), [int]($x.IOWriteBytesPerSec / 1KB))
    }
}
Write-Output ("=" * 78)
