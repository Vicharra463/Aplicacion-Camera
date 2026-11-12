# Limpieza automatica de almacenamiento - Camara CYTV Ancon
# Borra imagenes procesadas y descartadas con mas de RETENCION_DIAS dias.
# Programado con Task Scheduler para correr cada 2 semanas (ver README).
#
# Uso manual:  powershell -ExecutionPolicy Bypass -File limpiar_almacenamiento.ps1
# Log:         C:\CytvAncon\logs\limpieza.log

$ErrorActionPreference = 'Stop'

# ---------- Configuracion ----------
$RETENCION_DIAS = 14          # borra todo lo anterior a N dias
$RUTAS = @(
    'C:\CytvAncon\imagenes_procesadas',
    'C:\CytvAncon\descartadas',
    'C:\CytvAncon\ftp_recibidos'
)
$LOG = 'C:\CytvAncon\logs\limpieza.log'
# -----------------------------------

function Escribir-Log([string]$msg) {
    $linea = "{0} [LIMPIEZA] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $LOG -Value $linea -Encoding UTF8
    Write-Output $linea
}

$corte = (Get-Date).AddDays(-$RETENCION_DIAS)
$totalBorrados = 0
$totalMB = 0.0

Escribir-Log "=== Inicio de limpieza (retencion: $RETENCION_DIAS dias, corte: $($corte.ToString('yyyy-MM-dd'))) ==="

foreach ($ruta in $RUTAS) {
    if (-not (Test-Path $ruta)) {
        Escribir-Log "  Omitida (no existe): $ruta"
        continue
    }

    # 1) Borrar archivos sueltos mas viejos que el corte
    $archivos = Get-ChildItem $ruta -File -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -lt $corte }
    foreach ($f in $archivos) {
        $totalMB += $f.Length / 1MB
        Remove-Item $f.FullName -Force
        $totalBorrados++
    }
    if ($archivos.Count -gt 0) {
        Escribir-Log ("  {0}: {1} archivos sueltos borrados" -f $ruta, $archivos.Count)
    }

    # 2) Borrar subcarpetas de fecha (YYYY-MM-DD) antiguas
    #    Solo borra carpetas cuyo nombre parezca fecha y cuyo contenido sea todo viejo.
    $carpetasFecha = Get-ChildItem $ruta -Directory -ErrorAction SilentlyContinue |
                     Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' -and $_.Name -lt $corte.ToString('yyyy-MM-dd') }
    foreach ($carp in $carpetasFecha) {
        $size = (Get-ChildItem $carp.FullName -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object Length -Sum).Sum
        $totalMB += if ($size) { $size / 1MB } else { 0 }
        $count = (Get-ChildItem $carp.FullName -Recurse -File -ErrorAction SilentlyContinue).Count
        Remove-Item $carp.FullName -Recurse -Force
        $totalBorrados += $count
        Escribir-Log ("  {0}: carpeta {1} borrada ({2} archivos, {3:N1} MB)" -f $ruta, $carp.Name, $count, ($size/1MB))
    }

    # 3) Borrar carpetas vacias residuales antiguas
    $vacias = Get-ChildItem $ruta -Directory -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' -and $_.Name -lt $corte.ToString('yyyy-MM-dd') }
    foreach ($v in $vacias) {
        Remove-Item $v.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 4) Compactar registro.db si crecio mucho (opcional, solo informa)
$db = 'C:\CytvAncon\registro.db'
if (Test-Path $db) {
    $dbMB = [math]::Round((Get-Item $db).Length / 1MB, 2)
    Escribir-Log "  registro.db actual: $dbMB MB"
}

Escribir-Log ("=== Fin: {0} archivos borrados, {1:N1} MB liberados ===" -f $totalBorrados, $totalMB)

# Mantener el log de limpieza acotado (ultimas 500 lineas)
if ((Get-Content $LOG -ErrorAction SilentlyContinue).Count -gt 500) {
    Get-Content $LOG -Tail 500 | Set-Content $LOG -Encoding UTF8
}
