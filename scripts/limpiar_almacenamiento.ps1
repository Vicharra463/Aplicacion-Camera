                                                           
                                                                         
                                                                        
 
                                                                                  
 
                                                                      
                                                                                          
                                                                        
                                                   
 
                                             

param(
    [switch]$DescartadasDiaria
)

$ErrorActionPreference = 'Stop'

                                     
$RETENCION_DIAS = 14                                                         
if ($DescartadasDiaria) {
                                                                             
    $RUTAS = @('C:\CytvAncon\descartadas')
    $corteFecha = (Get-Date).ToString('yyyy-MM-dd')                     
} else {
    $RUTAS = @(
        'C:\CytvAncon\imagenes_procesadas',
        'C:\CytvAncon\descartadas',
        'C:\CytvAncon\ftp_recibidos'
    )
    $corteFecha = (Get-Date).AddDays(-$RETENCION_DIAS).ToString('yyyy-MM-dd')
}
$corte = (Get-Date).AddDays(-$RETENCION_DIAS)
$LOG = 'C:\CytvAncon\logs\limpieza.log'
                                     

function Escribir-Log([string]$msg) {
    $linea = "{0} [LIMPIEZA] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $LOG -Value $linea -Encoding UTF8
    Write-Output $linea
}

$totalBorrados = 0
$totalMB = 0.0
$modo = if ($DescartadasDiaria) { 'fin de jornada (descartadas del dia)' }
        else { "retencion $RETENCION_DIAS dias" }

Escribir-Log "=== Inicio de limpieza ($modo, corte carpeta: $corteFecha) ==="

foreach ($ruta in $RUTAS) {
    if (-not (Test-Path $ruta)) {
        Escribir-Log "  Omitida (no existe): $ruta"
        continue
    }

                                                             
    $archivos = Get-ChildItem $ruta -File -ErrorAction SilentlyContinue |
                Where-Object { $DescartadasDiaria -or $_.LastWriteTime -lt $corte }
    foreach ($f in $archivos) {
        $totalMB += $f.Length / 1MB
        Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
        $totalBorrados++
    }
    if ($archivos.Count -gt 0) {
        Escribir-Log ("  {0}: {1} archivos sueltos borrados" -f $ruta, $archivos.Count)
    }

                                                                    
                                                           
    $carpetasFecha = Get-ChildItem $ruta -Directory -ErrorAction SilentlyContinue |
                     Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' } |
                     Where-Object {
                         if ($DescartadasDiaria) { $_.Name -le $corteFecha }
                         else                    { $_.Name -lt $corteFecha }
                     }
    foreach ($carp in $carpetasFecha) {
        $hijos = Get-ChildItem $carp.FullName -Recurse -File -ErrorAction SilentlyContinue
        $size = ($hijos | Measure-Object Length -Sum).Sum
        $count = $hijos.Count
        $totalMB += if ($size) { $size / 1MB } else { 0 }
                                                                            
                                                                               
        Remove-Item $carp.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $totalBorrados += $count
        if (Test-Path $carp.FullName) {
            Escribir-Log ("  {0}: carpeta {1} parcialmente en uso; se reintenta despues" -f $ruta, $carp.Name)
        } else {
            Escribir-Log ("  {0}: carpeta {1} borrada ({2} archivos, {3:N1} MB)" -f $ruta, $carp.Name, $count, ($size/1MB))
        }
    }
}

                                                 
$db = 'C:\CytvAncon\registro.db'
if (Test-Path $db) {
    $dbMB = [math]::Round((Get-Item $db).Length / 1MB, 2)
    Escribir-Log "  registro.db actual: $dbMB MB"
}

Escribir-Log ("=== Fin: {0} archivos borrados, {1:N1} MB liberados ===" -f $totalBorrados, $totalMB)

                                                          
if ((Get-Content $LOG -ErrorAction SilentlyContinue).Count -gt 500) {
    Get-Content $LOG -Tail 500 | Set-Content $LOG -Encoding UTF8
}
