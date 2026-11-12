# Registra la tarea programada de limpieza de almacenamiento.
# Corre cada 2 semanas (domingo) a las 3:00 AM con privilegios elevados.
# Se lanza elevado (UAC) una sola vez; despues Windows la ejecuta solo.

$ErrorActionPreference = 'Stop'
$tarea = 'CYTV_LimpiezaAlmacenamiento'
$script = 'C:\CytvAncon\limpiar_almacenamiento.ps1'

# Si ya existe, eliminarla para recrearla limpia
$existente = Get-ScheduledTask -TaskName $tarea -ErrorAction SilentlyContinue
if ($existente) {
    Unregister-ScheduledTask -TaskName $tarea -Confirm:$false
    Write-Output "Tarea anterior '$tarea' eliminada."
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""

$trigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval 2 -DaysOfWeek Sunday -At 3:00AM

$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $tarea `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description 'Limpieza automatica de imagenes procesadas y descartadas de la camara CYTV Ancon (retencion 14 dias, cada 2 semanas).' | Out-Null

Write-Output "Tarea '$tarea' registrada correctamente."
Write-Output "  Frecuencia : cada 2 semanas (domingo) 3:00 AM"
Write-Output "  Usuario    : SYSTEM (no requiere login)"
Write-Output "  Script     : $script"
Write-Output "  Retencion  : 14 dias"

# Verificar
$t = Get-ScheduledTask -TaskName $tarea
$t | Format-List TaskName, State
$t.Triggers | Format-List
