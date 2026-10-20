                                                             
                                                                        
                                                                       

$ErrorActionPreference = 'Stop'
$tarea = 'CYTV_LimpiezaAlmacenamiento'
$script = 'C:\CytvAncon\scripts\limpiar_almacenamiento.ps1'

                                                
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

                                                                            
                                                                           
                                                                     
                                                                            
$tareaDiaria = 'CYTV_DescartadasDiaria'

$existenteDiaria = Get-ScheduledTask -TaskName $tareaDiaria -ErrorAction SilentlyContinue
if ($existenteDiaria) {
    Unregister-ScheduledTask -TaskName $tareaDiaria -Confirm:$false
    Write-Output "Tarea anterior '$tareaDiaria' eliminada."
}

$accionDiaria = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -DescartadasDiaria"

$triggerDiaria = New-ScheduledTaskTrigger -Daily -At 19:00

Register-ScheduledTask -TaskName $tareaDiaria `
    -Action $accionDiaria `
    -Trigger $triggerDiaria `
    -Principal $principal `
    -Settings $settings `
    -Description 'Purga de duplicados CYTV Ancon: descartadas solo contiene duplicados y se borran todos los dias a las 7:00 PM.' | Out-Null

Write-Output "Tarea '$tareaDiaria' registrada correctamente."
Write-Output "  Frecuencia : diaria 7:00 PM (duplicados)"
Write-Output "  Usuario    : SYSTEM (no requiere login)"
Write-Output "  Script     : $script -DescartadasDiaria"

           
Write-Output ''
Write-Output 'Estado de las tareas:'
Get-ScheduledTask -TaskName $tarea, $tareaDiaria -ErrorAction SilentlyContinue |
    Format-List TaskName, State
(Get-ScheduledTask -TaskName $tarea -ErrorAction SilentlyContinue).Triggers | Format-List
(Get-ScheduledTask -TaskName $tareaDiaria -ErrorAction SilentlyContinue).Triggers | Format-List
