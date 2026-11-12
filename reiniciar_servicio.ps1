# Reinicia el servicio NSSM de la camara CYTV Ancon.
# Se lanza elevado (UAC) desde cytv.ps1 o manualmente.
$svc = 'CamaraCYTVAncon'
Write-Output "Reiniciando servicio $svc..."
Restart-Service -Name $svc -Force
Start-Sleep -Seconds 2
$st = Get-Service -Name $svc
Write-Output "Estado: $($st.Status)"
