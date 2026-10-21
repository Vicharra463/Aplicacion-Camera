$log = "C:\Users\felip\OneDrive\Desktop\Motorgas\APP CAMARA\FOTOS PLACA\APP SUBIDA\CytvAncon\entrenamiento\gpu_fix.log"
$ids = "PCI\VEN_10DE&DEV_1F91&SUBSYS_13321025&REV_A1\4&121BEA1D&0&0008"
"=== fix2 inicio $(Get-Date -Format o) ===" | Set-Content $log
pnputil /scan-devices 2>&1 | Out-File -Append $log
Start-Sleep 8
$d = Get-PnpDevice -InstanceId $ids -ErrorAction SilentlyContinue
"scan1: $($d.Status)/$($d.Problem)" | Add-Content $log
if ($d.Status -ne 'OK') {
  "remove + rescan" | Add-Content $log
  pnputil /remove-device $ids 2>&1 | Out-File -Append $log
  Start-Sleep 3
  pnputil /scan-devices 2>&1 | Out-File -Append $log
  Start-Sleep 15
  $d2 = Get-PnpDevice -InstanceId $ids -ErrorAction SilentlyContinue
  "scan2: $($d2.Status)/$($d2.Problem)" | Add-Content $log
}
& "C:\Windows\System32\nvidia-smi.exe" 2>&1 | Out-File -Append $log
"=== fin ===" | Add-Content $log
