#requires -Version 3.0
param([switch]$Check)
                                                              
                                                        
                                                               
                                                    
                                                              
                                                             
 
                                                                 
                                                               
                                                           
                                                             
                                                                   
                                                                  
                                           
                                                             
                                 
 
                                                                       
                                                              
$ErrorActionPreference = 'Continue'
$raiz = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
                                                                            
                                                                           
                                                       
if (-not (Test-Path -LiteralPath (Join-Path $raiz 'ftp_server.py'))) {
    $padre = Split-Path -Parent $raiz
    if (Test-Path -LiteralPath (Join-Path $padre 'ftp_server.py')) { $raiz = $padre }
}
$dest = 'C:\CytvAncon'
$serv = 'CamaraCytvAncon'

function L($m) {
                                                                   
                                                                            
    $c = [ConsoleColor]::Gray
    if ($m -match '(^|\s)OK(\s|$)|EXITO|INSTALACION TERMINADA|RUNNING') { $c = [ConsoleColor]::Green }
    if ($m -match 'AVISO|OCUPADO|pendiente') { $c = [ConsoleColor]::Yellow }
    if ($m -match 'ERROR|FALTA') { $c = [ConsoleColor]::Red }
    if ($m -match '^\s*Paso ') { $c = [ConsoleColor]::Cyan }
    Write-Host ('{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m) -ForegroundColor $c
}

                                                                               
function Get-PidEnPuerto($puerto) {
    $pat = ':' + $puerto + ' '
    foreach ($ln in (& netstat -ano -p tcp)) {
        if ($ln -match 'LISTENING' -and $ln -match $pat) {
            $toks = ($ln -split '\s+') | Where-Object { $_ -ne '' }
            try { return [int]$toks[$toks.Count - 1] } catch { return $null }
        }
    }
    return $null
}

                                                                    
function Find-PythonBase($lo, $hi) {
    $dirs = @()
    foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($pf) {
            $dirs += Get-ChildItem -LiteralPath $pf -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like 'Python3*' }
        }
    }
    foreach ($d in $dirs) {
        $exe = Join-Path $d.FullName 'python.exe'
        if (Test-Path -LiteralPath $exe) {
            $v = & $exe -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))" 2>$null
            if ($v -match '^(\d+)\.(\d+)$') {
                $maj = [int]$Matches[1]; $mn = [int]$Matches[2]
                $num = $maj * 100 + $mn
                if ($num -ge $lo -and ($hi -eq 0 -or $num -le $hi)) {
                    return $exe
                }
            }
        }
    }
    return $null
}

                                                                          
                                                                           
if ($Check) {
    L '============================================================'
    L ' Diagnostico - no modifica nada'
    L '============================================================'
    L ('Carpeta:   ' + $dest)

    $idC = [Security.Principal.WindowsIdentity]::GetCurrent()
    $prC = New-Object Security.Principal.WindowsPrincipal($idC)
    if ($prC.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        L 'OK     Elevacion: SI - ya se ejecuta como administrador'
    } else {
        L 'AVISO  Elevacion: NO - al instalar Windows pedira permiso'
    }

    $cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction SilentlyContinue
    $b = 0; try { $b = [int]$cv.CurrentBuild } catch { $b = 0 }
    L ('Sistema:    ' + $cv.ProductName + ' build ' + $cv.CurrentBuild)
    if ($b -lt 9600)      { L 'ERROR  Soporte:    NO - se necesita 8.1 / Server 2012 R2 o superior (64 bits)' }
    elseif ($b -lt 14393) { L 'OK     Soporte:    SI - se instalara Python 3.9' }
    else                  { L 'OK     Soporte:    SI - se instalara Python 3.13' }

    $pyHallado = $false
    foreach ($pexe in @('C:\Program Files\Python313\python.exe', 'C:\Program Files\Python39\python.exe')) {
        if (Test-Path -LiteralPath $pexe) {
            L ('OK     Python:     presente (' + (Split-Path -Parent $pexe) + ')')
            $pyHallado = $true
        }
    }
    if (-not $pyHallado) { L 'FALTA  Python:     no encontrado - lo instala el instalador' }

    if (Test-Path -LiteralPath (Join-Path $dest 'scripts\nssm.exe')) {
        L ('OK     NSSM:       presente en ' + $dest + '\scripts')
    } else {
        L 'FALTA  NSSM:       falta - va junto al proyecto (scripts\)'
    }

    if (Test-Path -LiteralPath (Join-Path $dest 'ftp_server.py')) {
        L ('OK     Proyecto:   instalado en ' + $dest)
    } else {
        L 'FALTA  Proyecto:   aun no instalado'
    }

    $svc = Get-Service -Name $serv -ErrorAction SilentlyContinue
    if (-not $svc)                     { L 'FALTA  Servicio:   no instalado' }
    elseif ($svc.Status -eq 'Running') { L ('OK     Servicio:   ' + $svc.Status) }
    else                               { L ('FALTA  Servicio:   ' + $svc.Status) }

    $pid2121 = Get-PidEnPuerto 2121
    if (-not $pid2121) { L 'OK     Puerto 2121: libre' }
    else { L ('AVISO  Puerto 2121: OCUPADO (PID ' + $pid2121 + ' - ver con netstat -ano ^| findstr :2121)') }

    & netsh advfirewall firewall show rule name='CYTV Ancon FTP' > $null 2>&1
    if ($LASTEXITCODE -eq 0) { L 'OK     Firewall:   regla presente' }
    else { L 'FALTA  Firewall:   sin regla - se crea al instalar' }

    exit 0
}

L '============================================================'
L ' Instalador universal - Servidor FTP Camara CYTV Ancon'
L '============================================================'

                           
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$pr = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    L 'ERROR: hace falta elevacion. Corra instalar_servicio.bat (pide permiso solo).'
    exit 1
}

                                    
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    L "ERROR: se necesita Windows de 64 bits (detectado: $env:PROCESSOR_ARCHITECTURE)."
    exit 1
}
$cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction SilentlyContinue
$build = 0
try { $build = [int]$cv.CurrentBuild } catch { }
$prod = $cv.ProductName
L "Sistema detectado: $prod (build $build)"
if ($build -lt 9600) {
    L 'ERROR: se necesita Windows 8.1 / Server 2012 R2 o superior de 64 bits.'
    L '       (En sistemas mas viejos el Python 3.9+ ya no corre.)'
    exit 1
}
if ($build -lt 14393) {
    $pyVer = '3.9.13'
    $pyUrl = 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe'
    $req   = 'requirements-py39.txt'
    $loNum = 309; $hiNum = 309
    L 'Ruta: Windows viejo -> Python 3.9.13 + dependencias pineadas'
} else {
    $pyVer = '3.13.14'
    $pyUrl = 'https://www.python.org/ftp/python/3.13.14/python-3.13.14-amd64.exe'
    $req   = 'requirements.txt'
    $loNum = 310; $hiNum = 0
    L 'Ruta: Windows moderno -> Python 3.13.14 + requirements.txt'
}

                               
$pid2121 = Get-PidEnPuerto 2121
if ($pid2121) {
    $dueno = Get-CimInstance Win32_Process -Filter "ProcessId=$pid2121" -ErrorAction SilentlyContinue
    $nom = if ($dueno) { $dueno.Name } else { 'desconocido' }
    L "ERROR: el puerto 2121 ya lo usa otro proceso (PID $pid2121, $nom)."
    L '       Detengalo primero o cambie el puerto en config.ini.'
    exit 1
}
L 'Puerto 2121: libre'

                                          
if ($dest -like ($env:UserProfile + '\*')) {
    L "ERROR: el destino ($dest) no puede vivir en el perfil de usuario:"
    L '       el servicio corre como SYSTEM y SYSTEM no ejecuta .exe de C:\Users.'
    exit 1
}
if ($raiz -ne $dest) {
    L "Paso 3: copiando proyecto $raiz -> $dest"
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    robocopy $raiz $dest /E /XD venv __pycache__ logs ftp_recibidos imagenes_procesadas descartadas `
        /R:1 /W:1 /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { L "ERROR: robocopy fallo (codigo $LASTEXITCODE)"; exit 1 }
    L '       Copia terminada (sin venv, sin imagenes de maquinas anteriores)'
} else {
    L "Paso 3: ya se corre desde $dest - no hace falta copiar"
}
                                                             
Remove-Item -LiteralPath (Join-Path $dest 'instalacion.log') -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $PSScriptRoot 'instalacion.log') -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path (Join-Path $dest 'logs') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dest 'ftp_recibidos') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dest 'imagenes_procesadas') -Force | Out-Null

                                                        
$cfg = Join-Path $dest 'config.ini'
if (-not (Test-Path -LiteralPath $cfg)) {
    L "ERROR: falta config.ini en $dest - copie el proyecto completo."
    exit 1
}
L ('Paso 4: config.ini -> rutas absolutas de ' + $dest)
$map = @{
    'recibidos'  = (Join-Path $dest 'ftp_recibidos')
    'procesadas' = (Join-Path $dest 'imagenes_procesadas')
    'registro'   = (Join-Path $dest 'registro.db')
    'logs'       = (Join-Path $dest 'logs')
    'heartbeat'  = (Join-Path $dest 'heartbeat.txt')
    'descartes'  = (Join-Path $dest 'descartadas')
    'reportes'   = (Join-Path $dest 'reportes')
}
$lines = @(Get-Content -LiteralPath $cfg)
for ($i = 0; $i -lt $lines.Count; $i++) {
    foreach ($k in $map.Keys) {
        if ($lines[$i] -match ('^\s*' + $k + '\s*=')) { $lines[$i] = ($k + ' = ' + $map[$k]) }
    }
}
[IO.File]::WriteAllLines($cfg, $lines, (New-Object System.Text.UTF8Encoding($false)))
L '       OK'

                                             
L ('Paso 5: localizando/instalando Python ' + $pyVer)
$pyBase = Find-PythonBase $loNum $hiNum
if (-not $pyBase) {
    L '       No esta - descargando desde python.org (puede tardar)'
    $setup = Join-Path $env:TEMP ("python-" + $pyVer + "-amd64.exe")
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $pyUrl -OutFile $setup -TimeoutSec 600
        Unblock-File -LiteralPath $setup -ErrorAction SilentlyContinue
    } catch {
        L ('       ERROR de descarga: ' + $_.Exception.Message)
        L '       Si no hay internet, descarge el instalador a mano y coloquelo en:'
        L ('       ' + $setup)
        exit 1
    }
    $p = Start-Process -FilePath $setup -ArgumentList '/quiet','InstallAllUsers=1','PrependPath=0','AssociateFiles=0','Shortcuts=0','Include_test=0' -Wait -PassThru
    L ('       Instalador python exit=' + $p.ExitCode)
    $pyBase = Find-PythonBase $loNum $hiNum
}
if (-not $pyBase) {
    L '       ERROR: no se instalo el Python esperado.'
    exit 1
}
L ('       Python base: ' + $pyBase)

                                                               
                                                                         
                                                                         
                                                                        
                                                                           
                                                                           
                                                                       
                                       
if ($build -lt 14393) {
    L 'Paso 5.5: Windows viejo - verificando Visual C++ Runtime (VS2015/2019)'
    $sys32 = Join-Path $env:SystemRoot 'System32'
    $dllsCrt = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll')
    $ucrtDll = Join-Path $sys32 'api-ms-win-crt-runtime-l1-1-0.dll'
    $faltaCrt  = @($dllsCrt  | Where-Object { -not (Test-Path -LiteralPath (Join-Path $sys32 $_)) })
    $faltaUcrt = -not (Test-Path -LiteralPath $ucrtDll)
    if (-not $faltaCrt -and -not $faltaUcrt) {
        L '       VC++ Runtime completo ya presente - se omite'
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        if ($faltaUcrt) {
                                                                        
                                                    
            L '       Falta Universal CRT - instalando VC++ 2015 Update 3...'
            $url15 = 'https://download.microsoft.com/download/6/A/A/6AA4EDFF-645B-48C5-81CC-ED5963AEAD48/vc_redist.x64.exe'
            $exe15 = Join-Path $env:TEMP 'vc_redist_2015u3_x64.exe'
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $url15 -OutFile $exe15 -TimeoutSec 300
                Unblock-File -LiteralPath $exe15 -ErrorAction SilentlyContinue
            } catch {
                L ('       ERROR de descarga VC++ 2015: ' + $_.Exception.Message)
                L '       Descarguelo a mano (id=53840, vc_redist.x64.exe), ejecutelo'
                L '       y vuelva a correr este instalador.'
                exit 1
            }
            $p = Start-Process -FilePath $exe15 -ArgumentList '/install','/quiet','/norestart' -Wait -PassThru
            L ('       VC++ 2015 U3 exit=' + $p.ExitCode + ' (0 o 3010 = OK)')
            $faltaCrt = @($dllsCrt | Where-Object { -not (Test-Path -LiteralPath (Join-Path $sys32 $_)) })
        }
        if ($faltaCrt.Count -gt 0) {
                                                                               
                                                                      
            L ('       Faltan: ' + ($faltaCrt -join ', ') + ' - instalando VC++ VS2019 Runtime...')
            $url19 = 'https://aka.ms/vs/16/release/vc_redist.x64.exe'
            $exe19 = Join-Path $env:TEMP 'vc_redist_vs2019_x64.exe'
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $url19 -OutFile $exe19 -TimeoutSec 300
                Unblock-File -LiteralPath $exe19 -ErrorAction SilentlyContinue
            } catch {
                L ('       ERROR de descarga VC++ VS2019: ' + $_.Exception.Message)
                L '       Descarguelo a mano desde https://aka.ms/vs/16/release/vc_redist.x64.exe'
                L '       y vuelva a correr este instalador.'
                exit 1
            }
            $p = Start-Process -FilePath $exe19 -ArgumentList '/install','/quiet','/norestart' -Wait -PassThru
            L ('       VC++ VS2019 exit=' + $p.ExitCode + ' (0 o 3010 = OK)')
        }
                                            
        $aunFaltan = @()
        foreach ($d in $dllsCrt) { if (-not (Test-Path -LiteralPath (Join-Path $sys32 $d))) { $aunFaltan += $d } }
        if (-not (Test-Path -LiteralPath $ucrtDll)) { $aunFaltan += 'api-ms-win-crt-runtime-l1-1-0.dll' }
        if ($aunFaltan.Count -gt 0) {
            L ('       ERROR: siguen faltando DLLs: ' + ($aunFaltan -join ', '))
            L '       Instale los vc_redist a mano y vuelva a correr este instalador.'
            exit 1
        }
        L '       VC++ Runtime completo instalado OK'
    }
}

                                                     
                                                                        
                                                                    
                                                                          
                                                                            
                                       
if ($build -lt 14393) {
    $mfplat = Join-Path $env:SystemRoot 'System32\mfplat.dll'
    if (Test-Path -LiteralPath $mfplat) {
        L 'Paso 5.6: Media Foundation ya presente - se omite'
    } else {
        L 'Paso 5.6: habilitando feature Media Foundation con DISM...'
                                                                           
                                                                              
                                                                                
                                                          
                                                                         
                                                                      
                               
        $dism = Join-Path $env:SystemRoot 'System32\dism.exe'
        $candidatos = @('ServerMediaFoundation', 'Server-Media-Foundation', 'MediaFoundation')
        $featOk = $null
        foreach ($c in $candidatos) {
            L ('       probando feature: ' + $c)
            & $dism /Online /Enable-Feature ('/FeatureName:' + $c) /All /NoRestart
            $exitDism = $LASTEXITCODE
            if (($exitDism -eq 0) -or ($exitDism -eq 3010)) { $featOk = $c; break }
            L ('       "' + $c + '" no existe aqui (exit ' + $exitDism + ') - se prueba el siguiente')
        }
        if (-not $featOk) {
            L '       ERROR: ningun nombre de feature funciono.'
            L '       Liste los nombres reales con:'
            L '       dism /Online /Get-Features | findstr /i Media'
            exit 1
        }
        L ('       feature habilitada: ' + $featOk + ' (exit ' + $exitDism + '; 3010 = reinicio pendiente)')
        if (Test-Path -LiteralPath $mfplat) {
            L '       Media Foundation habilitado OK'
        } elseif ($exitDism -eq 3010) {
            L '       Media Foundation quedo pendiente de REINICIO (codigo 3010).'
            L '       NO corta la instalacion: requirements-py39 fija opencv-python-headless,'
            L '       que carga sin Media Foundation. Despues del reinicio mfplat.dll quedara'
            L '       disponible y el instalador lo detectara (paso 5.6).'
        } else {
            L '       AVISO: DISM no habilito Media Foundation y no pide reinicio.'
            L '       Sigue igual - el headless no la necesita. Revise despues con:'
            L '       dism /Online /Enable-Feature /FeatureName:ServerMediaFoundation /All'
        }
    }
}

                                 
$vp = Join-Path $dest 'venv\Scripts\python.exe'
$venvOk = $false
if (Test-Path -LiteralPath $vp) {
                           
    $v = & $vp -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))" 2>$null
    $n = 0
    if ($v -match '^(\d+)\.(\d+)$') { $n = [int]$Matches[1] * 100 + [int]$Matches[2] }
    if ($n -ge $loNum -and ($hiNum -eq 0 -or $n -le $hiNum)) {
                                                                          
                                                                    
        $smoke = & $vp -c "import cv2, numpy, onnxruntime, pyftpdlib; print('OK')" 2>&1 | Out-String
        if ($smoke -match 'OK') {
            $venvOk = $true
            L 'Paso 6: venv existente compatible y funcional - se reutiliza'
        } else {
            L 'Paso 6: venv existente pero ROTO (imports fallan) - se recrea'
            L ('       detalle: ' + ($smoke -replace '\s+', ' ').Trim())
        }
    } else {
        L ('Paso 6: venv existente con Python ' + $v + ' - incompatible, se recrea')
    }
} else {
    L 'Paso 6: no hay venv - se crea'
}
if (-not $venvOk) {
    L ('       creando venv limpio en ' + $dest)
    if (Test-Path -LiteralPath (Join-Path $dest 'venv')) {
        Remove-Item -LiteralPath (Join-Path $dest 'venv') -Recurse -Force -ErrorAction SilentlyContinue
    }
    & $pyBase -m venv (Join-Path $dest 'venv') 2>&1 | ForEach-Object { L ('       venv: ' + $_) }
}
if (-not (Test-Path -LiteralPath $vp)) { L '       ERROR: no se pudo crear el venv.'; exit 1 }

$reqPath = Join-Path $dest $req
if (-not (Test-Path -LiteralPath $reqPath)) { L ("ERROR: falta $req en $dest"); exit 1 }

                                                                       
                                                                       
                                                                      
                                                                       
                                                              
if ($req -eq 'requirements-py39.txt') {
    $di = Join-Path $dest 'venv\Lib\site-packages\opencv_python-4.5.3.56.dist-info'
    New-Item -ItemType Directory -Path $di -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $di 'METADATA') -Value "Metadata-Version: 2.1`nName: opencv-python`nVersion: 4.5.3.56" -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $di 'INSTALLER') -Value 'manual' -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $di 'RECORD') -Value '' -Encoding ASCII
    L '       metadato falso opencv-python 4.5.3.56 listo (pip no instalara el normal)'
}

L ('       pip install -r ' + $req + ' (varios minutos)')
& $vp -m pip install --disable-pip-version-check --quiet -r $reqPath 2>&1 |
    Select-Object -Last 5 | ForEach-Object { L ('       pip: ' + $_) }
if ($LASTEXITCODE -ne 0) { L '       ERROR: fallo pip. Revise internet/config.'; exit 1 }

                                                                        
                                                                 
                                                                    
                                                                        
if ($req -eq 'requirements-py39.txt') {
    $sp = Join-Path $dest 'venv\Lib\site-packages'
    $malos = Get-ChildItem -LiteralPath $sp -Directory -Filter 'opencv_python-*' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne 'opencv_python-4.5.3.56.dist-info' }
    if ($malos) {
        L ('       AVISO: pip instalo opencv-python normal (' + (($malos | ForEach-Object { $_.Name }) -join ', ') + ') - se corrige al headless')
        & $vp -m pip uninstall -y opencv-python 2>&1 | Out-Null
        & $vp -m pip install --disable-pip-version-check --quiet 'opencv-python-headless==4.5.3.56' 2>&1 | Out-Null
    }
}

L '       prueba de humo: imports criticos'
$humo = & $vp -c "import pyftpdlib, cv2, numpy, onnxruntime; from rapidocr_onnxruntime import RapidOCR; print('imports OK')" 2>&1 | Out-String
                    if ($humo -notmatch 'imports OK') {
                        L '       ERROR: los imports no cargan - el venv o el sistema estan incompletos.'
                        L ('       detalle: ' + ($humo -replace '\s+', ' ').Trim())
                        if ($build -lt 14393) {
                                if ($humo -match 'MFPlat|Media Foundation' -or
                                -not (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\mfplat.dll'))) {
                                L '       Parece faltar Media Foundation - en DISM el nombre de feature es'
                                L '       ServerMediaFoundation (sin guiones):'
                                L '       dism /Online /Enable-Feature /FeatureName:ServerMediaFoundation /All'
                                L '       (si ese nombre no existe, liste los reales con'
                                L '        dism /Online /Get-Features | findstr /i Media)'
                            }
                            if ($humo -match 'vcruntime|msvcp|DLL load failed' -and
                                -not (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\vcruntime140_1.dll'))) {
                                L '       Parece faltar el VC++ VS2019 Runtime - corra los pasos 5.5 otra vez'
                                L '       o instale https://aka.ms/vs/16/release/vc_redist.x64.exe a mano.'
                            }
                        } else {
                            L '       Instale Visual C++ Redistributable 2015-2022 y vuelva a correr.'
                            L '       https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist'
                        }
                        exit 1
                    }
                    L ('       ' + ($humo -replace '\s+', ' ').Trim())
                    if (-not (Test-Path -LiteralPath (Join-Path $dest 'modelo\placas.onnx'))) {
                        L '       AVISO: falta modelo\placas.onnx - arrancara SIN detector de placas.'
                    }

                                                                       
                                                                                  
                                                                                 
                                                     
                    $cytvCmd = Join-Path $dest 'cytv.cmd'
                    if (-not (Test-Path -LiteralPath $cytvCmd)) {
                        L '       AVISO: falta cytv.cmd - el comando cytv no estara disponible.'
                    } else {
                        L 'Paso 6.5: registrando comando cytv en el PATH del sistema'
                        $mp = [Environment]::GetEnvironmentVariable('Path', 'Machine')
                        $parts = @()
                        if ($mp) { $parts = @($mp -split ';') | Where-Object { $_ -ne '' } }
                        $destNorm = $dest.TrimEnd('\')
                        $presente = $false
                        foreach ($p in $parts) {
                            if ($p.TrimEnd('\') -ieq $destNorm) { $presente = $true }
                        }
                        if ($presente) {
                            L '       la carpeta ya esta en el PATH - se omite'
                        } else {
                            $nuevo = ($parts -join ';') + ';' + $dest
                            [Environment]::SetEnvironmentVariable('Path', $nuevo, 'Machine')
                            $verif = [Environment]::GetEnvironmentVariable('Path', 'Machine')
                            if ($verif -match 'CytvAncon') {
                                L '       OK: ' + $dest + ' agregado al PATH del sistema'
                                L '       (abra una consola NUEVA para que cytv responda)'
                            } else {
                                L '       AVISO: no se pudo agregar al PATH - a mano, elevado:'
                                L ('       setx /M PATH "%PATH%;' + $dest + '"')
                            }
                        }
                    }

                                                 
$nssm = Join-Path $dest 'scripts\nssm.exe'
if (-not (Test-Path -LiteralPath $nssm)) { L '       AVISO: falta nssm.exe en la carpeta - el servicio no se puede registrar.'; exit 1 }
L ('Paso 7: registrando servicio ' + $serv)
$existe = Get-Service -Name $serv -ErrorAction SilentlyContinue
if ($existe) { Stop-Service -Name $serv -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2 }
if (-not $existe) {
    & $nssm install $serv $vp (Join-Path $dest 'ftp_server.py') 2>&1 | ForEach-Object { L ('       nssm: ' + $_) }
}
& $nssm set $serv Application $vp 2>&1 | Out-Null
& $nssm set $serv AppDirectory $dest 2>&1 | Out-Null
                                                                                      
$par = 'HKLM:\SYSTEM\CurrentControlSet\Services\' + $serv + '\Parameters'
Set-ItemProperty -LiteralPath $par -Name AppDirectory -Value $dest -Type String
Set-ItemProperty -LiteralPath $par -Name AppParameters -Value ('"' + (Join-Path $dest 'ftp_server.py') + '"') -Type String
& $nssm set $serv DisplayName 'Servidor FTP Camara CYTV Ancon' 2>&1 | Out-Null
& $nssm set $serv Description 'Recibe las fotos de la camara Hikvision, las archiva por fecha y registra las placas' 2>&1 | Out-Null
& $nssm set $serv Start SERVICE_AUTO_START 2>&1 | Out-Null
& $nssm set $serv AppExit Default Restart 2>&1 | Out-Null
& $nssm set $serv AppRestartDelay 5000 2>&1 | Out-Null
& $nssm set $serv AppStdout (Join-Path $dest 'logs\stdout.log') 2>&1 | Out-Null
& $nssm set $serv AppStderr (Join-Path $dest 'logs\stderr.log') 2>&1 | Out-Null
& $nssm set $serv AppRotateFiles 1 2>&1 | Out-Null
$regAppDir  = (Get-ItemProperty -LiteralPath $par -Name AppDirectory -ErrorAction SilentlyContinue).AppDirectory
$regAppArgs = (Get-ItemProperty -LiteralPath $par -Name AppParameters -ErrorAction SilentlyContinue).AppParameters
L ('       verif AppDirectory  = ' + $regAppDir)
L ('       verif AppParameters = ' + $regAppArgs)
if ($regAppDir -ne $dest) { L '       ERROR: AppDirectory quedo mal.'; exit 1 }

                      
L 'Paso 8: firewall'
$puerto = '2121'; $pas = '60000-60099'
foreach ($ln in (Get-Content -LiteralPath $cfg)) {
    if ($ln -match '^\s*puerto\s*=\s*(\d+)') { $puerto = $Matches[1] }
    if ($ln -match '^\s*puertos_pasivos\s*=\s*([\d\-]+)') { $pas = $Matches[1] }
}
& netsh advfirewall firewall delete rule name='CYTV Ancon FTP' | Out-Null
& netsh advfirewall firewall add rule name='CYTV Ancon FTP' dir=in action=allow protocol=TCP localport="$puerto,$pas" | Out-Null
if ($LASTEXITCODE -ne 0) {
    L ('       AVISO: no se pudo crear la regla. Abra a mano TCP ' + $puerto + ' y ' + $pas)
} else {
    L ('       OK: TCP ' + $puerto + ' y pasivos ' + $pas)
}

                                           
L 'Paso 9: iniciando servicio...'
Start-Service -Name $serv -ErrorAction SilentlyContinue
Start-Sleep -Seconds 15
$st = (Get-Service -Name $serv).Status
L ('       estado del servicio: ' + $st)
$pidFinal = Get-PidEnPuerto $puerto
if ($pidFinal) {
    L ('       EXITO: FTP escuchando en 0.0.0.0:' + $puerto + ' (PID ' + $pidFinal + ')')
    L '============================================================'
    L ' INSTALACION TERMINADA'
    L ' Servicio: services.msc -> "Servidor FTP Camara CYTV Ancon"'
    L (' Log:      ' + (Join-Path $dest 'logs\servidor.log'))
    L (' Info para la camara (IP, usuario): ' + (Join-Path $dest 'INFO_CAMARA.txt'))
    L ' La IP se regenera en cada arranque - apunte la camara ahi.'
    L '============================================================'
    exit 0
} else {
    L '       ERROR: el servicio no levanto el FTP. Tail de stderr.log:'
    Get-Content -LiteralPath (Join-Path $dest 'logs\stderr.log') -Tail 6 -ErrorAction SilentlyContinue |
        ForEach-Object { L ('       ! ' + $_) }
    exit 1
}
