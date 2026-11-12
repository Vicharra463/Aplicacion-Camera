#requires -Version 3.0
# ============================================================
#  Instalador GRAFICO - Servidor FTP Camara CYTV Ancon
#
#  Ventana WinForms que DELEGA en instalar.ps1 (una sola fuente
#  de verdad de la logica de instalacion). Muestra el archivo
#  instalacion.log en vivo y, al terminar bien, ofrece abrir la
#  consola de control (cytv) para reiniciar / parar / arrancar
#  el servicio.
#
#  Compatible con PowerShell 3.0+ (Server 2012 R2 trae PS4).
#  Lo eleva el .bat que lo invoca; si se corre directo sin
#  administrador, al instalar Windows pedira UAC.
#
#  Uso:
#    powershell -ExecutionPolicy Bypass -File instalar_gui.ps1
#    powershell -ExecutionPolicy Bypass -File instalar_gui.ps1 -SelfTest
# ============================================================
param(
    [switch]$SelfTest
)

$ErrorActionPreference = 'Continue'
$raiz = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$dest = 'C:\CytvAncon'
$log  = Join-Path $raiz 'instalacion.log'
$ps1  = Join-Path $raiz 'instalar.ps1'

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# --- Elevacion ---
$identidad = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identidad)
$esAdmin   = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# --- Ventana y controles ---
$form = New-Object System.Windows.Forms.Form
$form.Text = 'Instalador - Servidor FTP Camara CYTV Ancon'
$form.Size = New-Object System.Drawing.Size(800, 540)
$form.MinimumSize = New-Object System.Drawing.Size(640, 400)
$form.StartPosition = 'CenterScreen'
$form.Font = New-Object System.Drawing.Font('Segoe UI', 9)

$lblIntro = New-Object System.Windows.Forms.Label
$lblIntro.Location = New-Object System.Drawing.Point(12, 10)
$lblIntro.Size = New-Object System.Drawing.Size(760, 34)
$lblIntro.Text = 'Instala todo automaticamente en C:\CytvAncon: Python, dependencias, Media Foundation (si hace falta),' +
                 ' servicio Windows y firewall. El detalle aparece abajo en vivo.'
$lblIntro.AutoSize = $false

$txtLog = New-Object System.Windows.Forms.TextBox
$txtLog.Location = New-Object System.Drawing.Point(12, 50)
$txtLog.Size = New-Object System.Drawing.Size(760, 360)
$txtLog.Multiline = $true
$txtLog.ReadOnly = $true
$txtLog.ScrollBars = 'Both'
$txtLog.WordWrap = $false
$txtLog.Font = New-Object System.Drawing.Font('Consolas', 9)
$txtLog.BackColor = [System.Drawing.Color]::FromArgb(20, 20, 20)
$txtLog.ForeColor = [System.Drawing.Color]::FromArgb(0, 220, 0)

$lblEstado = New-Object System.Windows.Forms.Label
$lblEstado.Location = New-Object System.Drawing.Point(12, 420)
$lblEstado.Size = New-Object System.Drawing.Size(640, 22)
if ($esAdmin) {
    $lblEstado.Text = 'Listo para instalar (permiso de administrador OK).'
} else {
    $lblEstado.Text = 'AVISO: sin permiso de administrador - Windows pedira UAC al instalar.'
    $lblEstado.ForeColor = [System.Drawing.Color]::FromArgb(160, 100, 0)
}

$btnInstalar = New-Object System.Windows.Forms.Button
$btnInstalar.Location = New-Object System.Drawing.Point(12, 450)
$btnInstalar.Size = New-Object System.Drawing.Size(160, 36)
$btnInstalar.Text = 'Instalar ahora'

$btnConsola = New-Object System.Windows.Forms.Button
$btnConsola.Location = New-Object System.Drawing.Point(184, 450)
$btnConsola.Size = New-Object System.Drawing.Size(240, 36)
$btnConsola.Text = 'Abrir consola de control (cytv)'
$btnConsola.Enabled = $false

$btnCerrar = New-Object System.Windows.Forms.Button
$btnCerrar.Location = New-Object System.Drawing.Point(672, 450)
$btnCerrar.Size = New-Object System.Drawing.Size(100, 36)
$btnCerrar.Text = 'Cerrar'

$form.Controls.AddRange(@($lblIntro, $txtLog, $lblEstado, $btnInstalar, $btnConsola, $btnCerrar))

# --- Estado del proceso hijo (instalar.ps1) ---
$script:proc        = $null
$script:lineasLeidas = 0
$script:codigoSalida = 1

function Agregar-LogLineas {
    if (-not (Test-Path -LiteralPath $log)) { return }
    try { $todas = @(Get-Content -LiteralPath $log -Encoding UTF8) } catch { return }
    if ($todas.Count -gt $script:lineasLeidas) {
        $nuevas = $todas[$script:lineasLeidas..($todas.Count - 1)]
        foreach ($l in $nuevas) { [void]$txtLog.AppendText($l + [Environment]::NewLine) }
        $script:lineasLeidas = $todas.Count
        $txtLog.SelectionStart = $txtLog.TextLength
        $txtLog.ScrollToCaret()
    }
}

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 400

$timer.Add_Tick({
    Agregar-LogLineas
    if ($script:proc -and $script:proc.HasExited) {
        $timer.Stop()
        Agregar-LogLineas
        $script:codigoSalida = $script:proc.ExitCode
        if ($script:codigoSalida -eq 0) {
            $lblEstado.Text = 'INSTALACION EXITOSA - abra la consola de control para administrar el servicio.'
            $lblEstado.ForeColor = [System.Drawing.Color]::FromArgb(0, 140, 0)
            $btnConsola.Enabled = $true
        } else {
            $lblEstado.Text = ('La instalacion fallo (codigo ' + $script:codigoSalida + ') - revise el log de arriba.')
            $lblEstado.ForeColor = [System.Drawing.Color]::FromArgb(180, 0, 0)
        }
        $btnInstalar.Enabled = $true
        $btnInstalar.Text = 'Reintentar instalacion'
    }
})

$btnInstalar.Add_Click({
    $btnInstalar.Enabled = $false
    $lblEstado.Text = 'Instalando... puede tardar varios minutos (descargas).'
    $lblEstado.ForeColor = [System.Drawing.Color]::FromArgb(0, 0, 0)
    if (-not (Test-Path -LiteralPath $ps1)) {
        [void]$txtLog.AppendText(('ERROR: falta ' + $ps1) + [Environment]::NewLine)
        [void]$txtLog.AppendText('Copie la carpeta COMPLETA del proyecto y vuelva a correrlo.' + [Environment]::NewLine)
        $lblEstado.Text = 'Falta instalar.ps1 junto a este archivo.'
        $btnInstalar.Enabled = $true
        return
    }
    $argumentos = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $ps1)
    if ($esAdmin) {
        $script:proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentos -PassThru -WindowStyle Hidden
    } else {
        $script:proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentos -PassThru -Verb RunAs -WindowStyle Hidden
    }
    $timer.Start()
})

$btnConsola.Add_Click({
    # Ruta ABSOLUTA: el PATH del sistema pudo actualizarse recien en esta
    # misma instalacion y este proceso no lo hereda todavia.
    $cytv = Join-Path $dest 'cytv.cmd'
    if (-not (Test-Path -LiteralPath $cytv)) {
        [void]$txtLog.AppendText(('AVISO: no se encontro ' + $cytv) + [Environment]::NewLine)
        return
    }
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoExit', '-NoProfile', '-Command', ("& '{0}' ayuda" -f $cytv)
    )
})

$btnCerrar.Add_Click({ $form.Close() })
$form.Add_FormClosed({
    $timer.Dispose()
    exit $script:codigoSalida
})

if ($SelfTest) {
    Write-Output ('SELFTEST OK - controles construidos; esAdmin=' + $esAdmin + '; log=' + $log)
    exit 0
}

[System.Windows.Forms.Application]::Run($form)
