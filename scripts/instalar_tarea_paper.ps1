<#
.SYNOPSIS
    Registra (o quita) la tarea diaria del Programador de tareas de Windows que avanza el paper interno.

.DESCRIPTION
    Responsabilidad: crear la tarea "Sistema_trading paper interno", que ejecuta `uv run python -m ypyquant.paper avanzar`
    con el repositorio como directorio de trabajo y deja la salida en `data/paper_interno/avance.log` (se añade al final).
    El avance es idempotente y hace catch-up de todos los días cerrados pendientes, así que ningún disparador es crítico:
      - cada día a la hora indicada (por defecto 23:30 hora local: el cierre de EE. UU. es a las 22:00 en Madrid y se deja
        margen para que yfinance consolide la barra);
      - al iniciar sesión (recupera lo perdido si el PC estaba apagado a esa hora).
    Ajustes: `StartWhenAvailable` (si se perdió la hora, se ejecuta en cuanto sea posible), `RunOnlyIfNetworkAvailable`,
    sin límite de batería y un máximo de 2 horas de ejecución. `-Despertar` añade `WakeToRun`.

    NO se ejecuta nada al crear el archivo: lo lanza el usuario a mano. Con `-SoloMostrar` solo imprime lo que haría.
    Para quitar la tarea: `-Desinstalar`. Para ver el log: `Get-Content data\paper_interno\avance.log -Tail 50`.
    Se relaciona con: src/ypyquant/paper/__main__.py (la orden `avanzar`) y docs/ui.md (sección «Paper interno»).

.PARAMETER Hora
    Hora local del disparador diario, formato HH:mm (por defecto 23:30).

.PARAMETER Despertar
    Permite que la tarea despierte al equipo del reposo para ejecutarse.

.PARAMETER SoloMostrar
    Imprime la definición de la tarea sin registrarla.

.PARAMETER Desinstalar
    Elimina la tarea registrada.

.EXAMPLE
    .\scripts\instalar_tarea_paper.ps1 -SoloMostrar
    .\scripts\instalar_tarea_paper.ps1 -Hora 23:45
    .\scripts\instalar_tarea_paper.ps1 -Desinstalar
#>
[CmdletBinding()]
param(
    [string]$Hora = '23:30',
    [switch]$Despertar,
    [switch]$SoloMostrar,
    [switch]$Desinstalar
)

$ErrorActionPreference = 'Stop'
$NombreTarea = 'Sistema_trading paper interno'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$CarpetaLog = Join-Path $Repo 'data\paper_interno'
$Log = Join-Path $CarpetaLog 'avance.log'

if ($Desinstalar) {
    if (Get-ScheduledTask -TaskName $NombreTarea -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $NombreTarea -Confirm:$false
        Write-Host "Tarea '$NombreTarea' eliminada."
    } else {
        Write-Host "La tarea '$NombreTarea' no existe."
    }
    return
}

$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) { throw "No se encuentra 'uv' en el PATH: instálalo o ábrelo desde una consola donde funcione." }

# powershell -Command: crea la carpeta del log, anota la hora de inicio y añade la salida (stdout y stderr) al log
$Orden = "New-Item -ItemType Directory -Force -Path '$CarpetaLog' | Out-Null; " +
         "Add-Content -Path '$Log' -Value ('=== ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ==='); " +
         "& '$Uv' run python -m ypyquant.paper avanzar *>> '$Log'; exit `$LASTEXITCODE"
$Accion = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $Repo `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command `"$Orden`""
$Disparadores = @(
    (New-ScheduledTaskTrigger -Daily -At $Hora),
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME)
)
$Ajustes = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew
if ($Despertar) { $Ajustes.WakeToRun = $true }

Write-Host "Tarea:          $NombreTarea"
Write-Host "Directorio:     $Repo"
Write-Host "Disparadores:   cada día a las $Hora y al iniciar sesión"
Write-Host "Orden:          $Uv run python -m ypyquant.paper avanzar"
Write-Host "Log:            $Log"
Write-Host "Despertar PC:   $($Despertar.IsPresent)"

if ($SoloMostrar) {
    Write-Host "`n(-SoloMostrar: no se ha registrado nada)"
    return
}

Register-ScheduledTask -TaskName $NombreTarea -Action $Accion -Trigger $Disparadores -Settings $Ajustes `
    -Description 'Avanza las sesiones de paper interno de Sistema_trading (catch-up de los días cerrados pendientes).' -Force | Out-Null
Write-Host "`nTarea registrada. Pruébala con: Start-ScheduledTask -TaskName '$NombreTarea'"
