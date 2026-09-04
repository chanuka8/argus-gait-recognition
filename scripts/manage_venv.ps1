<#
.SYNOPSIS
    ARGUS AI - Backward-compatibility shim for scripts/manage_venv.ps1.
    Forwards parameters to tools/manage_venv.ps1.
#>
[CmdletBinding()]
param (
    [ValidateSet('Recreate', 'Clean', 'StopProcesses', 'Status', 'InstallRequirements')]
    [string]$Action = 'Recreate',
    [switch]$InstallRequirements,
    [switch]$StopVsCodeWorkers,
    [string]$BasePython = '',
    [switch]$Force
)

$target = Join-Path $PSScriptRoot "..\tools\manage_venv.ps1"
& $target -Action $Action -InstallRequirements:$InstallRequirements -StopVsCodeWorkers:$StopVsCodeWorkers -BasePython $BasePython -Force:$Force
