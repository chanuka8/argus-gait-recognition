<#
.SYNOPSIS
    ARGUS AI - Backward-compatibility shim for scripts/activate_venv.ps1.
    Forwards execution to tools/activate_venv.ps1.
#>
$target = Join-Path $PSScriptRoot "..\tools\activate_venv.ps1"
& $target
