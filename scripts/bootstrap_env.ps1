<#
.SYNOPSIS
    ARGUS AI - Production-Grade Automated Environment Bootstrap & Repair.

.DESCRIPTION
    Scans host hardware (NVIDIA GPU, driver, CUDA), inspects the active .venv,
    compares against the environment fingerprint, and automatically verifies or repairs
    the matching PyTorch and ONNX-GPU compute stacks with live, unbuffered visual progress.

.PARAMETER ForceRepair
    Forces re-download and re-installation of compute wheels even if the environment is already healthy.

.PARAMETER InstallRequirements
    Installs additional dependencies from requirements.txt following bootstrap.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/bootstrap_env.ps1
#>

[CmdletBinding()]
param (
    [switch]$ForceRepair,
    [switch]$ForceCpu,
    [switch]$InstallRequirements
)

$ErrorActionPreference = 'Stop'

# Resolve workspace root
if ($PSCommandPath) {
    $ScriptDir = Split-Path -Parent $PSCommandPath
} elseif ($PSScriptRoot) {
    $ScriptDir = $PSScriptRoot
} else {
    $ScriptDir = (Get-Location).Path
}

$RepoRoot = Split-Path -Parent $ScriptDir
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
Set-Location -LiteralPath $RepoRoot

$VenvDir    = Join-Path $RepoRoot '.venv'
$PythonExe  = Join-Path $VenvDir 'Scripts\python.exe'

function Write-Argus([string]$msg, [string]$color = 'Green') {
    Write-Host "[ARGUS] $msg" -ForegroundColor $color
}

# 1. Ensure .venv exists
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Argus "Virtual environment (.venv) not found. Creating fresh .venv..." 'Yellow'
    
    $basePython = 'python'
    try {
        $pyCheck = (& py -3.11 -c "import sys; print(sys.executable)" 2>&1).ToString().Trim()
        if ($LASTEXITCODE -eq 0) { $basePython = $pyCheck }
    } catch {}

    & $basePython -m venv $VenvDir
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Write-Host "[ARGUS ERROR] Failed to create virtual environment at $VenvDir" -ForegroundColor Red
        exit 1
    }
    Write-Argus "Created .venv successfully." 'Green'
}

# 2. Run Production-Grade ARGUS AI Environment Bootstrap Orchestrator
$bootstrapArgs = @("-u", "-m", "automation.bootstrap")
if ($ForceRepair) {
    $bootstrapArgs += "--force-repair"
}
if ($ForceCpu) {
    $bootstrapArgs += "--force-cpu"
}

& $PythonExe $bootstrapArgs
$bootstrapExit = $LASTEXITCODE

if ($InstallRequirements -and (Test-Path "$RepoRoot\requirements.txt")) {
    Write-Argus "Installing additional dependencies from requirements.txt..." 'Cyan'
    & $PythonExe -u -m pip install -r "$RepoRoot\requirements.txt"
}

if ($bootstrapExit -eq 0) {
    exit 0
} else {
    Write-Host "`n[ARGUS ERROR] Environment bootstrap failed with exit code $bootstrapExit." -ForegroundColor Red
    exit 1
}
