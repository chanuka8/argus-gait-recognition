<#
.SYNOPSIS
    ARGUS AI - Automatic Python virtual environment activation.

.DESCRIPTION
    Activates the project venv when a terminal is opened inside
    the ARGUS AI repository. Designed to be launched via:

        powershell -NoExit -ExecutionPolicy Bypass -File activate_venv.ps1

    Safety guarantees:
      - Skips activation when the correct venv is already active.
      - Deactivates a foreign venv before activating the project venv.
      - Warns but keeps the terminal open when the venv or interpreter
        is missing.
      - Performs no network, testing, linting, compilation, Git, or
        package-installation operations.
#>

$ErrorActionPreference = 'Stop'

# Resolve repository root: parent directory of scripts/
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)

# Start terminal in the repository root
Set-Location -LiteralPath $RepoRoot

# Derive virtual environment paths
$VenvDir = Join-Path $RepoRoot 'venv'
$ActivateScript = Join-Path $VenvDir 'Scripts\Activate.ps1'
$PythonExe = Join-Path $VenvDir 'Scripts\python.exe'

# Validate venv directory
if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Warning "[ARGUS] Virtual environment not found: $VenvDir"
    Write-Warning "[ARGUS] Create it with: python -m venv `"$VenvDir`""
    return
}

# Validate activation script
if (-not (Test-Path -LiteralPath $ActivateScript -PathType Leaf)) {
    Write-Warning "[ARGUS] Activation script not found: $ActivateScript"
    return
}

# Validate Python interpreter
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Warning "[ARGUS] Python interpreter not found: $PythonExe"
    return
}

$TargetVenv = [System.IO.Path]::GetFullPath($VenvDir)

# Skip only when this exact environment is already fully activated
if ($global:__ARGUS_VENV_ACTIVATED -and $env:VIRTUAL_ENV) {
    try {
        $CurrentVenv = [System.IO.Path]::GetFullPath($env:VIRTUAL_ENV)

        if ($CurrentVenv -eq $TargetVenv) {
            return
        }
    }
    catch {
        # Invalid inherited VIRTUAL_ENV path; continue with normal activation.
    }
}

# Deactivate a different active virtual environment
if ($env:VIRTUAL_ENV) {
    try {
        $CurrentVenv = [System.IO.Path]::GetFullPath($env:VIRTUAL_ENV)

        if ($CurrentVenv -ne $TargetVenv) {
            $DeactivateCommand = Get-Command deactivate -ErrorAction SilentlyContinue

            if ($DeactivateCommand) {
                deactivate
            }
            else {
                Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
            }
        }
    }
    catch {
        Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
    }
}

# Activate the ARGUS virtual environment
try {
    . $ActivateScript

    $global:__ARGUS_VENV_ACTIVATED = $true

    $PythonVersion = (& $PythonExe --version 2>&1).ToString().Trim()

    Write-Host "[ARGUS] venv activated - $PythonVersion" -ForegroundColor Green
}
catch {
    $global:__ARGUS_VENV_ACTIVATED = $false
    Write-Warning "[ARGUS] Activation failed: $($_.Exception.Message)"
}