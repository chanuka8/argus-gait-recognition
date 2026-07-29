<#
.SYNOPSIS
    ARGUS AI — Automatic Python virtual environment activation.

.DESCRIPTION
    Activates the project venv when a terminal is opened inside
    the ARGUS AI repository.  Designed to be launched via:

        powershell -NoExit -ExecutionPolicy Bypass -File activate_venv.ps1

    Safety guarantees:
      • Skips activation when the correct venv is already active in session.
      • Deactivates a foreign venv before activating the project one.
      • Warns (but stays open) when the venv directory or interpreter
        is missing.
      • Performs NO network, testing, linting, compilation, git, or
        package-installation operations.
#>

# ── Resolve repository root (parent of the scripts/ directory) ──────────
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

# Normalise to a fully-qualified path so string comparisons are reliable.
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)

# Set the working directory to the repository root.
Set-Location -LiteralPath $RepoRoot

# ── Derive venv paths ───────────────────────────────────────────────────
$VenvDir        = Join-Path $RepoRoot 'venv'
$ActivateScript = Join-Path $VenvDir  'Scripts\Activate.ps1'
$PythonExe      = Join-Path $VenvDir  'Scripts\python.exe'

# ── Guard: venv directory must exist ────────────────────────────────────
if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Warning "[ARGUS] Virtual environment not found at: $VenvDir"
    Write-Warning "[ARGUS] Create it with:  python -m venv `"$VenvDir`""
    return   # terminal stays open (-NoExit)
}

# ── Guard: Activate.ps1 must exist ─────────────────────────────────────
if (-not (Test-Path -LiteralPath $ActivateScript -PathType Leaf)) {
    Write-Warning "[ARGUS] Activation script missing: $ActivateScript"
    return
}

# ── Guard: python.exe must exist ───────────────────────────────────────
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Warning "[ARGUS] Python interpreter missing: $PythonExe"
    return
}

# ── Check if already activated in current PowerShell session ─────────────
$NormTarget = [System.IO.Path]::GetFullPath($VenvDir)

if ($global:__ARGUS_VENV_ACTIVATED -and $env:VIRTUAL_ENV) {
    $NormCurrent = [System.IO.Path]::GetFullPath($env:VIRTUAL_ENV)
    if ($NormCurrent -eq $NormTarget) {
        # Already fully activated in this session
        return
    }
}

# ── If a different venv is active, deactivate it first ──────────────────
if ($env:VIRTUAL_ENV) {
    $NormCurrent = [System.IO.Path]::GetFullPath($env:VIRTUAL_ENV)
    if ($NormCurrent -ne $NormTarget) {
        if (Get-Command deactivate -ErrorAction SilentlyContinue) {
            deactivate
        }
    }
}

# ── Activate ────────────────────────────────────────────────────────────
try {
    . $ActivateScript
    $global:__ARGUS_VENV_ACTIVATED = $true
    $PyVer = (& $PythonExe --version 2>&1).ToString().Replace('Python ', '')
    Write-Host ('[ARGUS] venv activated  —  Python ' + $PyVer) -ForegroundColor Green
}
catch {
    Write-Warning ('[ARGUS] Activation failed: ' + $_.Exception.Message)
}
