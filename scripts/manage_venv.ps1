<#
.SYNOPSIS
    ARGUS AI - Safe Virtual Environment Manager for Windows.

.DESCRIPTION
    Safely resolves Windows file-locking issues (WinError 5 / Access Denied)
    by terminating zombie/locking Python interpreter processes, language server
    workers, and background daemons before modifying or recreating .venv.

.PARAMETER Action
    Operation to perform: 'Recreate' (default), 'Clean', 'StopProcesses', 'Status', 'InstallRequirements'.

.PARAMETER InstallRequirements
    If specified with Recreate, automatically installs dependencies from requirements.txt.

.PARAMETER StopVsCodeWorkers
    Forces termination of background IDE language server/linter workers bound to the workspace venv.

.PARAMETER BasePython
    Optional path or command to the base Python executable (defaults to auto-detection: py -3.11, python3.11, python).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/manage_venv.ps1 -Action Recreate -InstallRequirements
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

$DotVenvDir   = Join-Path $RepoRoot '.venv'
$LegacyVenv   = Join-Path $RepoRoot 'venv'
$LegacyEnv    = Join-Path $RepoRoot 'env'
$ReqFile      = Join-Path $RepoRoot 'requirements.txt'

function Write-ArgusLog([string]$Message, [string]$Color = 'Cyan') {
    Write-Host "[ARGUS VENV] $Message" -ForegroundColor $Color
}

function Write-ArgusWarn([string]$Message) {
    Write-Host "[ARGUS WARN] $Message" -ForegroundColor Yellow
}

function Write-ArgusErr([string]$Message) {
    Write-Host "[ARGUS ERROR] $Message" -ForegroundColor Red
}

function Get-LockingProcesses {
    $matchingPids = @()
    $candidateDirs = @($DotVenvDir, $LegacyVenv, $LegacyEnv)

    try {
        $processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^(python|pythonw|pytest|uvicorn|gunicorn)\.exe$' }

        foreach ($p in $processes) {
            $isMatch = $false
            $exePath = if ($p.ExecutablePath) { $p.ExecutablePath.ToLower() } else { '' }
            $cmdLine = if ($p.CommandLine) { $p.CommandLine.ToLower() } else { '' }

            foreach ($dir in $candidateDirs) {
                $dirLower = $dir.ToLower()
                if ($exePath.StartsWith($dirLower) -or $cmdLine.Contains($dirLower)) {
                    $isMatch = $true
                    break
                }
            }

            if ($StopVsCodeWorkers -and ($cmdLine.Contains("ms-python") -or $cmdLine.Contains("pylance") -or $cmdLine.Contains("lsp_server"))) {
                if ($cmdLine.Contains($RepoRoot.ToLower())) {
                    $isMatch = $true
                }
            }

            if ($isMatch) {
                $matchingPids += [PSCustomObject]@{
                    ProcessId   = $p.ProcessId
                    Name        = $p.Name
                    Path        = $p.ExecutablePath
                    CommandLine = $p.CommandLine
                }
            }
        }
    } catch {
        Write-ArgusWarn "Process inspection encountered a non-fatal error: $($_.Exception.Message)"
    }

    return $matchingPids
}

function Stop-LockingProcesses {
    Write-ArgusLog "Scanning for processes locking virtual environment files..." 'Cyan'

    # Deactivate current session if active
    if ($env:VIRTUAL_ENV) {
        $targetLower = $env:VIRTUAL_ENV.ToLower()
        if ($targetLower.StartsWith($RepoRoot.ToLower())) {
            Write-ArgusLog "Deactivating active environment in current shell session..." 'Yellow'
            $DeactivateCmd = Get-Command deactivate -ErrorAction SilentlyContinue
            if ($DeactivateCmd) { & deactivate }
            Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
        }
    }

    $processes = Get-LockingProcesses
    if ($processes.Count -eq 0) {
        Write-ArgusLog "No locking Python processes detected." 'Green'
        return
    }

    foreach ($proc in $processes) {
        Write-ArgusLog "Terminating process PID $($proc.ProcessId) ($($proc.Name))..." 'Yellow'
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        } catch {}

        # Fallback to taskkill for stubborness
        try {
            Start-Process -FilePath "taskkill.exe" -ArgumentList "/F", "/PID", "$($proc.ProcessId)" -NoNewWindow -Wait -ErrorAction SilentlyContinue | Out-Null
        } catch {}
    }

    # Pause briefly for Windows kernel to release open file handles
    Start-Sleep -Milliseconds 800
    Write-ArgusLog "Locking processes terminated." 'Green'
}

function Remove-DirectorySafely([string]$PathToRemove) {
    if (-not (Test-Path -LiteralPath $PathToRemove)) {
        return
    }

    Write-ArgusLog "Removing folder: $PathToRemove" 'Cyan'

    $maxRetries = 4
    $retryDelay = 1000

    for ($i = 1; $i -le $maxRetries; $i++) {
        try {
            [System.IO.Directory]::Delete($PathToRemove, $true)
            Write-ArgusLog "Successfully deleted: $PathToRemove" 'Green'
            return
        } catch {
            if ($i -lt $maxRetries) {
                Write-ArgusWarn "Folder locked ($($_.Exception.Message)). Attempting to terminate handles (Attempt $i/$maxRetries)..."
                Stop-LockingProcesses
                Start-Sleep -Milliseconds $retryDelay
            } else {
                # Final attempt using cmd rmdir
                try {
                    cmd.exe /c "rmdir /s /q `"$PathToRemove`"" 2>$null
                    if (-not (Test-Path -LiteralPath $PathToRemove)) {
                        Write-ArgusLog "Successfully removed via system rmdir: $PathToRemove" 'Green'
                        return
                    }
                } catch {}

                throw "Failed to delete '$PathToRemove' after $maxRetries attempts: $($_.Exception.Message)"
            }
        }
    }
}

function Find-BasePython {
    if ($BasePython -and (Test-Path -LiteralPath $BasePython)) {
        return $BasePython
    }

    # 1. Try Python Launcher (py -3.11 or py -3)
    try {
        $pyPath = (& py -3.11 -c "import sys; print(sys.executable)" 2>&1).ToString().Trim()
        if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $pyPath)) {
            return $pyPath
        }
    } catch {}

    try {
        $pyPath = (& py -3 -c "import sys; print(sys.executable)" 2>&1).ToString().Trim()
        if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $pyPath)) {
            return $pyPath
        }
    } catch {}

    # 2. Try python from PATH (ensuring it's not inside a venv)
    $systemPythons = Get-Command python.exe -All -ErrorAction SilentlyContinue
    foreach ($cmd in $systemPythons) {
        $src = $cmd.Source
        if ($src -notmatch '\\\.venv\\' -and $src -notmatch '\\venv\\' -and $src -notmatch '\\env\\') {
            return $src
        }
    }

    # 3. Known standard Windows paths
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe",
        "C:\Program Files\Python311\python.exe"
    )

    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            return $c
        }
    }

    return 'python'
}

# --- Action Dispatcher ---

switch ($Action) {
    'Status' {
        Write-ArgusLog "=== ARGUS Virtual Environment Status ===" 'Cyan'
        Write-Host "Repository Root : $RepoRoot"
        Write-Host "Standard .venv  : $(if (Test-Path $DotVenvDir) { 'EXISTS [Active Standard]' } else { 'NOT FOUND' })"
        Write-Host "Legacy venv     : $(if (Test-Path $LegacyVenv) { 'EXISTS [Legacy - Recommend Clean]' } else { 'NOT FOUND' })"
        Write-Host "Legacy env      : $(if (Test-Path $LegacyEnv) { 'EXISTS [Legacy - Recommend Clean]' } else { 'NOT FOUND' })"
        Write-Host "Active Shell Env: $(if ($env:VIRTUAL_ENV) { $env:VIRTUAL_ENV } else { 'None' })"

        $pyInVenv = Join-Path $DotVenvDir 'Scripts\python.exe'
        if (Test-Path -LiteralPath $pyInVenv) {
            $ver = (& $pyInVenv --version 2>&1).ToString().Trim()
            Write-Host "Interpreter     : $pyInVenv ($ver)" -ForegroundColor Green
        }

        $locking = @(Get-LockingProcesses)
        Write-Host "Locking Process : $(if ($locking.Count -gt 0) { "$($locking.Count) process(es) running" } else { '0 processes' })"
        foreach ($lp in $locking) {
            Write-Host "  - PID $($lp.ProcessId): $($lp.Path)" -ForegroundColor Yellow
        }
    }

    'StopProcesses' {
        Stop-LockingProcesses
    }

    'Clean' {
        Stop-LockingProcesses
        Remove-DirectorySafely $DotVenvDir
        Remove-DirectorySafely $LegacyVenv
        Remove-DirectorySafely $LegacyEnv
        Write-ArgusLog "Environment cleanup complete." 'Green'
    }

    'InstallRequirements' {
        $pyInVenv = Join-Path $DotVenvDir 'Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $pyInVenv)) {
            Write-ArgusErr "Cannot install requirements: .venv not found at $DotVenvDir. Run with -Action Recreate first."
            exit 1
        }
        if (-not (Test-Path -LiteralPath $ReqFile)) {
            Write-ArgusWarn "requirements.txt not found at $ReqFile. Skipping installation."
            return
        }

        Write-ArgusLog "Upgrading pip and installing requirements..." 'Cyan'
        & $pyInVenv -m pip install --upgrade pip
        & $pyInVenv -m pip install -r $ReqFile
        Write-ArgusLog "Requirements installed successfully." 'Green'
    }

    'Recreate' {
        Write-ArgusLog "Starting safe virtual environment recreation..." 'Cyan'
        Stop-LockingProcesses

        # Remove all legacy and current venvs to enforce single standard
        Remove-DirectorySafely $DotVenvDir
        Remove-DirectorySafely $LegacyVenv
        Remove-DirectorySafely $LegacyEnv

        $BaseExe = Find-BasePython
        Write-ArgusLog "Using base Python: $BaseExe" 'Cyan'

        Write-ArgusLog "Creating fresh virtual environment at .venv..." 'Cyan'
        & $BaseExe -m venv $DotVenvDir

        $pyInVenv = Join-Path $DotVenvDir 'Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $pyInVenv)) {
            Write-ArgusErr "Virtual environment creation failed: $pyInVenv not found."
            exit 1
        }

        $ver = (& $pyInVenv --version 2>&1).ToString().Trim()
        Write-ArgusLog "Created .venv successfully with $ver" 'Green'

        if ($InstallRequirements -and (Test-Path -LiteralPath $ReqFile)) {
            Write-ArgusLog "Installing requirements from requirements.txt..." 'Cyan'
            & $pyInVenv -m pip install --upgrade pip
            & $pyInVenv -m pip install -r $ReqFile
            Write-ArgusLog "Dependencies installed." 'Green'
        }

        Write-ArgusLog "Virtual environment is ready at: $DotVenvDir" 'Green'
        Write-ArgusLog "Activate in PowerShell with: powershell -ExecutionPolicy Bypass -File scripts/activate_venv.ps1" 'Cyan'
    }
}
