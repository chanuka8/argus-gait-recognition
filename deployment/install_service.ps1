# ARGUS AI Windows Service Installation Script
# Uses NSSM (Non-Sucking Service Manager) to register ARGUS AI as an automatic Windows Service.

param (
    [string]$NssmPath = "nssm.exe",
    [string]$ServiceName = "ArgusAiGaitService",
    [string]$DisplayName = "ARGUS AI Gait Recognition Service",
    [string]$Description = "Always-running CCTV Gait Recognition & Biometric Surveillance Engine",
    [string]$PythonPath = "",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ARGUS AI Windows Service Installation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Resolve Project Root if not provided
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Get-Item -Path $PSScriptRoot\..).FullName
}

Write-Host "[INFO] Project Root: $ProjectRoot" -ForegroundColor Yellow

# Resolve Python Executable if not provided
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $PythonPath = $VenvPython
    } else {
        $PythonPath = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    }
}

if (-not (Test-Path $PythonPath)) {
    Write-Error "[ERROR] Python executable not found at '$PythonPath'. Please specify -PythonPath."
}

Write-Host "[INFO] Python Executable: $PythonPath" -ForegroundColor Yellow

# Check NSSM availability
$NssmCmd = Get-Command $NssmPath -ErrorAction SilentlyContinue
if ($null -eq $NssmCmd -and -not (Test-Path $NssmPath)) {
    Write-Host "[WARNING] NSSM not found in PATH. Checking local directory..." -ForegroundColor Yellow
    $LocalNssm = Join-Path $ProjectRoot "deployment\nssm.exe"
    if (Test-Path $LocalNssm) {
        $NssmPath = $LocalNssm
    } else {
        Write-Error "[ERROR] NSSM binary not found. Please install NSSM or place nssm.exe in deployment directory."
    }
}

# Stop existing service if installed
Write-Host "[INFO] Checking existing service '$ServiceName'..." -ForegroundColor Yellow
& $NssmPath status $ServiceName 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[INFO] Stopping and removing existing service '$ServiceName'..." -ForegroundColor Yellow
    & $NssmPath stop $ServiceName
    & $NssmPath remove $ServiceName confirm
}

# Install service
Write-Host "[INFO] Installing service '$ServiceName'..." -ForegroundColor Green
$ServiceScript = Join-Path $ProjectRoot "services\argus_service.py"

& $NssmPath install $ServiceName $PythonPath "`"$ServiceScript`" --headless"
& $NssmPath set $ServiceName AppDirectory $ProjectRoot
& $NssmPath set $ServiceName DisplayName $DisplayName
& $NssmPath set $ServiceName Description $Description
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName AppExit Default Restart
& $NssmPath set $ServiceName AppRestartDelay 5000

# Configure logging redirect in NSSM
$LogDir = Join-Path $ProjectRoot "outputs\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

& $NssmPath set $ServiceName AppStdout (Join-Path $LogDir "service_stdout.log")
& $NssmPath set $ServiceName AppStderr (Join-Path $LogDir "service_stderr.log")

Write-Host "[INFO] Starting service '$ServiceName'..." -ForegroundColor Green
& $NssmPath start $ServiceName

Write-Host "[SUCCESS] ARGUS AI Service installed and started successfully!" -ForegroundColor Green
Write-Host "Status check: & '$NssmPath' status $ServiceName" -ForegroundColor Cyan
