# ARGUS AI Windows Service Uninstallation Script
# Removes the ARGUS AI Windows service registered with NSSM.

param (
    [string]$NssmPath = "nssm.exe",
    [string]$ServiceName = "ArgusAiGaitService"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ARGUS AI Windows Service Uninstallation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check NSSM availability
$NssmCmd = Get-Command $NssmPath -ErrorAction SilentlyContinue
if ($null -eq $NssmCmd -and -not (Test-Path $NssmPath)) {
    $LocalNssm = Join-Path $PSScriptRoot "nssm.exe"
    if (Test-Path $LocalNssm) {
        $NssmPath = $LocalNssm
    } else {
        Write-Error "[ERROR] NSSM binary not found. Cannot stop/uninstall service."
    }
}

Write-Host "[INFO] Stopping service '$ServiceName'..." -ForegroundColor Yellow
& $NssmPath stop $ServiceName 2>$null

Write-Host "[INFO] Removing service '$ServiceName'..." -ForegroundColor Yellow
& $NssmPath remove $ServiceName confirm

Write-Host "[SUCCESS] ARGUS AI Service '$ServiceName' uninstalled successfully!" -ForegroundColor Green
