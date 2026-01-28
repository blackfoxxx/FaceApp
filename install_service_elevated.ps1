# FaceApp Service Installation with Auto-Elevation
# This script will automatically request administrator privileges

param([switch]$Elevated)

function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-FaceAppService {
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host "Installing FaceApp Windows Service" -ForegroundColor Green
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host ""

    # Change to FaceApp directory
    Set-Location -Path "c:\FaceApp"
    
    # Activate virtual environment
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\activate.bat"
    
    # Install required Windows service dependencies
    Write-Host "[INFO] Installing Windows service dependencies..." -ForegroundColor Yellow
    & ".\venv\Scripts\pip.exe" install pywin32
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install pywin32" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }
    
    # Install the service
    Write-Host "[INFO] Installing FaceApp service..." -ForegroundColor Yellow
    & ".\venv\Scripts\python.exe" faceapp_service.py install
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install service" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }
    
    # Start the service
    Write-Host "[INFO] Starting FaceApp service..." -ForegroundColor Yellow
    & ".\venv\Scripts\python.exe" faceapp_service.py start
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to start service" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }
    
    Write-Host ""
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host "FaceApp Service Installation Complete!" -ForegroundColor Green
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Service Name: FaceAppService" -ForegroundColor Cyan
    Write-Host "Display Name: FaceApp Face Recognition Service" -ForegroundColor Cyan
    Write-Host "Status: Started" -ForegroundColor Green
    Write-Host ""
    Write-Host "The FaceApp will now start automatically when Windows boots." -ForegroundColor Yellow
    Write-Host "You can manage the service through Windows Services (services.msc)" -ForegroundColor Yellow
    Write-Host "or use the manage_service.bat script." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "FaceApp should be accessible at: http://localhost:8080" -ForegroundColor Cyan
    Write-Host ""
    
    # Test if service is running
    Write-Host "[INFO] Testing service status..." -ForegroundColor Yellow
    $service = Get-Service -Name "FaceAppService" -ErrorAction SilentlyContinue
    if ($service) {
        Write-Host "Service Status: $($service.Status)" -ForegroundColor $(if ($service.Status -eq 'Running') { 'Green' } else { 'Red' })
    }
    
    Read-Host "Press Enter to continue"
}

# Check if running as administrator
if (-NOT (Test-Admin)) {
    if ($Elevated) {
        Write-Host "ERROR: Failed to obtain administrator privileges" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    } else {
        Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
        Start-Process PowerShell -Verb RunAs -ArgumentList ("-NoProfile -ExecutionPolicy Bypass -File `"{0}`" -Elevated" -f ($myinvocation.MyCommand.Definition))
        exit 0
    }
}

# If we reach here, we have admin privileges
Write-Host "Running with administrator privileges - OK" -ForegroundColor Green
Install-FaceAppService