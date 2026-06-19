# =============================================================================
# DeepGuard Platform - Windows Dev Startup Script
# =============================================================================
# Usage: .\infra\deployment\start_dev.ps1
#
# Starts the FastAPI backend server (no Docker needed).
# =============================================================================

Write-Host '' 
Write-Host '======================================================' -ForegroundColor Cyan
Write-Host '  DeepGuard Platform - Dev Server Startup' -ForegroundColor Cyan
Write-Host '======================================================' -ForegroundColor Cyan
Write-Host ''

# Check .env file exists
if (-Not (Test-Path '.env')) {
    Write-Host 'ERROR: .env file not found.' -ForegroundColor Red
    Write-Host 'Copy .env.example to .env and fill in your Neon DATABASE_URL.' -ForegroundColor Yellow
    Write-Host '  copy .env.example .env' -ForegroundColor White
    exit 1
}

# Check Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host 'ERROR: Python not found. Install Python 3.10 or 3.11.' -ForegroundColor Red
    exit 1
}

# Activate venv if it exists
if (Test-Path '.\venv\Scripts\Activate.ps1') {
    Write-Host 'Activating virtualenv...' -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host 'No venv found - using system Python. Run:' -ForegroundColor Yellow
    Write-Host '  python -m venv venv && .\venv\Scripts\Activate.ps1' -ForegroundColor White
}

Write-Host ''
Write-Host 'Starting DeepGuard Platform API...' -ForegroundColor Green
Write-Host '  URL:  http://localhost:8000' -ForegroundColor White
Write-Host '  Docs: http://localhost:8000/docs' -ForegroundColor White
Write-Host '  Press Ctrl+C to stop' -ForegroundColor White
Write-Host ''

uvicorn apps.api.main:app --reload --port 8000 --host 0.0.0.0
