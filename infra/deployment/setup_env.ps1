# =============================================================================
# DeepGuard Platform - Environment Setup Script (Windows)
# =============================================================================
# Run this ONCE to set up your development environment.
# Usage: .\infra\deployment\setup_env.ps1
# =============================================================================

Write-Host ''
Write-Host '======================================================' -ForegroundColor Cyan
Write-Host '  DeepGuard Platform - Environment Setup' -ForegroundColor Cyan
Write-Host '======================================================' -ForegroundColor Cyan
Write-Host ''

# Step 1: Create .env from .env.example
if (-Not (Test-Path '.env')) {
    Write-Host '[1/5] Creating .env from .env.example...' -ForegroundColor Yellow
    Copy-Item '.env.example' '.env'
    Write-Host '      .env created. IMPORTANT: Edit .env and add your Neon DATABASE_URL.' -ForegroundColor Green
} else {
    Write-Host '[1/5] .env already exists. Skipping.' -ForegroundColor Gray
}

# Step 2: Create virtualenv
if (-Not (Test-Path '.\venv')) {
    Write-Host '[2/5] Creating Python virtualenv...' -ForegroundColor Yellow
    python -m venv venv
    Write-Host '      venv created.' -ForegroundColor Green
} else {
    Write-Host '[2/5] venv already exists. Skipping.' -ForegroundColor Gray
}

# Step 3: Activate venv
Write-Host '[3/5] Activating venv...' -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Step 4: Install dependencies
Write-Host '[4/5] Installing Python dependencies...' -ForegroundColor Yellow
pip install -r requirements.txt

# Step 5: Create DB tables
Write-Host '[5/5] Creating database tables on Neon...' -ForegroundColor Yellow
Write-Host '      (Make sure DATABASE_URL is set in .env first!)' -ForegroundColor White
python -m apps.api.db.migrate

Write-Host ''
Write-Host '======================================================' -ForegroundColor Green
Write-Host '  Setup complete!' -ForegroundColor Green
Write-Host '  Run: .\infra\deployment\start_dev.ps1' -ForegroundColor Green
Write-Host '======================================================' -ForegroundColor Green
Write-Host ''
