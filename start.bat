@echo off
title DeepGuard Platform
echo ==================================================
echo Starting DeepGuard Platform...
echo ==================================================

echo [1/2] Starting Next.js Frontend...
start "DeepGuard Frontend" cmd /k "cd ADS\frontend && npm run dev"

echo [2/2] Starting FastAPI Backend...
start "DeepGuard Backend" cmd /k "venv\Scripts\python.exe -m uvicorn apps.api.main:app --port 8001 --host 127.0.0.1 --reload"

echo.
echo Both servers have been launched in new windows!
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:8001
echo ==================================================
pause
