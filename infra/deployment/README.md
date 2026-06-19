# Deployment — Lightweight (No Docker)

This project uses a lightweight local development stack with **no Docker required**.

## Quick Start (Windows)

### First-time setup
```powershell
.\infra\deployment\setup_env.ps1
```

This will:
1. Create `.env` from `.env.example`
2. Create a Python virtualenv
3. Install all dependencies (`pip install -r requirements.txt`)
4. Create database tables on Neon

### Edit your .env
Open `.env` and set your Neon DATABASE_URL:
```
DATABASE_URL=postgresql+asyncpg://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
```

### Start the dev server
```powershell
.\infra\deployment\start_dev.ps1
```

API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc

## Services — No Docker Stack

| Service | Docker Version | Lightweight Alternative |
|---|---|---|
| Database | PostgreSQL container | Neon Serverless PostgreSQL (free, cloud) |
| Task Queue | Celery + Redis | FastAPI BackgroundTasks (built-in) |
| ML Tracking | MLflow server container | File-based MLflow (`mlruns/` directory) |
| Object Storage | MinIO | Local filesystem (`storage/`) |
| Reverse Proxy | Nginx | Uvicorn direct |

## RAM Usage (This Stack vs Docker)

| | Docker Stack | This Stack |
|---|---|---|
| Database | ~800 MB | 0 MB (cloud) |
| Task Queue | ~400 MB | 0 MB (built-in) |
| ML Tracking | ~300 MB | ~100 MB (when UI open) |
| **Total** | **~1.5 GB** | **~100 MB** |
