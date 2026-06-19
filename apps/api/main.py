"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apps.api.core.config import settings
from apps.api.db.database import init_db
from apps.api.routers import auth, audio, fusion, image, jobs, video, users


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and once at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Called once when the FastAPI server starts.
    Creates all database tables if they don't already exist.
    This is safe to call every time — it is a no-op if tables already exist.
    """
    await init_db()
    yield
    # (shutdown code would go here if needed)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DeepGuard Platform API",
    description=(
        "REST API for DeepGuard — Video, Audio, and Image Deepfake Detection Platform.\n\n"
        "**Authentication:** Register at `/api/v1/auth/register`, then login at "
        "`/api/v1/auth/login` to receive a JWT bearer token. Pass it as:\n"
        "`Authorization: Bearer <token>`"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow all origins in dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated heatmaps, graphs, and forensic reports as static files
app.mount("/artifacts", StaticFiles(directory=settings.ARTIFACTS_DIR), name="artifacts")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(auth.router,   prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(video.router,  prefix=settings.API_V1_STR,              tags=["video"])
app.include_router(image.router,  prefix=settings.API_V1_STR,              tags=["image"])
app.include_router(audio.router,  prefix=settings.API_V1_STR,              tags=["audio"])
app.include_router(jobs.router,   prefix=settings.API_V1_STR,              tags=["jobs"])
app.include_router(fusion.router, prefix=settings.API_V1_STR,              tags=["fusion"])
app.include_router(users.router,  prefix=f"{settings.API_V1_STR}/users",   tags=["users"])


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", tags=["health"])
async def root():
    return {
        "message": "Welcome to DeepGuard Platform API. Visit /docs for the full API schema.",
        "version": "2.0.0",
        "endpoints": {
            "register":     "/api/v1/auth/register",
            "login":        "/api/v1/auth/login",
            "me":           "/api/v1/auth/me",
            "video":        "/api/v1/analyze/video",
            "image":        "/api/v1/analyze/image",
            "audio":        "/api/v1/analyze/audio",
            "fusion":       "/api/v1/analyze/fusion",
            "job_status":   "/api/v1/jobs/{job_id}",
            "jobs_list":    "/api/v1/jobs",
            "platform_stats": "/api/v1/stats",
        },
    }
