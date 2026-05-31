"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apps.api.core.config import settings
from apps.api.routers import video

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="REST API for DeepGuard Video Detection Subsystem",
    version="1.0.0"
)

# CORS middleware for potential frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the artifacts directory to serve generated graphs, heatmaps, and reports
app.mount("/artifacts", StaticFiles(directory=settings.ARTIFACTS_DIR), name="artifacts")

# Include the video analysis router
app.include_router(video.router, prefix=settings.API_V1_STR, tags=["video"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to DeepGuard VDS API. Refer to /docs for the API schema."
    }
