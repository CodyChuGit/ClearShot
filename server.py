from __future__ import annotations

"""
ClearShot — FastAPI server entry point.
Replaces the Gradio app with a decoupled REST + WebSocket backend.
"""

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router as api_router, OUTPUT_BASE
from api.websocket import router as ws_router


app = FastAPI(
    title="ClearShot API",
    description="Extract sharp, face-focused training data from video",
    version="2.0.0",
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)
app.include_router(ws_router)


# ---------------------------------------------------------------------------
# Serve result images
# ---------------------------------------------------------------------------

@app.get("/api/results/{job_id}/{filename}")
async def serve_result(job_id: str, filename: str):
    """Serve an individual result image."""
    file_path = os.path.join(OUTPUT_BASE, job_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

from api.routes import JOBS

@app.get("/api/video/{job_id}")
async def serve_video(job_id: str):
    """Serve the downloaded video file for scrubbing/preview."""
    job = JOBS.get(job_id)
    if not job or not job.get("video_path") or not os.path.exists(job["video_path"]):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(job["video_path"])


@app.get("/api/video/{job_id}/download")
async def download_video(job_id: str):
    """Download the source/downloaded video file."""
    job = JOBS.get(job_id)
    if not job or not job.get("video_path") or not os.path.exists(job["video_path"]):
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(job["video_path"])
    return FileResponse(
        str(video_path),
        filename=f"clearshot_{job_id}{video_path.suffix or '.mp4'}",
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Serve React frontend (production build)
# ---------------------------------------------------------------------------

FRONTEND_BUILD = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_BUILD.exists():
    # Serve static assets (JS, CSS, images)
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_BUILD / "assets")),
        name="frontend-assets",
    )

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React SPA — all non-API routes go to index.html."""
        file_path = FRONTEND_BUILD / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_BUILD / "index.html"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[".", "api", "pipeline"],
    )
