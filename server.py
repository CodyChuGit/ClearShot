from __future__ import annotations

"""
ClearShot — FastAPI server entry point.
Replaces the Gradio app with a decoupled REST + WebSocket backend.
"""

import os
import time
import shutil
import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router as api_router, OUTPUT_BASE, UPLOAD_DIR, JOBS
from api.websocket import router as ws_router

# ---------------------------------------------------------------------------
# Background Garbage Collection
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS = 2 * 60 * 60  # 2 hours

async def garbage_collection_loop():
    """Periodically purges orphaned cache files and stale jobs."""
    while True:
        try:
            await asyncio.sleep(30 * 60)  # run every 30 minutes
            now = time.time()
            
            # Clean UPLOAD_DIR
            if os.path.exists(UPLOAD_DIR):
                for f in os.listdir(UPLOAD_DIR):
                    p = os.path.join(UPLOAD_DIR, f)
                    if os.path.isfile(p) and (now - os.path.getmtime(p)) > CACHE_TTL_SECONDS:
                        try: os.remove(p)
                        except: pass

            # Clean OUTPUT_BASE
            if os.path.exists(OUTPUT_BASE):
                for d in os.listdir(OUTPUT_BASE):
                    p = os.path.join(OUTPUT_BASE, d)
                    if os.path.isdir(p) and (now - os.path.getmtime(p)) > CACHE_TTL_SECONDS:
                        try: shutil.rmtree(p)
                        except: pass
                        
            # Clean JOBS dictionary
            stale_jobs = [jid for jid, j in JOBS.items() if (now - j.get("created_at", 0)) > CACHE_TTL_SECONDS]
            for jid in stale_jobs:
                del JOBS[jid]
                
        except Exception:
            pass


app = FastAPI(
    title="ClearShot API",
    description="Extract sharp, face-focused training data from video",
    version="2.0.0",
)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(garbage_collection_loop())

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully wipe all cache when the server is stopped."""
    from api.routes import JOBS
    JOBS.clear()
    
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            try: os.remove(os.path.join(UPLOAD_DIR, f))
            except: pass
            
    if os.path.exists(OUTPUT_BASE):
        for d in os.listdir(OUTPUT_BASE):
            try: shutil.rmtree(os.path.join(OUTPUT_BASE, d))
            except: pass

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
    base_dir = Path(os.path.join(OUTPUT_BASE, job_id)).resolve()
    file_path = (base_dir / filename).resolve()
    
    # Security: Ensure resolved path is within the intended directory
    if not str(file_path).startswith(str(base_dir)):
        raise HTTPException(status_code=403, detail="Access denied")
        
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))

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


@app.get("/api/video/{job_id}/preview")
async def preview_video(job_id: str):
    """Extract a random frame from the video as a JPEG preview."""
    import cv2
    import random
    from fastapi import Response

    job = JOBS.get(job_id)
    if not job or not job.get("video_path") or not os.path.exists(job["video_path"]):
        raise HTTPException(status_code=404, detail="Video not found")

    cap = cv2.VideoCapture(job["video_path"])
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100

    # Pick a random frame between 10% and 90% to avoid black screens
    target_frame = random.randint(int(total_frames * 0.1), int(total_frames * 0.9))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Failed to extract frame")

    # Encode to JPEG
    success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


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
        base_dir = FRONTEND_BUILD.resolve()
        file_path = (base_dir / full_path).resolve()
        
        # Security: Prevent path traversal by validating containment
        if not str(file_path).startswith(str(base_dir)):
            return FileResponse(str(base_dir / "index.html"))
            
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(base_dir / "index.html"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[".", "api", "pipeline"],
    )
