from __future__ import annotations

"""
REST API routes for ClearShot.
"""

import os
import shutil
import tempfile
import uuid
import zipfile
import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline.extractor import probe_video


router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Job storage (in-memory, single-user desktop app)
# ---------------------------------------------------------------------------

JOBS: dict[str, dict[str, Any]] = {}
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "clearshot_uploads")
OUTPUT_BASE = os.path.join(tempfile.gettempdir(), "clearshot_output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)


def _clean_result_files(output_dir: str) -> None:
    """Remove generated frames without deleting uploaded/downloaded videos."""
    for pattern in ("frame_*.png", "frame_*.jpg", "frame_*.jpeg"):
        for path in Path(output_dir).glob(pattern):
            if path.is_file():
                path.unlink()


class ExtractionSettings(BaseModel):
    target_fps: float = 2.0
    blur_threshold: float = 100.0
    detection_confidence: float = 0.5
    crop_mode: str = "face"
    padding_pct: float = 20.0
    square_method: str = "center_crop"
    output_size: int = 512
    output_format: str = "png"
    dedup_threshold: int = 8
    filter_occluded: bool = True
    download_resolution: str = "max"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and return job metadata."""
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "video.mp4").suffix
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    # Save uploaded file
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    # Probe video metadata
    try:
        meta = probe_video(video_path)
    except Exception:
        meta = {"fps": 0, "duration": 0, "width": 0, "height": 0, "frame_count": 0}

    # Create output directory for this job
    output_dir = os.path.join(OUTPUT_BASE, job_id)
    os.makedirs(output_dir, exist_ok=True)

    JOBS[job_id] = {
        "id": job_id,
        "video_path": video_path,
        "output_dir": output_dir,
        "status": "uploaded",
        "progress": 0.0,
        "stage": "",
        "stats": {},
        "results": [],
        "meta": meta,
    }

    return {"job_id": job_id, "meta": meta}


class DownloadUrlRequest(BaseModel):
    url: str

@router.post("/download-url")
async def download_url(req: DownloadUrlRequest):
    """Register a URL download job."""
    from pipeline.downloader import probe_url
    
    job_id = str(uuid.uuid4())[:8]
    
    # Probe metadata
    try:
        meta = await asyncio.to_thread(probe_url, req.url)
        meta["is_url"] = True
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch video info: {e}")

    # Create output directory for this job
    output_dir = os.path.join(OUTPUT_BASE, job_id)
    os.makedirs(output_dir, exist_ok=True)

    JOBS[job_id] = {
        "id": job_id,
        "video_path": None,  # Will be populated during extraction/download
        "url": req.url,
        "output_dir": output_dir,
        "status": "ready",
        "progress": 0.0,
        "stage": "",
        "stats": {},
        "results": [],
        "meta": meta,
    }

    return {"job_id": job_id, "meta": meta}

@router.post("/extract/{job_id}")
async def start_extraction(job_id: str, settings: ExtractionSettings):
    """
    Start extraction for a job. Actual processing happens via WebSocket.
    This endpoint validates and stores settings.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("uploaded", "ready", "downloaded", "complete", "error"):
        raise HTTPException(status_code=409, detail="Extraction already running")
    if not job.get("video_path") or not os.path.exists(job["video_path"]):
        raise HTTPException(status_code=400, detail="Video is not available for extraction")

    # Reset job state
    job["status"] = "pending"
    job["progress"] = 0.0
    job["stage"] = ""
    job["stats"] = {}
    job["results"] = []
    job["settings"] = settings.model_dump()

    # Clean previous generated frames, preserving downloaded videos in the same folder.
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    _clean_result_files(output_dir)

    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "stats": job["stats"],
        "results": job["results"],
        "meta": job["meta"],
    }


@router.get("/jobs/{job_id}/download")
async def download_results(job_id: str):
    """Download all results as a ZIP file."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job["results"]:
        raise HTTPException(status_code=404, detail="No results to download")

    zip_path = os.path.join(OUTPUT_BASE, f"{job_id}_export.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in job["results"]:
            if os.path.exists(p):
                zf.write(p, arcname=Path(p).name)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"clearshot_{job_id}.zip",
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Cleanup files
    if job.get("video_path") and os.path.exists(job["video_path"]):
        os.remove(job["video_path"])
    if os.path.exists(job["output_dir"]):
        shutil.rmtree(job["output_dir"])

    del JOBS[job_id]
    return {"status": "deleted"}


@router.get("/gpu-info")
async def gpu_info():
    """Return detected GPU backend info."""
    try:
        from pipeline.gpu import detect_gpu
        info = detect_gpu()
        return info
    except ImportError:
        return {"backend": "cpu", "device": "CPU (gpu module not available)", "provider": "CPUExecutionProvider"}
