from __future__ import annotations

"""
WebSocket handler for real-time extraction progress.
"""

import asyncio
import json
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pipeline.extractor import extract_frames
from api.routes import JOBS


router = APIRouter()


@router.websocket("/ws/{job_id}")
async def extraction_ws(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for running extraction with real-time progress.

    Flow:
    1. Client connects after POST /api/extract/{job_id}
    2. Server runs extraction in a background thread
    3. Progress updates are pushed via the WebSocket
    4. Final results are sent on completion
    """
    await websocket.accept()

    job = JOBS.get(job_id)
    if not job:
        await websocket.send_json({"type": "error", "error": "Job not found"})
        await websocket.close()
        return

    progress_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def dl_progress(pct: float, msg: str):
        loop.call_soon_threadsafe(progress_queue.put_nowait, {
            "type": "progress",
            "progress": round(pct, 3),
            "stage": "downloading",
            "message": msg,
        })

    def ex_progress(pct: float, msg: str):
        stage = "processing"
        if "sampl" in msg.lower(): stage = "sampling"
        elif "detect" in msg.lower(): stage = "detecting"
        elif "extract" in msg.lower(): stage = "extracting"
        elif "done" in msg.lower(): stage = "complete"

        job["progress"] = pct
        job["stage"] = stage

        loop.call_soon_threadsafe(progress_queue.put_nowait, {
            "type": "progress",
            "progress": round(pct, 3),
            "stage": stage,
            "message": msg,
        })

    def run_download():
        try:
            from pipeline.downloader import download_video
            job["abort"] = False
            job["video_path"] = download_video(
                url=job["url"],
                output_dir=job["output_dir"],
                job_id=job["id"],
                progress_callback=dl_progress,
                format_id=job.get("meta", {}).get("format_id", "bestvideo"),
                should_abort=lambda: job.get("abort", False)
            )
            loop.call_soon_threadsafe(progress_queue.put_nowait, {
                "type": "download_complete",
                "video_path": job["video_path"]
            })
            job["status"] = "downloaded"
        except Exception as e:
            if str(e) == "Download aborted by user":
                job["status"] = "ready"
                # Optionally clean up partial file here
                loop.call_soon_threadsafe(progress_queue.put_nowait, {
                    "type": "download_aborted"
                })
            else:
                job["status"] = "error"
                loop.call_soon_threadsafe(progress_queue.put_nowait, {
                    "type": "error",
                    "error": str(e)
                })

    def run_extract(settings: dict):
        try:
            paths, stats = extract_frames(
                video_path=job["video_path"],
                output_dir=job["output_dir"],
                target_fps=settings.get("target_fps", 2.0),
                blur_threshold=settings.get("blur_threshold", 100.0),
                detection_confidence=settings.get("detection_confidence", 0.5),
                crop_mode=settings.get("crop_mode", "face"),
                padding_pct=settings.get("padding_pct", 20.0) / 100.0,
                square_method=settings.get("square_method", "center_crop"),
                output_size=settings.get("output_size", 512),
                output_format=settings.get("output_format", "png"),
                dedup_threshold=settings.get("dedup_threshold", 8),
                progress_callback=ex_progress,
            )

            job["status"] = "complete"
            job["stats"] = stats
            job["results"] = paths
            job["progress"] = 1.0

            result_filenames = [f"/api/results/{job_id}/{p.split('/')[-1]}" for p in paths]

            loop.call_soon_threadsafe(progress_queue.put_nowait, {
                "type": "complete",
                "stats": stats,
                "results": result_filenames,
                "total": len(paths),
            })
        except Exception as e:
            loop.call_soon_threadsafe(progress_queue.put_nowait, {
                "type": "error",
                "error": str(e)
            })
            job["status"] = "error"

    async def consume_queue():
        try:
            while True:
                msg = await progress_queue.get()
                if msg is None:
                    break
                await websocket.send_json(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    consumer_task = asyncio.create_task(consume_queue())

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "download":
                job["status"] = "downloading"
                fmt_id = data.get("format_id") or job.get("meta", {}).get("format_id", "bestvideo")
                
                # We need to pass the custom format_id to run_download
                job["meta"]["format_id"] = fmt_id
                asyncio.create_task(asyncio.to_thread(run_download))
            
            elif action == "abort":
                job["abort"] = True
                
            elif action == "extract":
                job["status"] = "running"
                settings_dict = data.get("settings", {})
                job["settings"] = settings_dict
                asyncio.create_task(asyncio.to_thread(run_extract, settings_dict))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        job["status"] = "error"
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    finally:
        await progress_queue.put(None)
        await consumer_task
        try:
            await websocket.close()
        except:
            pass
