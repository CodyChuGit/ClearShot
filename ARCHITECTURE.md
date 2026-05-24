# ClearShot Architecture & Developer Guide

## System Overview
ClearShot is a full-stack application designed to automatically extract high-quality, clear shots (frames) containing people and faces from local video files or YouTube URLs. It achieves this by combining standard video processing tools with hardware-accelerated AI models.

The system is split into a **React/TypeScript Frontend** and a **FastAPI Python Backend**, communicating via REST for simple operations and WebSockets for long-running, asynchronous tasks.

---

## Technology Stack

### Frontend
- **Framework:** React 18 with TypeScript
- **Bundler:** Vite
- **Styling:** Vanilla CSS (`index.css`) utilizing CSS variables for theme tokens
- **Animations:** Framer Motion
- **Icons:** Lucide React
- **Video Player:** Native HTML5 `<video>` (Styled to accommodate native browser controls—specifically maintaining a minimum width >350px to prevent Chrome/Safari from collapsing the timeline scrubber).

### Backend
- **Framework:** FastAPI / Uvicorn (Asynchronous ASGI server)
- **Real-time API:** WebSockets for continuous progress updates
- **Video Processing:** FFmpeg (used via `subprocess` for frame extraction)
- **YouTube Downloader:** `pytubefix` (for resolving and downloading streams)
- **AI Inference Engine:** ONNX Runtime (`onnxruntime`) 
  - *Note:* Configured to utilize the `CoreMLExecutionProvider` for GPU acceleration on Apple Silicon (M1/M2/M3).

---

## AI Pipeline & Models

The AI pipeline is designed to be fast and lightweight, prioritizing ONNX over heavy dependencies like PyTorch or MediaPipe. Models are automatically downloaded from HuggingFace mirrors on the first run.

1. **Face Detection (`pipeline/face_detector.py`):**
   - **Model:** SCRFD 2.5G (InsightFace)
   - **Details:** High-performance, single-stage face detector. Originally from InsightFace but uses a HuggingFace mirror due to upstream link deprecation.
   - **Output:** Bounding boxes, confidence scores, and facial keypoints (though primarily bounding boxes are used for quality assessment).

2. **Body Pose Detection (`pipeline/body_detector.py`):**
   - **Model:** YOLOv8n-pose (Ultralytics)
   - **Details:** Extracts body bounding boxes to ensure the subject is well-framed within the shot. Uses a HuggingFace mirror for the ONNX file.

3. **Quality & Scoring (`pipeline/quality.py` & `scorer.py`):**
   - **Blur Detection:** Uses Variance of Laplacian to filter out blurry/in-motion frames.
   - **Scoring:** Ranks frames based on face size, face confidence, and distance from the center of the frame.
   - **Deduplication:** Prevents highly similar adjacent frames from being selected.

---

## System Workflows

### 1. YouTube Download Workflow
1. User submits a URL to the frontend.
2. The frontend POSTs to `/api/download-url`, and the backend initializes a Job ID.
3. The frontend establishes a WebSocket connection using the Job ID.
4. The backend uses `pytubefix` in a background thread to download the video.
5. Real-time progress is streamed over the WebSocket.
6. **Concurrency/Abort Handling:** If the WebSocket disconnects (e.g., user refreshes, closes tab, or hits "X"), the backend traps the `WebSocketDisconnect` exception, flags the job as `abort = True`, and the background thread silently exits. This prevents runaway threads from locking up Uvicorn during hot-reloads.

### 2. Video Extraction Workflow
1. The backend triggers the `extractor.py` pipeline.
2. FFmpeg extracts frames into memory (or a temporary disk space) at a configured sample rate (e.g., 2 FPS).
3. The AI models run over each frame.
4. Frames without faces or bodies, or frames failing the blur threshold, are discarded.
5. The remaining frames are ranked, and the top *N* frames are saved.
6. The WebSocket receives the `complete` status, and the frontend fetches the final thumbnails via `/api/results/{job_id}`.

---

## Key Developer Notes & Gotchas

1. **Native Video Player Width Limitations:** 
   Native browser video controls (Chrome/Safari) automatically switch to a "mini-player" layout (hiding the timeline and volume sliders) if the video container is narrower than ~350px. The UI layout uses a `420px` grid column to ensure the video always exceeds this threshold.
   
2. **Vertical Video Handling:**
   The `.preview-container` utilizes a dynamic inline `aspect-ratio` based on the downloaded video's metadata (`videoMeta.width / videoMeta.height`). It is capped at `max-height: 70vh` to prevent 9:16 videos from pushing the controls off the screen.

3. **ONNX Output Indexing:**
   When working with the SCRFD ONNX model, ensure you map the output arrays correctly. The model returns groups of arrays for different stride scales (8, 16, 32). The scores and bounding boxes must be paired perfectly (e.g., Output 1 [shape: 3200x1] matches Output 4 [shape: 3200x4]) to avoid `IndexError` crashes.

4. **MediaPipe Fallback Abandoned:**
   The application used to fallback to Google's MediaPipe if the ONNX models failed to download. Because recent versions of `mediapipe` dropped the `solutions` package for macOS arm64, the fallback will crash. Always ensure the ONNX download URLs are healthy, as they are the primary and only reliable inference path.

5. **Hot-Reloading Thread Leaks:**
   Never launch infinite or long-running tasks using `asyncio.to_thread` without an `abort` flag check. If Uvicorn attempts to shut down while a thread is blindly reading a queue or downloading a file without checking for application exit or disconnects, the server will hang indefinitely.

---

## Directory Structure

- `/frontend` - React/Vite source code.
  - `/src/components` - UI components (e.g., `VideoUpload.tsx`).
  - `/src/hooks` - Custom React hooks (e.g., `useWebSocket.ts`).
- `/api` - FastAPI routes and WebSockets.
- `/pipeline` - Core AI logic, ONNX model management, and FFmpeg processing.
- `/models` *(Generated)* - Downloaded `.onnx` files are stored in `~/.clearshot/models/`.
- `/data/uploads` *(Generated)* - Temporary storage for uploaded/downloaded videos.
- `/data/output` *(Generated)* - Final high-quality extracted frames.
