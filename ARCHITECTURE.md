# ClearShot Architecture & Rebuild Prompt

ClearShot is a desktop-friendly full-stack web app for extracting sharp face/body crops from local videos or downloaded online videos. It is built as a React/Vite frontend served by a FastAPI backend. Long-running download and extraction work is coordinated through WebSockets so the UI stays responsive and shows live progress.

This document is both the current architecture guide and a prompt-quality specification that can be used to recreate the app.

> For detailed UI/UX principles, CSS architecture, and visual aesthetics, please refer to **[DESIGN.md](./DESIGN.md)**.

---

## Product Goal

Build an app that lets a user:

1. Upload a local video or paste a video URL.
2. For URLs, inspect available resolutions, download the chosen video, and then preview/download that source video.
3. Tune extraction settings.
4. Extract sharp, square face-focused or body-focused images.
5. Review extracted frames in a gallery.
6. Download all extracted frames as a ZIP.
7. Re-extract with new settings without deleting the already downloaded source video.

The app is intended for AI/ML training data preparation. It should feel like a modern utility, not a marketing site: compact controls, clear status, stable layout, and fast iteration.

---

## Stack

### Frontend

- React + TypeScript
- Vite
- Vanilla CSS in `frontend/src/index.css` (Supporting automatic system Light/Dark mode and manual 'M' key toggle)
- `motion/react` for small entrance animations
- `lucide-react` for action icons
- Native HTML5 video preview
- Minimalist floating status footer replacing traditional heavy target UI elements

### Backend

- Python 3.10+
- FastAPI + Uvicorn
- WebSockets for download/extraction progress
- OpenCV for frame decoding, image writing, resizing, blur scoring, and NMS helpers
- NumPy
- ONNX Runtime for detector inference
- CoreML Execution Provider on Apple Silicon, using Metal-capable `CPUAndGPU` compute units
- `pytubefix` for online video probing/downloading
- `ffprobe` when available for accurate video metadata; OpenCV metadata fallback

### Runtime Storage

- Uploaded files: `tempfile.gettempdir()/clearshot_uploads`
- Job output folders: `tempfile.gettempdir()/clearshot_output/{job_id}`
- ONNX models: `~/.clearshot/models`
- Generated frame names: `frame_{frame_idx:06d}_face_{det_idx}.{png|jpg}`

Job state is in memory in `api.routes.JOBS`. This is a single-user local app design, not a multi-user durable queue.

---

## Run Commands

Backend:

```bash
cd clear-shot
pip install -r requirements.txt
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Development backend with reload:

```bash
python3 server.py
```

Frontend:

```bash
cd clear-shot/frontend
npm install
npm run dev
```

Production build:

```bash
cd clear-shot/frontend
npm run build
cd ..
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

When `frontend/dist` exists, `server.py` serves the React app and static assets directly.

---

## Directory Map

```text
api/
  routes.py        REST API, job registry, upload/download/extract validation
  websocket.py     WebSocket actions and background thread orchestration

pipeline/
  downloader.py    URL probe/download with pytubefix and abortable progress
  extractor.py     Main frame sampling, filtering, crop, dedup, save pipeline
  face_detector.py SCRFD ONNX detector with GPU session and CPU retry
  body_detector.py YOLOv8n-pose ONNX detector with GPU session and CPU retry
  cropper.py       Face/body crop, padding, square, resize helpers
  gpu.py           ONNX Runtime provider detection/session creation
  quality.py       Laplacian blur scoring

frontend/src/
  App.tsx                    Main two-panel workflow UI
  components/VideoUpload.tsx Upload/URL input and preview card
  components/SettingsPanel.tsx Extraction controls
  components/Gallery.tsx     Result thumbnails
  components/StatsCard.tsx   Extraction counters and GPU status
  hooks/useExtraction.ts     App state machine and REST/WebSocket bridge
  hooks/useWebSocket.ts      Reconnecting/pending-message WebSocket wrapper
  services/api.ts            REST and WebSocket URL helpers
  types/index.ts             Shared frontend types and defaults

server.py          FastAPI app, routes, result/video file serving, SPA serving
```

---

## Frontend State Machine

`useExtraction` owns the workflow phase:

- `idle`: no job loaded
- `uploading`: local upload or URL probe in progress
- `ready`: local upload is ready, or URL has been probed and is ready to download
- `downloading`: URL video download is running
- `downloaded`: URL video is downloaded and extraction settings are shown
- `extracting`: extractor is running
- `complete`: results are ready
- `error`: terminal UI error state

The UI uses a two-column app layout:

- Left panel: upload/import, video preview, settings, action buttons.
- Right panel: error banner, progress, stats, gallery, empty state.

Important frontend behaviors:

- URL jobs show the quality selector and `Download Video` action before settings.
- Local upload jobs show settings immediately.
- Downloaded URL jobs update preview metadata with the actual downloaded file specs, not just probed remote metadata.
- `Re-extract` reuses the current job and settings, clears prior frame results, and preserves the source video.
- The source video download button appears after URL download and after extraction completion.
- The ZIP button appears after extraction completion.
- Sample rate max is clamped to the actual video FPS.
- Detection confidence slider is 30% to 80%, step 5%.
- Minimum sharpness slider is 20 to 250.
- Padding slider is 0% to 200%.
- Output sizes are `256`, `512`, `768`, `1024`, `1280`, `1536`, `1920`.

Default settings:

```ts
{
  target_fps: 2,
  blur_threshold: 100,
  detection_confidence: 0.5,
  crop_mode: 'face',
  padding_pct: 20,
  square_method: 'center_crop',
  output_size: 512,
  output_format: 'png',
  dedup_threshold: 8,
  occlusion_threshold: 50,
  download_resolution: 'max',
}
```

- **Occlusion Filter**: When `occlusion_threshold` is above `0`, the extractor initializes the YOLOv8n-pose body detector and rejects faces when visible arm/hand keypoints land inside the face box. MediaPipe hand-landmarker checks are opt-in with `CLEARSHOT_ENABLE_HAND_OCCLUSION=1` because that native runtime can hard-abort on some macOS/headless environments.

---

## API Contract

### REST

`POST /api/upload`

- Multipart upload with field `file`.
- Stores video in temp upload dir.
- Probes metadata.
- Creates `JOBS[job_id]`.
- Returns `{ job_id, meta }`.

`POST /api/download-url`

- Body: `{ "url": "..." }`
- Uses `pipeline.downloader.probe_url`.
- Returns job metadata including `available_formats`, `thumbnail_url`, `format_id`, `is_url: true`.
- Does not download yet.

`POST /api/extract/{job_id}`

- Body is extraction settings.
- Validates job exists, is idle enough to run, and has a source video.
- Resets progress/stats/results.
- Removes only previous `frame_*.png|jpg|jpeg` outputs, preserving uploaded/downloaded source videos.
- Returns `{ job_id, status: "pending" }`.
- Actual extraction begins when the WebSocket receives `{ action: "extract" }`.

`GET /api/jobs/{job_id}`

- Returns job status, progress, stage, stats, results, and metadata.

`GET /api/jobs/{job_id}/download`

- Returns a ZIP of extracted result frames.

`DELETE /api/jobs/{job_id}`

- Deletes job, source video, and output folder.

`GET /api/gpu-info`

- Returns `{ backend, device, provider }`, for example `{ "backend": "coreml", "device": "Apple M1", "provider": "CoreMLExecutionProvider" }`.

`GET /api/results/{job_id}/{filename}`

- Serves a single extracted frame.

`GET /api/video/{job_id}`

- Serves the source/downloaded video for preview and scrubbing.

`GET /api/video/{job_id}/download`

- Downloads the source/downloaded video as `clearshot_{job_id}.{ext}`.

### WebSocket

Path: `/ws/{job_id}`

Client sends:

```json
{ "action": "download", "format_id": "..." }
```

```json
{ "action": "abort" }
```

```json
{ "action": "extract", "settings": { "...": "..." } }
```

Server sends:

```json
{
  "type": "progress",
  "progress": 0.423,
  "stage": "detecting",
  "message": "Detecting faces - frame 1200/3000 | Extracted: 12"
}
```

```json
{
  "type": "download_complete",
  "video_path": "/tmp/clearshot_output/abcd1234/abcd1234.mp4",
  "meta": {
    "fps": 29.97,
    "duration": 120.2,
    "width": 1080,
    "height": 1920,
    "frame_count": 3600,
    "is_url": false,
    "downloaded_from_url": true
  }
}
```

```json
{
  "type": "complete",
  "stats": {
    "total_sampled": 100,
    "blurry_discarded": 12,
    "low_resolution_discarded": 5,
    "no_face_discarded": 20,
    "duplicate_discarded": 8,
    "extracted": 55,
    "gpu_backend": "coreml"
  },
  "results": ["/api/results/abcd1234/frame_000030_face_0.png"],
  "total": 55
}
```

```json
{ "type": "download_aborted" }
```

```json
{ "type": "error", "error": "..." }
```

The WebSocket handler runs download/extraction in `asyncio.to_thread`, bridges progress back to the socket through an `asyncio.Queue`, and sets `job["abort"] = True` on disconnect.

---

## Extraction Pipeline

Entry point: `pipeline.extractor.extract_frames(...)`

Settings:

- `target_fps`: sampled frames per second; must be greater than `0`
- `blur_threshold`: minimum Laplacian variance for the detected face region
- `detection_confidence`: detector confidence threshold
- `crop_mode`: `"face"` or `"body"`
- `padding_pct`: fractional padding after API converts percent to fraction
- `square_method`: `"center_crop"` or `"letterbox"`
- `output_size`: output square size in pixels
- `output_format`: `"png"`, `"jpg"`, or `"jpeg"`
- `dedup_threshold`: perceptual hash hamming distance; `0` disables dedup

Flow:

1. Validate settings.
2. Create output dir.
3. Remove previous generated frames only.
4. Probe video metadata with `ffprobe`, falling back to OpenCV.
5. Compute `frame_interval = max(1, round(video_fps / target_fps))`.
6. Initialize SCRFD face detector.
7. Initialize YOLOv8n-pose body detector only for body crop mode.
8. Open video with `cv2.VideoCapture`.
9. Iterate frames:
   - use `cap.grab()` to skip unsampled frames without full decode
   - decode sampled frames with `cap.read()`
   - detect faces
   - discard no-face frames
   - for each detected face:
     - reject truly tiny source faces using source-frame-aware thresholds
     - compute blur on face ROI
     - crop face or body
     - reject truly tiny source crops
     - compute blur on crop with a softer threshold (`blur_threshold * 0.35`)
     - optionally deduplicate with perceptual hash
     - square and resize
     - write asynchronously with a thread pool
10. Wait for all writes and raise if any image write failed.
11. Close detectors and video capture.
12. Return output paths and stats.

Low-resolution filtering is intentionally based on source frame dimensions, not requested output size. This prevents normal vertical 480p/720p videos from extracting zero frames just because the user chose a 512px or larger output.

Stats returned:

```json
{
  "total_sampled": 0,
  "blurry_discarded": 0,
  "low_resolution_discarded": 0,
  "no_face_discarded": 0,
  "duplicate_discarded": 0,
  "extracted": 0,
  "gpu_backend": "coreml"
}
```

---

## Detectors And GPU

### Face Detector

File: `pipeline/face_detector.py`

- Model: SCRFD 2.5G ONNX.
- Model path: `~/.clearshot/models/scrfd_2.5g.onnx`.
- Input: BGR OpenCV frame.
- Preprocess: letterbox to 640x640, normalize with `(img - 127.5) / 128.0`, convert HWC to NCHW.
- Postprocess: decode SCRFD stride levels 8, 16, 32, account for anchor count, undo padding/scale, clamp to original frame, apply OpenCV NMS.
- Returns `FaceDetection(x, y, w, h, score)`.

### Body Detector

File: `pipeline/body_detector.py`

- Model: YOLOv8n-pose ONNX from HuggingFace `Xenova/yolov8n-pose`.
- Model path: `~/.clearshot/models/yolov8n-pose.onnx`.
- Input: BGR OpenCV frame.
- Preprocess: letterbox to 640x640, normalize to `[0, 1]`, convert HWC to NCHW.
- Postprocess: decode `[1, 56, N]` or `[N, 56]`, filter by confidence, undo padding/scale, parse 17 COCO keypoints, apply NMS.
- Returns `BodyDetection(x, y, w, h, score, keypoints)`.

### GPU Session Management

File: `pipeline/gpu.py`

- `detect_gpu()` checks ONNX Runtime providers and prefers CUDA, then CoreML, then CPU.
- Apple Silicon uses `CoreMLExecutionProvider`.
- CoreML provider options include `ModelFormat: MLProgram` and `RequireStaticInputShapes: 1` for optimal Metal GPU dispatch.
- Set `CLEARSHOT_COREML_COMPUTE_UNITS` to override CoreML units (default: `CPUAndGPU`).
- Set `CLEARSHOT_DISABLE_COREML=1` to force CPU path.
- `COREML_COMPUTE_UNITS` env var is also set as a framework-level fallback because ORT 1.19's Python bindings silently ignore the dict-based CoreML options.
- CoreML temp files are directed to an app-owned `clearshot_coreml_tmp` folder because CoreML compilation can fail when macOS gives a bare temp root.
- Session creation falls back to CPU if GPU provider initialization fails.
- Detector inference catches GPU runtime failures, rebuilds a CPU session, and retries the same inference.
- Extractor `gpu_backend` stats report the backend actually in use after any runtime fallback.

#### Critical: ONNX Models Must Have Static Input Shapes

CoreML's GPU and Neural Engine paths require **static tensor shapes** to compile optimized Metal shaders. If an ONNX model declares dynamic spatial dims (e.g. `[1, 3, ?, ?]`), CoreML will silently fall back to CPU-only execution even when `MLComputeUnits` is set to `ALL`.

The script `scripts/fix_onnx_shapes.py` freezes dynamic dims to `[1, 3, 640, 640]`. Run it after downloading new models:

```bash
python3 scripts/fix_onnx_shapes.py
```

**Benchmark results (Apple M1):**

| Model | Before (dynamic) | After (static) | CPU-only | GPU speedup |
|---|---|---|---|---|
| SCRFD 2.5G | 17.3ms | **5.7ms** | 30.1ms | **5.3x** |
| YOLOv8n-pose | 13.9ms | 13.5ms | 42.6ms | 3.2x |

The SCRFD improvement came from static shapes reducing partitions from 7→4 and increasing CoreML node coverage from 128/152 → 133/136.

---

## Cropping And Quality Rules

Face crop:

- Expand face bbox by `padding_pct`.
- Clamp to frame bounds.
- Crop region.

Body crop:

- Run body detector.
- Choose body detection whose center is closest to the face center.
- Crop from visible pose keypoints with `padding_pct`.
- If body crop fails, fall back to face crop with doubled padding.

Square output:

- `center_crop`: crop longer dimension to square.
- `letterbox`: pad shorter dimension with black.
- Resize to `output_size` using Lanczos.

Quality:

- Blur score uses Variance of Laplacian.
- Face ROI must pass `blur_threshold`.
- Final crop must pass `blur_threshold * 0.35`.
- Source face/crop must pass source-size thresholds.
- Optional perceptual hash dedup skips near-identical crops.

---

## UI Design Notes

- Use a quiet dark app shell with restrained panels, compact settings, and clear actions.
- Use icon+text buttons for primary actions and download actions.
- Do not build a marketing landing page; the first screen is the tool itself.
- Keep action buttons aligned and evenly padded.
- Native video controls need enough width to avoid browser mini-control collapse.
- Preview aspect ratio should use the current video metadata and support vertical video.
- The preview card must update after a URL download with the downloaded file specs.
- Cards should have modern spacing and no nested decorative cards.

---

## Gotchas

- Do not delete downloaded source videos during re-extract. Only remove generated `frame_*` images.
- Do not trust probed URL dimensions after download; probe the actual downloaded file and update metadata.
- Do not tie source-quality filters directly to output size. Output size is a target canvas, not proof of source usability.
- Do not rely on MediaPipe as the primary path on macOS arm64. ONNX Runtime is the intended path.
- CoreML warnings about partial graph partitioning are normal. They do not mean extraction failed.
- Long-running downloads must support abort checks so Uvicorn can shut down cleanly.
- WebSocket messages may be sent before the socket is open; the frontend queues pending messages and flushes on open.
- Since jobs are in memory, backend restart loses current jobs.

---

## Validation Checklist

Backend:

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/clearshot-pycache \
  python3 -m py_compile \
  pipeline/extractor.py pipeline/face_detector.py pipeline/body_detector.py \
  pipeline/gpu.py api/routes.py api/websocket.py server.py
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Repository:

```bash
git diff --check
```

Manual checks:

- Upload a local video.
- Confirm sample-rate max matches video FPS.
- Extract in face mode.
- Re-extract with changed sharpness/confidence/padding.
- Import a URL.
- Select a quality and download.
- Confirm preview dimensions update after download.
- Download source video.
- Extract from downloaded video.
- Download ZIP.
- Confirm `/api/gpu-info` reports CoreML on Apple Silicon.

---

## Prompt To Recreate This App

Use this prompt to rebuild ClearShot from scratch:

```text
Build ClearShot, a full-stack local web app for extracting sharp square face/body crops from videos for AI training data.

Use a React + TypeScript + Vite frontend and a FastAPI + Python backend. The app should run locally, serve the production React build from FastAPI when built, and use REST for setup actions plus WebSockets for long-running download/extraction progress.

Core UX:
- First screen is the actual app, not a landing page.
- Use a modern dark utility UI with a two-panel layout.
- Left panel contains upload/import, video preview, settings, and action buttons.
- Right panel contains progress, stats, gallery, and empty/error states.
- Support local video upload and URL import.
- For URL import, first probe available resolutions with pytubefix, then show a quality selector and Download Video button.
- During download, show progress, speed, ETA, and a cancel button.
- After URL download, probe the actual downloaded video and update preview specs from that file.
- After URL download, show settings and allow source video download.
- After extraction, show Re-extract, source Video download when available, and ZIP download.
- Re-extract must preserve the uploaded/downloaded source video and delete only prior generated frame images.

Frontend settings:
- Sample Rate slider: 0.5 to actual video FPS, step 0.5, default 2.
- Minimum Sharpness slider: 20 to 250, step 10, default 100.
- Detection Confidence slider: 30% to 80%, step 5%, default 50%.
- Crop Mode segmented control: Face or Body.
- Padding slider: 0% to 200%, step 5%, default 20%.
- Square Method segmented control: Center Crop or Letterbox.
- Output Size segmented control: 256, 512, 768, 1024, 1280, 1536, 1920.
- Format segmented control: PNG or JPG.
- De-duplication slider: 0 to 20, step 1, default 8, where 0 is off.

Backend API:
- POST /api/upload accepts multipart file, stores it in temp storage, probes metadata, creates an in-memory job, returns { job_id, meta }.
- POST /api/download-url accepts { url }, probes pytubefix streams, returns { job_id, meta } with available_formats, thumbnail_url, format_id, is_url true.
- POST /api/extract/{job_id} validates the job and settings, resets stats/results, deletes only frame_*.png/jpg/jpeg in the job output folder, returns pending.
- GET /api/jobs/{job_id} returns status/progress/stats/results/meta.
- GET /api/jobs/{job_id}/download returns a ZIP of extracted frames.
- DELETE /api/jobs/{job_id} deletes the job and files.
- GET /api/gpu-info returns detected ONNX Runtime backend/device/provider.
- GET /api/results/{job_id}/{filename} serves extracted images.
- GET /api/video/{job_id} serves the source video for preview.
- GET /api/video/{job_id}/download downloads the source video.
- WebSocket /ws/{job_id} accepts actions: download, abort, extract. It emits progress, download_complete, complete, download_aborted, and error messages.

Backend storage:
- Keep jobs in an in-memory JOBS dict for single-user local use.
- Store uploads under tempfile.gettempdir()/clearshot_uploads.
- Store outputs/downloads under tempfile.gettempdir()/clearshot_output/{job_id}.
- Store ONNX models under ~/.clearshot/models.

Extraction pipeline:
- Use OpenCV VideoCapture to sample frames.
- Probe metadata with ffprobe when available and OpenCV as fallback.
- Compute frame_interval = max(1, round(video_fps / target_fps)).
- Use cap.grab() to skip unsampled frames and cap.read() for sampled frames.
- Detect faces with an ONNX SCRFD detector.
- For body mode, detect pose with an ONNX YOLOv8n-pose detector and choose the body closest to the face center.
- Crop face with padding, or crop body from keypoints with padding. If body crop fails, fallback to face crop with doubled padding.
- Reject frames with no faces.
- Reject blurry faces using Variance of Laplacian on the face ROI.
- Reject blurry crops using a softer crop threshold.
- Reject truly tiny source faces/crops using thresholds based on source frame dimensions, not requested output size.
- Optionally deduplicate crops with perceptual hash when imagehash is installed and dedup_threshold > 0.
- Convert to square using center crop or letterbox, resize to requested output size, save PNG/JPG.
- Write images asynchronously but raise if any write fails.
- Return stats: total_sampled, blurry_discarded, low_resolution_discarded, no_face_discarded, duplicate_discarded, extracted, gpu_backend.

ONNX/GPU:
- Implement pipeline/gpu.py with detect_gpu(), get_providers(), create_session(), create_cpu_session().
- Prefer CUDA if available, then CoreML, then CPU.
- On Apple Silicon use CoreMLExecutionProvider with ModelFormat: "MLProgram" and RequireStaticInputShapes: "1".
- Set COREML_COMPUTE_UNITS env var as a framework-level fallback since ORT 1.19 silently ignores CoreML dict options.
- Allow CLEARSHOT_COREML_COMPUTE_UNITS override (default CPUAndGPU) and CLEARSHOT_DISABLE_COREML=1.
- Prepare an app-owned CoreML temp dir under clearshot_coreml_tmp and set TMPDIR to it.
- If GPU session creation fails, fallback to CPU.
- If GPU inference fails at runtime, rebuild a CPU session and retry inference.
- Report actual runtime backend in extraction stats.

Models:
- ALL ONNX models must have their dynamic spatial dimensions frozen to static [1, 3, 640, 640] using a script (e.g. scripts/fix_onnx_shapes.py). CoreML cannot compile Metal shaders for dynamic shapes and will silently fallback to CPU.
- Face: SCRFD 2.5G ONNX, letterbox 640x640, normalize (img - 127.5) / 128.0, decode stride outputs 8/16/32, apply NMS, return x/y/w/h/score.
- Body: YOLOv8n-pose ONNX from HuggingFace Xenova/yolov8n-pose, letterbox 640x640, normalize 0..1, decode 56-channel output into bbox/confidence/17 keypoints, apply NMS.

Implementation requirements:
- Use TypeScript types for VideoMeta, ExtractionSettings, ExtractionStats, and WebSocket messages.
- Queue WebSocket messages while connecting and reconnect if needed.
- Clamp frontend target FPS to actual video FPS.
- Clamp sharpness to 250.
- Clamp detection confidence to 0.3..0.8.
- Keep generated UI text concise; no instructional landing page.
- Use lucide-react icons for actions.
- Validate with Python py_compile, npm run lint, npm run build, and git diff --check.
```
