from __future__ import annotations

"""
Main extraction pipeline: video → sharp face/body crops.

Uses ONNX Runtime-based detectors with GPU acceleration (CUDA/CoreML/CPU).
Falls back to MediaPipe if ONNX models are unavailable.
"""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

try:
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

from pipeline.quality import compute_blur_score, compute_blur_score_roi
from pipeline.cropper import crop_face, crop_body_from_keypoints, make_square, resize_square


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def probe_video(path: str) -> dict:
    """
    Get video metadata. Tries ffprobe first, falls back to OpenCV.

    Returns:
        dict with keys: fps, duration, width, height, frame_count
    """
    meta = _probe_ffprobe(path)
    if meta is None:
        meta = _probe_opencv(path)
    return meta


def _probe_ffprobe(path: str) -> dict | None:
    """Use ffprobe to get video metadata."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    try:
        cmd = [
            ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        video_stream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                video_stream = s
                break

        if video_stream is None:
            return None

        # Parse FPS from r_frame_rate (e.g., "30/1" or "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den)

        duration = float(data.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        frame_count = int(video_stream.get("nb_frames", 0))

        if frame_count == 0 and fps > 0 and duration > 0:
            frame_count = int(fps * duration)

        return {
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
    except Exception:
        return None


def _probe_opencv(path: str) -> dict:
    """Fallback: use OpenCV to get video metadata."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        return {
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _perceptual_hash(image: np.ndarray) -> object | None:
    """Compute perceptual hash for dedup. Returns None if imagehash unavailable."""
    if not HAS_IMAGEHASH:
        return None
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil, hash_size=8)


# ---------------------------------------------------------------------------
# Detector initialization
# ---------------------------------------------------------------------------

def _init_face_detector(detection_confidence: float):
    """
    Initialize face detector. Tries ONNX-based SCRFD first,
    falls back to MediaPipe.
    """
    try:
        from pipeline.face_detector import FaceDetector
        detector = FaceDetector()
        print(f"[ClearShot] Face detector: SCRFD via ONNX Runtime ({detector.backend})")
        return ("onnx", detector)
    except Exception as e:
        print(f"[ClearShot] ONNX face detector unavailable ({e}), falling back to MediaPipe")
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=detection_confidence,
        )
        return ("mediapipe", detector)


def _init_body_detector(detection_confidence: float):
    """
    Initialize body detector. Tries ONNX-based YOLOv8-pose first,
    falls back to MediaPipe Pose.
    """
    try:
        from pipeline.body_detector import BodyDetector
        detector = BodyDetector()
        print(f"[ClearShot] Body detector: YOLOv8-pose via ONNX Runtime ({detector.backend})")
        return ("onnx", detector)
    except Exception as e:
        print(f"[ClearShot] ONNX body detector unavailable ({e}), falling back to MediaPipe")
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        detector = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
        )
        return ("mediapipe", detector)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 2.0,
    blur_threshold: float = 100.0,
    detection_confidence: float = 0.5,
    crop_mode: str = "face",          # "face" or "body"
    padding_pct: float = 0.2,
    square_method: str = "center_crop",
    output_size: int = 512,
    output_format: str = "png",
    dedup_threshold: int = 8,         # hamming distance for dedup
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[str], dict]:
    """
    Extract high-quality face/body crops from a video.

    Uses GPU-accelerated ONNX detectors when available, with MediaPipe fallback.

    Returns:
        Tuple of (list of output file paths, stats dict).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Probe video ---
    meta = probe_video(video_path)
    video_fps = meta["fps"]
    frame_count = meta["frame_count"]

    # Frame interval: sample every N frames
    frame_interval = max(1, int(round(video_fps / target_fps)))

    # --- Init detectors ---
    face_type, face_detector = _init_face_detector(detection_confidence)

    body_type, body_detector = None, None
    if crop_mode == "body":
        body_type, body_detector = _init_body_detector(detection_confidence)

    # Determine GPU backend for stats
    try:
        from pipeline.gpu import detect_gpu
        gpu_info = detect_gpu()
        gpu_backend = gpu_info["backend"]
    except Exception:
        gpu_backend = "cpu"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_paths = []
    seen_hashes = set()

    stats = {
        "total_sampled": 0,
        "blurry_discarded": 0,
        "no_face_discarded": 0,
        "duplicate_discarded": 0,
        "extracted": 0,
        "gpu_backend": gpu_backend,
    }

    ext = "png" if output_format == "png" else "jpg"
    frame_idx = 0

    import concurrent.futures
    # Use a thread pool to write images asynchronously, unblocking the GPU/CPU inference loop
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    try:
        while True:
            # OPTIMIZATION 1: Bypass decoding for frames we are going to skip
            if frame_idx % frame_interval != 0:
                ret = cap.grab()
                if not ret:
                    break
                frame_idx += 1
                continue

            # Decode the target frame
            ret, frame = cap.read()
            if not ret:
                break

            stats["total_sampled"] += 1

            # Progress update
            if progress_callback and frame_count > 0:
                pct = min(frame_idx / frame_count, 1.0)
                progress_callback(
                    pct,
                    f"Detecting faces — frame {frame_idx}/{frame_count} | Extracted: {stats['extracted']}"
                )

            fh, fw = frame.shape[:2]

            # --- Face detection ---
            face_bboxes = _detect_faces(face_type, face_detector, frame, detection_confidence)

            if not face_bboxes:
                stats["no_face_discarded"] += 1
                frame_idx += 1
                continue

            # Process every detected face in the frame
            for det_idx, face_bbox in enumerate(face_bboxes):
                # --- Blur check on face region ---
                blur_score = compute_blur_score_roi(frame, face_bbox)
                if blur_score < blur_threshold:
                    stats["blurry_discarded"] += 1
                    continue

                # --- Crop ---
                if crop_mode == "body" and body_detector is not None:
                    cropped = _crop_body(
                        body_type, body_detector, frame, face_bbox, padding_pct
                    )
                    if cropped is None:
                        cropped = crop_face(frame, face_bbox, padding_pct * 2)
                else:
                    cropped = crop_face(frame, face_bbox, padding_pct)

                if cropped is None or cropped.size == 0:
                    continue

                # --- Dedup ---
                if HAS_IMAGEHASH and dedup_threshold > 0:
                    phash = _perceptual_hash(cropped)
                    if phash is not None:
                        is_dup = any(
                            (phash - h) < dedup_threshold for h in seen_hashes
                        )
                        if is_dup:
                            stats["duplicate_discarded"] += 1
                            continue
                        seen_hashes.add(phash)

                # --- Square + resize ---
                squared = make_square(cropped, method=square_method)
                final = resize_square(squared, size=output_size)

                # --- Save ---
                # OPTIMIZATION 2: Async I/O for saving images
                filename = f"frame_{frame_idx:06d}_face_{det_idx}.{ext}"
                out_path = os.path.join(output_dir, filename)
                executor.submit(cv2.imwrite, out_path, final)
                output_paths.append(out_path)
                stats["extracted"] += 1

            frame_idx += 1

    finally:
        cap.release()
        _close_detector(face_type, face_detector)
        if body_detector is not None:
            _close_detector(body_type, body_detector)
        
        # Ensure all writes complete before returning
        executor.shutdown(wait=True)

    # Final progress
    if progress_callback:
        progress_callback(1.0, f"Done! Extracted {stats['extracted']} images.")

    return output_paths, stats


# ---------------------------------------------------------------------------
# Detector abstraction helpers
# ---------------------------------------------------------------------------

def _detect_faces(
    detector_type: str,
    detector,
    frame: np.ndarray,
    confidence: float,
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces using either ONNX or MediaPipe detector.

    Returns list of (x, y, w, h) bounding boxes.
    """
    fh, fw = frame.shape[:2]

    if detector_type == "onnx":
        detections = detector.detect(frame, confidence=confidence)
        return [(d.x, d.y, d.w, d.h) for d in detections]
    else:
        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = detector.process(rgb)
        if not face_results.detections:
            return []
        bboxes = []
        for detection in face_results.detections:
            rbb = detection.location_data.relative_bounding_box
            bboxes.append((
                int(rbb.xmin * fw),
                int(rbb.ymin * fh),
                int(rbb.width * fw),
                int(rbb.height * fh),
            ))
        return bboxes


def _crop_body(
    detector_type: str,
    detector,
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    padding_pct: float,
) -> np.ndarray | None:
    """
    Crop body region using either ONNX or MediaPipe detector.
    """
    if detector_type == "onnx":
        bodies = detector.detect(frame)
        if bodies:
            # Find the body detection closest to the face
            face_cx = face_bbox[0] + face_bbox[2] / 2
            face_cy = face_bbox[1] + face_bbox[3] / 2

            best_body = min(bodies, key=lambda b: (
                (b.x + b.w / 2 - face_cx) ** 2 +
                (b.y + b.h / 2 - face_cy) ** 2
            ))

            # Use keypoints for tighter body crop
            return crop_body_from_keypoints(
                frame,
                [(kp.x, kp.y, kp.confidence) for kp in best_body.keypoints],
                padding_pct,
            )
        return None
    else:
        # MediaPipe Pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = detector.process(rgb)
        if pose_results.pose_landmarks and pose_results.pose_landmarks.landmark:
            return crop_body(
                frame,
                pose_results.pose_landmarks.landmark,
                padding_pct,
            )
        return None


def _close_detector(detector_type: str, detector):
    """Close a detector safely."""
    try:
        detector.close()
    except Exception:
        pass
