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
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import queue
import threading

try:
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

from pipeline.quality import compute_blur_score, compute_blur_score_roi
from pipeline.cropper import crop_face, crop_body, crop_body_from_keypoints, make_square, resize_square
from pipeline.face_detector import FaceDetection


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


def _remove_previous_outputs(output_dir: str) -> None:
    """Delete previous extracted frames while preserving uploaded/downloaded videos."""
    for pattern in ("frame_*.png", "frame_*.jpg", "frame_*.jpeg"):
        for path in Path(output_dir).glob(pattern):
            if path.is_file():
                path.unlink()


def _write_image(path: str, image: np.ndarray) -> None:
    """Write an image and raise if OpenCV fails silently."""
    ok = cv2.imwrite(path, image)
    if not ok:
        raise RuntimeError(f"Failed to write extracted image: {path}")






def _min_source_face_size(frame: np.ndarray) -> int:
    """
    Minimum face size in original video pixels.

    This is intentionally based on the source frame, not the requested export
    size. Export size controls the final canvas; it should not make a normal
    480p/720p portrait video extract zero frames just because the user asked
    for a 512px output.
    """
    short_edge = min(frame.shape[:2])
    return int(min(96, max(48, round(short_edge * 0.12))))


def _min_source_crop_size(frame: np.ndarray) -> int:
    """Reject genuinely tiny source crops before they are enlarged."""
    short_edge = min(frame.shape[:2])
    return int(min(160, max(80, round(short_edge * 0.18))))


def _clamped_bbox_size(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = bbox
    if w < 0:
        x += w
        w = abs(w)
    if h < 0:
        y += h
        h = abs(h)

    fh, fw = frame.shape[:2]
    x1 = max(0, min(fw, int(round(x))))
    y1 = max(0, min(fh, int(round(y))))
    x2 = max(0, min(fw, int(round(x + w))))
    y2 = max(0, min(fh, int(round(y + h))))
    return max(0, x2 - x1), max(0, y2 - y1)


def _is_low_resolution_source(
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    crop: np.ndarray,
) -> bool:
    face_w, face_h = _clamped_bbox_size(frame, face_bbox)
    crop_h, crop_w = crop.shape[:2]

    return (
        min(face_w, face_h) < _min_source_face_size(frame)
        or min(crop_w, crop_h) < _min_source_crop_size(frame)
    )


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
    occlusion_threshold: int = 50,    # 0 to 100, 0 = off
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[str], dict]:
    """
    Extract high-quality face/body crops from a video.

    Uses GPU-accelerated ONNX detectors when available, with MediaPipe fallback.

    Returns:
        Tuple of (list of output file paths, stats dict).
    """
    output_format = output_format.lower()

    if target_fps <= 0:
        raise ValueError("target_fps must be greater than 0")
    if output_size <= 0:
        raise ValueError("output_size must be greater than 0")
    if crop_mode not in {"face", "body"}:
        raise ValueError("crop_mode must be 'face' or 'body'")
    if square_method not in {"center_crop", "letterbox"}:
        raise ValueError("square_method must be 'center_crop' or 'letterbox'")
    if output_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("output_format must be 'png' or 'jpg'")

    os.makedirs(output_dir, exist_ok=True)
    _remove_previous_outputs(output_dir)

    # --- Probe video ---
    meta = probe_video(video_path)
    video_fps = meta["fps"]
    frame_count = meta["frame_count"]

    # Frame interval: sample every N frames
    frame_interval = max(1, int(round(video_fps / target_fps)))

    # --- Init detectors ---
    face_type, face_detector = _init_face_detector(detection_confidence)

    body_type, body_detector = None, None
    if crop_mode == "body" or occlusion_threshold > 0:
        body_type, body_detector = _init_body_detector(detection_confidence)
        
    hand_detector = None
    if occlusion_threshold > 0:
        try:
            from pipeline.hand_detector import HandDetector
            hand_detector = HandDetector(max_hands=4)
        except Exception as e:
            print(f"[ClearShot] Hand detector unavailable ({e})")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_paths = []
    seen_hashes = set()

    stats = {
        "total_sampled": 0,
        "blurry_discarded": 0,
        "low_resolution_discarded": 0,
        "no_face_discarded": 0,
        "duplicate_discarded": 0,
        "occluded_discarded": 0,
        "extracted": 0,
        "gpu_backend": _active_backend(face_type, face_detector, body_type, body_detector),
    }

    ext = "png" if output_format == "png" else "jpg"
    
    import concurrent.futures
    # Use a thread pool to write images asynchronously, unblocking the GPU/CPU inference loop
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    write_futures: list[concurrent.futures.Future] = []
    
    # -----------------------------------------------------------------------
    # Async Video Decoding
    # -----------------------------------------------------------------------
    frame_q = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    
    def frame_reader():
        frame_idx = 0
        try:
            while not stop_event.is_set():
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
                
                # Block if the inference loop is slower than the decoding loop
                while not stop_event.is_set():
                    try:
                        frame_q.put((frame_idx, frame), timeout=0.1)
                        break
                    except queue.Full:
                        pass
                frame_idx += 1
        finally:
            try:
                frame_q.put(None, timeout=0.1)
            except queue.Full:
                pass
            
    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    try:
        while True:
            item = frame_q.get()
            if item is None:
                break
                
            frame_idx, frame = item
            body_detections = None

            stats["total_sampled"] += 1

            # Progress update
            if progress_callback and frame_count > 0:
                pct = min(frame_idx / frame_count, 1.0)
                progress_callback(
                    pct,
                    f"Detecting faces — frame {frame_idx}/{frame_count} | Extracted: {stats['extracted']}"
                )

            # --- Face detection ---
            face_bboxes = _detect_faces(face_type, face_detector, frame, detection_confidence)

            if not face_bboxes:
                stats["no_face_discarded"] += 1
                frame_idx += 1
                continue

            # Process every detected face in the frame
            for det_idx, face in enumerate(face_bboxes):
                face_bbox = (face.x, face.y, face.w, face.h)
                

                    
                if min(_clamped_bbox_size(frame, face_bbox)) < _min_source_face_size(frame):
                    stats["low_resolution_discarded"] += 1
                    continue

                # --- Blur check on face region (prioritizing eyes) ---
                blur_score = compute_blur_score_roi(frame, face_bbox, face.keypoints)
                if blur_score < blur_threshold:
                    stats["blurry_discarded"] += 1
                    continue

                # --- Crop ---
                if crop_mode == "body" and body_detector is not None:
                    cropped = _crop_body(
                        body_type,
                        body_detector,
                        frame,
                        face_bbox,
                        padding_pct,
                        detection_confidence,
                    )
                    if cropped is None:
                        cropped = crop_face(frame, face_bbox, padding_pct * 2)
                else:
                    cropped = crop_face(frame, face_bbox, padding_pct)

                if cropped is None or cropped.size == 0:
                    continue

                if _is_low_resolution_source(frame, face_bbox, cropped):
                    stats["low_resolution_discarded"] += 1
                    continue

                crop_blur_score = compute_blur_score(cropped)
                if crop_blur_score < blur_threshold * 0.35:
                    stats["blurry_discarded"] += 1
                    continue

                # --- Square + resize (Do this before dedup to stabilize aspect ratio) ---
                squared = make_square(cropped, method=square_method)
                final = resize_square(squared, size=output_size)

                # --- Dedup ---
                if HAS_IMAGEHASH and dedup_threshold > 0:
                    # Compute perceptual hash on the stabilized 1:1 image
                    phash = _perceptual_hash(final)
                    if phash is not None:
                        is_dup = any(
                            (phash - h) < dedup_threshold for h in seen_hashes
                        )
                        if is_dup:
                            stats["duplicate_discarded"] += 1
                            continue
                else:
                    phash = None
                            
                # --- Final Occulsion Checks ---
                # We only run these heavy checks on faces that survived all other filters
                if occlusion_threshold > 0:
                    is_occluded = False
                    
                    # 1. Body Pose Check (Shoulders, Elbows, Wrists)
                    if body_detector is not None:
                        # Lazy-evaluate body poses once per frame
                        if body_detections is None:
                            body_detections = body_detector.detect(frame)
                            
                        # COCO: 5,6(shoulders), 7,8(elbows), 9,10(wrists)
                        occlusion_kps = [5, 6, 7, 8, 9, 10]
                        fx, fy, fw, fh = face_bbox
                        
                        for body in body_detections:
                            for kp_idx in occlusion_kps:
                                if kp_idx < len(body.keypoints):
                                    kp = body.keypoints[kp_idx]
                                    if kp.confidence > 0.4:
                                        if fx <= kp.x <= fx + fw and fy <= kp.y <= fy + fh:
                                            is_occluded = True
                                            break
                            if is_occluded: break

                    # 2. Precise Hand Check (Fingers)
                    if not is_occluded and hand_detector is not None:
                        hand_detections = hand_detector.detect(cropped)
                        if hand_detections:
                            is_occluded = True

                    if is_occluded:
                        stats["occluded_discarded"] += 1
                        continue
                        
                # --- Dedup (Hash Add) ---
                if HAS_IMAGEHASH and dedup_threshold > 0 and phash is not None:
                    seen_hashes.add(phash)


                # --- Save ---
                # OPTIMIZATION 2: Async I/O for saving images
                filename = f"frame_{frame_idx:06d}_face_{det_idx}.{ext}"
                out_path = os.path.join(output_dir, filename)
                write_futures.append(executor.submit(_write_image, out_path, final))
                output_paths.append(out_path)
                stats["extracted"] += 1

            frame_idx += 1

        for future in write_futures:
            future.result()

    finally:
        stop_event.set()  # Signal reader_thread to stop
        
        stats["gpu_backend"] = _active_backend(face_type, face_detector, body_type, body_detector)
        cap.release()
        _close_detector(face_type, face_detector)
        if body_detector is not None:
            _close_detector(body_type, body_detector)
        if hand_detector is not None:
            hand_detector.close()
        
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
) -> list[FaceDetection]:
    """
    Detect faces using either ONNX or MediaPipe detector.

    Returns list of FaceDetection objects.
    """
    fh, fw = frame.shape[:2]

    if detector_type == "onnx":
        return detector.detect(frame, confidence=confidence)
    else:
        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = detector.process(rgb)
        if not face_results.detections:
            return []
        bboxes = []
        for detection in face_results.detections:
            rbb = detection.location_data.relative_bounding_box
            bboxes.append(FaceDetection(
                x=int(rbb.xmin * fw),
                y=int(rbb.ymin * fh),
                w=int(rbb.width * fw),
                h=int(rbb.height * fh),
                score=detection.score[0] if detection.score else 1.0,
                keypoints=None
            ))
        return bboxes


def _crop_body(
    detector_type: str,
    detector,
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    padding_pct: float,
    detection_confidence: float,
) -> np.ndarray | None:
    """
    Crop body region using either ONNX or MediaPipe detector.
    """
    if detector_type == "onnx":
        bodies = detector.detect(frame, confidence=detection_confidence)
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


def _active_backend(
    face_type: str,
    face_detector,
    body_type: str | None,
    body_detector,
) -> str:
    """Return the detector backend currently in use, including runtime fallback."""
    backends = []
    if face_type == "onnx":
        backends.append(getattr(face_detector, "backend", "cpu"))
    else:
        backends.append("cpu")

    if body_detector is not None:
        if body_type == "onnx":
            backends.append(getattr(body_detector, "backend", "cpu"))
        else:
            backends.append("cpu")

    unique_backends = list(dict.fromkeys(backends))
    non_cpu = [backend for backend in unique_backends if backend != "cpu"]
    if not non_cpu:
        return "cpu"

    ordered = non_cpu + [backend for backend in unique_backends if backend == "cpu"]
    return "+".join(ordered)


def _close_detector(detector_type: str, detector):
    """Close a detector safely."""
    try:
        detector.close()
    except Exception:
        pass
