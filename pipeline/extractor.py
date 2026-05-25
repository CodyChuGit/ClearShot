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
    meta = _probe_ffprobe(path)
    if meta is None:
        meta = _probe_opencv(path)
    return meta


def _probe_ffprobe(path: str) -> dict | None:
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
# Helpers
# ---------------------------------------------------------------------------

def _perceptual_hash(image: np.ndarray) -> object | None:
    if not HAS_IMAGEHASH:
        return None
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil, hash_size=8)


def _remove_previous_outputs(output_dir: str) -> None:
    for pattern in ("frame_*.png", "frame_*.jpg", "frame_*.jpeg"):
        for path in Path(output_dir).glob(pattern):
            if path.is_file():
                path.unlink()


def _write_image(path: str, image: np.ndarray) -> None:
    ok = cv2.imwrite(path, image)
    if not ok:
        raise RuntimeError(f"Failed to write extracted image: {path}")


def _min_source_face_size(frame: np.ndarray) -> int:
    short_edge = min(frame.shape[:2])
    return int(min(96, max(48, round(short_edge * 0.12))))


def _min_source_crop_size(frame: np.ndarray) -> int:
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
# Detector abstractions
# ---------------------------------------------------------------------------

def _init_face_detector(detection_confidence: float):
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


def _detect_faces(
    detector_type: str,
    detector,
    frame: np.ndarray,
    confidence: float,
) -> list[FaceDetection]:
    fh, fw = frame.shape[:2]

    if detector_type == "onnx":
        return detector.detect(frame, confidence=confidence)
    else:
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
    if detector_type == "onnx":
        bodies = detector.detect(frame, confidence=detection_confidence)
        if bodies:
            face_cx = face_bbox[0] + face_bbox[2] / 2
            face_cy = face_bbox[1] + face_bbox[3] / 2
            best_body = min(bodies, key=lambda b: (
                (b.x + b.w / 2 - face_cx) ** 2 +
                (b.y + b.h / 2 - face_cy) ** 2
            ))
            return crop_body_from_keypoints(
                frame,
                [(kp.x, kp.y, kp.confidence) for kp in best_body.keypoints],
                padding_pct,
            )
        return None
    else:
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
    try:
        detector.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OOP Extraction Pipeline
# ---------------------------------------------------------------------------

class VideoExtractor:
    """
    Object-oriented extraction pipeline that safely manages detector state,
    background threads, and async I/O.
    """
    
    def __init__(
        self,
        target_fps: float = 2.0,
        blur_threshold: float = 100.0,
        detection_confidence: float = 0.5,
        crop_mode: str = "face",
        padding_pct: float = 0.2,
        square_method: str = "center_crop",
        output_size: int = 512,
        output_format: str = "png",
        dedup_threshold: int = 18,
        occlusion_threshold: int = 50,
    ):
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than 0")
        if output_size <= 0:
            raise ValueError("output_size must be greater than 0")
        if crop_mode not in {"face", "body"}:
            raise ValueError("crop_mode must be 'face' or 'body'")
        if square_method not in {"center_crop", "letterbox"}:
            raise ValueError("square_method must be 'center_crop' or 'letterbox'")
        if output_format.lower() not in {"png", "jpg", "jpeg"}:
            raise ValueError("output_format must be 'png' or 'jpg'")

        self.target_fps = target_fps
        self.blur_threshold = blur_threshold
        self.detection_confidence = detection_confidence
        self.crop_mode = crop_mode
        self.padding_pct = padding_pct
        self.square_method = square_method
        self.output_size = output_size
        self.output_format = output_format.lower()
        self.dedup_threshold = dedup_threshold
        self.occlusion_threshold = occlusion_threshold
        
        # State
        self.face_type = None
        self.face_detector = None
        self.body_type = None
        self.body_detector = None
                
        self.seen_hashes = set()
        self.stats = {
            "total_sampled": 0,
            "blurry_discarded": 0,
            "low_resolution_discarded": 0,
            "no_face_discarded": 0,
            "duplicate_discarded": 0,
            "occluded_discarded": 0,
            "extracted": 0,
            "gpu_backend": "cpu",
        }
        
        self.executor = None
        self.write_futures = []
        self.stop_event = threading.Event()

    def __enter__(self):
        """Initialize detectors and thread pools."""
        self.face_type, self.face_detector = _init_face_detector(self.detection_confidence)
        
        if self.crop_mode == "body" or self.occlusion_threshold > 0:
            self.body_type, self.body_detector = _init_body_detector(self.detection_confidence)
            
        
                
        self.stats["gpu_backend"] = _active_backend(
            self.face_type, self.face_detector, self.body_type, self.body_detector
        )
        
        import concurrent.futures
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Safely clean up threads and C++ model resources."""
        self.stop_event.set()
        
        if self.face_detector:
            _close_detector(self.face_type, self.face_detector)
        if self.body_detector:
            _close_detector(self.body_type, self.body_detector)
        
                
        if self.executor:
            self.executor.shutdown(wait=True)

    def _check_dedup(self, image: np.ndarray) -> tuple[bool, object | None]:
        """Returns (is_duplicate, perceptual_hash)."""
        if not HAS_IMAGEHASH or self.dedup_threshold <= 0:
            return False, None
            
        phash = _perceptual_hash(image)
        if phash is None:
            return False, None
            
        is_dup = any((phash - h) < self.dedup_threshold for h in self.seen_hashes)
        return is_dup, phash

        
    def _check_occlusion(
        self, 
        frame, 
        face_bbox, 
        face_keypoints,
        body_detections
    ) -> tuple[bool, list | None]:
        if self.occlusion_threshold <= 0:
            return False, body_detections
            
        is_occluded = False
        fx, fy, fw, fh = face_bbox
        
        scale_factor = (self.occlusion_threshold / 50.0) * 2.7
        
        # 1. Body Pose Check (Shoulders, Elbows, Wrists overlapping face)
        if self.body_detector is not None:
            if body_detections is None:
                body_detections = self.body_detector.detect(frame)
                
            occlusion_kps = [5, 6, 7, 8, 9, 10]
            bx_margin = fw * 0.15 * (scale_factor - 1.0)
            by_margin = fh * 0.15 * (scale_factor - 1.0)
            
            bx1 = fx - bx_margin
            by1 = fy - by_margin
            bx2 = fx + fw + bx_margin
            by2 = fy + fh + by_margin
            
            for body in body_detections:
                for kp_idx in occlusion_kps:
                    if kp_idx < len(body.keypoints):
                        kp = body.keypoints[kp_idx]
                        if kp.confidence > 0.4:
                            if bx1 <= kp.x <= bx2 and by1 <= kp.y <= by2:
                                is_occluded = True
                                break
                if is_occluded: break
                
        # 2. Scale-Invariant Laplacian Texture Check
        # Instead of a brittle multi-feature ensemble, we use a single, highly-normalized texture metric.
        if not is_occluded and face_keypoints is not None and len(face_keypoints) == 5:
            import cv2
            import numpy as np
            mx1, my1 = face_keypoints[3]
            mx2, my2 = face_keypoints[4]
            
            cx, cy = (mx1 + mx2) / 2, (my1 + my2) / 2
            mw = max(abs(mx2 - mx1), 10)
            mh = mw * 1.5
            
            x1, y1 = max(0, int(cx - mw)), max(0, int(cy - mh/2))
            x2, y2 = min(frame.shape[1], int(cx + mw)), min(frame.shape[0], int(cy + mh/2))
            
            mouth_crop = frame[y1:y2, x1:x2]
            if mouth_crop.size > 0:
                gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
                # Resize to exactly 64x64 to make the Laplacian perfectly scale-invariant
                resized = cv2.resize(gray, (64, 64))
                lap_var = cv2.Laplacian(resized, cv2.CV_64F).var()
                
                # Normal face = ~150. Microphone/hand = > 700.
                # If threshold is 50 (default), allowed = 400.
                # If threshold is 100 (strictest), allowed = 150 (requires perfectly smooth skin).
                allowed_var = 650.0 - (self.occlusion_threshold * 5.0)
                
                if lap_var > allowed_var:
                    is_occluded = True
                    
        return is_occluded, body_detections

    def _process_face(
        self, 
        frame: np.ndarray, 
        face_bbox: tuple[int, int, int, int], 
        face_keypoints, 
        body_detections: list | None,
        
        frame_idx: int,
        det_idx: int,
        output_dir: str,
        ext: str
    ) -> tuple[list | None, list | None]:
        """Processes a single detected face, applying all filters and saving it if valid."""
        if min(_clamped_bbox_size(frame, face_bbox)) < _min_source_face_size(frame):
            self.stats["low_resolution_discarded"] += 1
            return body_detections
            
        # Map the UI 0-100% slider to the actual scale-invariant variance range (0 to ~50)
        # 0% = 0.0 (allow all)
        # 50% = 20.0 (sensible default for web video)
        # 100% = 40.0 (requires studio lighting/macro lens)
        mapped_blur_thresh = (self.blur_threshold / 100.0) * 40.0
        
        blur_score = compute_blur_score_roi(frame, face_bbox, face_keypoints)
        if blur_score < mapped_blur_thresh:
            self.stats["blurry_discarded"] += 1
            return body_detections
            
        if self.crop_mode == "body" and self.body_detector is not None:
            cropped = _crop_body(self.body_type, self.body_detector, frame, face_bbox, self.padding_pct, self.detection_confidence)
            if cropped is None:
                cropped = crop_face(frame, face_bbox, self.padding_pct * 2)
        else:
            cropped = crop_face(frame, face_bbox, self.padding_pct)
            
        if cropped is None or cropped.size == 0:
            return body_detections
            
        if _is_low_resolution_source(frame, face_bbox, cropped):
            self.stats["low_resolution_discarded"] += 1
            return body_detections
            
        crop_blur_score = compute_blur_score(cropped)
        if crop_blur_score < mapped_blur_thresh * 0.35:
            self.stats["blurry_discarded"] += 1
            return body_detections
            
        # Square + resize (done before dedup to stabilize aspect ratio)
        squared = make_square(cropped, method=self.square_method)
        final = resize_square(squared, size=self.output_size)
        
        # Dedup Check
        is_dup, phash = self._check_dedup(final)
        if is_dup:
            self.stats["duplicate_discarded"] += 1
            return body_detections
            
        # Final Occlusion Check
        is_occluded, body_detections = self._check_occlusion(frame, face_bbox, face_keypoints, body_detections)
        if is_occluded:
            self.stats["occluded_discarded"] += 1
            return body_detections
            
        # Add to seen hashes ONLY if it survives all filters
        if phash is not None:
            self.seen_hashes.add(phash)
            
        # Save Async
        filename = f"frame_{frame_idx:06d}_face_{det_idx}.{ext}"
        out_path = os.path.join(output_dir, filename)
        
        future = self.executor.submit(_write_image, out_path, final)
        self.write_futures.append((out_path, future))
        self.stats["extracted"] += 1
        
        return body_detections

    def extract(self, video_path: str, output_dir: str, progress_callback: Callable[[float, str], None] | None = None) -> tuple[list[str], dict]:
        """
        Orchestrator: runs async video decoding and evaluates frames.
        """
        os.makedirs(output_dir, exist_ok=True)
        _remove_previous_outputs(output_dir)
        
        meta = probe_video(video_path)
        video_fps = meta["fps"]
        frame_count = meta["frame_count"]
        frame_interval = max(1, int(round(video_fps / self.target_fps)))
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
            
        frame_q = queue.Queue(maxsize=4)
        
        def frame_reader():
            frame_idx = 0
            try:
                while not self.stop_event.is_set():
                    if frame_idx % frame_interval != 0:
                        ret = cap.grab()
                        if not ret: break
                        frame_idx += 1
                        continue
                        
                    ret, frame = cap.read()
                    if not ret: break
                    
                    while not self.stop_event.is_set():
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
        
        ext = "png" if self.output_format == "png" else "jpg"
        
        try:
            while True:
                item = frame_q.get()
                if item is None:
                    break
                    
                frame_idx, frame = item
                self.stats["total_sampled"] += 1
                
                if progress_callback and frame_count > 0:
                    pct = min(frame_idx / frame_count, 1.0)
                    progress_callback(pct, f"Detecting faces — frame {frame_idx}/{frame_count} | Extracted: {self.stats['extracted']}")
                    
                face_bboxes = _detect_faces(self.face_type, self.face_detector, frame, self.detection_confidence)
                
                if not face_bboxes:
                    self.stats["no_face_discarded"] += 1
                    continue
                    
                body_detections = None
                hand_detections = None
                for det_idx, face in enumerate(face_bboxes):
                    face_bbox = (face.x, face.y, face.w, face.h)
                    body_detections = self._process_face(frame, face_bbox, face.keypoints, body_detections, frame_idx, det_idx, output_dir, ext)
                    
        finally:
            cap.release()
            
        output_paths = []
        for out_path, future in self.write_futures:
            future.result()
            output_paths.append(out_path)
            
        if progress_callback:
            progress_callback(1.0, f"Done! Extracted {self.stats['extracted']} images.")
            
        return output_paths, self.stats


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
    dedup_threshold: int = 18,         # hamming distance for dedup
    occlusion_threshold: int = 50,    # 0 to 100, 0 = off
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[str], dict]:
    """
    Backwards compatibility wrapper for VideoExtractor.
    """
    with VideoExtractor(
        target_fps=target_fps,
        blur_threshold=blur_threshold,
        detection_confidence=detection_confidence,
        crop_mode=crop_mode,
        padding_pct=padding_pct,
        square_method=square_method,
        output_size=output_size,
        output_format=output_format,
        dedup_threshold=dedup_threshold,
        occlusion_threshold=occlusion_threshold,
    ) as extractor:
        return extractor.extract(video_path, output_dir, progress_callback)
