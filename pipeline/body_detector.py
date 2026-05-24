from __future__ import annotations

"""
ONNX-based body/pose detection using YOLOv8-pose models.

Provides full-body bounding boxes with pose keypoints,
GPU-accelerated via ONNX Runtime (CUDA/CoreML/CPU).
"""

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.gpu import create_session, detect_gpu


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(str(Path.home()), ".clearshot", "models")

BODY_MODELS = {
    "yolov8n-pose": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.onnx",
        "filename": "yolov8n-pose.onnx",
        "input_size": (640, 640),
    },
}

DEFAULT_MODEL = "yolov8n-pose"

# COCO keypoint indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class BodyDetection:
    """A single detected body."""
    x: int          # Top-left x of body bbox
    y: int          # Top-left y
    w: int          # Width
    h: int          # Height
    score: float    # Detection confidence
    keypoints: list[Keypoint]  # 17 COCO keypoints


# ---------------------------------------------------------------------------
# Body Detector
# ---------------------------------------------------------------------------

class BodyDetector:
    """
    ONNX Runtime-based body detector using YOLOv8-pose.

    Usage:
        detector = BodyDetector()
        bodies = detector.detect(bgr_frame)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model_config = BODY_MODELS[model_name]
        self.input_size = self.model_config["input_size"]

        model_path = self._ensure_model()
        self.session = create_session(model_path)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        gpu_info = detect_gpu()
        self.backend = gpu_info["backend"]

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> list[BodyDetection]:
        """
        Detect bodies with pose keypoints in a BGR frame.

        Args:
            frame: BGR image (numpy array).
            confidence: Minimum detection confidence.
            nms_threshold: NMS IoU threshold.

        Returns:
            List of BodyDetection objects.
        """
        fh, fw = frame.shape[:2]

        # Preprocess with letterbox
        blob, ratio, (pad_w, pad_h) = self._preprocess(frame)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Post-process
        bodies = self._postprocess(
            outputs[0], fw, fh, ratio, pad_w, pad_h,
            confidence, nms_threshold,
        )

        return bodies

    def close(self):
        """Release ONNX session."""
        del self.session

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _ensure_model(self) -> str:
        """Download model if not present."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, self.model_config["filename"])

        if not os.path.exists(model_path):
            print(f"[ClearShot] Downloading {self.model_name} model...")
            url = self.model_config["url"]
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"[ClearShot] Model saved to {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model from {url}: {e}\n"
                    f"Please download manually and place at {model_path}"
                )

        return model_path

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Letterbox resize, normalize, convert to NCHW."""
        iw, ih = self.input_size
        fh, fw = frame.shape[:2]

        # Compute scale maintaining aspect ratio
        ratio = min(iw / fw, ih / fh)
        new_w = int(fw * ratio)
        new_h = int(fh * ratio)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = (iw - new_w) / 2.0
        pad_h = (ih - new_h) / 2.0

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        img = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC → CHW → NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        return img, ratio, (pad_w, pad_h)

    def _postprocess(
        self,
        output: np.ndarray,
        orig_w: int,
        orig_h: int,
        ratio: float,
        pad_w: float,
        pad_h: float,
        confidence: float,
        nms_threshold: float,
    ) -> list[BodyDetection]:
        """
        Decode YOLOv8-pose output.

        Output shape: [1, 56, N] where 56 = 4 (bbox) + 1 (conf) + 51 (17 keypoints × 3)
        """
        # Transpose to [N, 56]
        if output.shape[0] == 1:
            output = output[0]
        if output.shape[0] == 56:
            output = output.T

        num_detections = output.shape[0]
        if num_detections == 0:
            return []

        # Split output
        boxes_raw = output[:, :4]     # cx, cy, w, h
        scores = output[:, 4]         # confidence
        keypoints_raw = output[:, 5:]  # 17 × 3 (x, y, conf)

        # Filter by confidence
        mask = scores > confidence
        if not mask.any():
            return []

        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        keypoints_raw = keypoints_raw[mask]

        # Convert cx,cy,w,h → x1,y1,w,h and scale back to original image
        all_boxes = []
        all_scores = []
        all_keypoints = []

        for i in range(len(boxes_raw)):
            cx, cy, bw, bh = boxes_raw[i]

            # Remove padding and scale
            x1 = (cx - bw / 2 - pad_w) / ratio
            y1 = (cy - bh / 2 - pad_h) / ratio
            w = bw / ratio
            h = bh / ratio

            # Clamp
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            w = min(orig_w - x1, int(w))
            h = min(orig_h - y1, int(h))

            if w < 10 or h < 10:
                continue

            all_boxes.append([x1, y1, w, h])
            all_scores.append(float(scores[i]))

            # Parse keypoints
            kps = []
            kp_data = keypoints_raw[i]
            for k in range(17):
                kx = (kp_data[k * 3] - pad_w) / ratio
                ky = (kp_data[k * 3 + 1] - pad_h) / ratio
                kc = float(kp_data[k * 3 + 2])
                kps.append(Keypoint(x=float(kx), y=float(ky), confidence=kc))

            all_keypoints.append(kps)

        if not all_boxes:
            return []

        # NMS
        indices = cv2.dnn.NMSBoxes(
            [list(map(float, b)) for b in all_boxes],
            all_scores,
            confidence,
            nms_threshold,
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = all_boxes[i]
                results.append(BodyDetection(
                    x=x, y=y, w=w, h=h,
                    score=all_scores[i],
                    keypoints=all_keypoints[i],
                ))

        return results
