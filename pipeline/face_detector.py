from __future__ import annotations

"""
ONNX-based face detection using SCRFD models.

Replaces MediaPipe Face Detection with GPU-accelerated ONNX Runtime inference.
Supports CUDA (NVIDIA), CoreML (Apple Metal), and CPU backends.
"""

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.gpu import create_cpu_session, create_session, detect_gpu


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(str(Path.home()), ".clearshot", "models")

MODELS = {
    "scrfd_500m": {
        "url": "https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx", # Fallback to 2.5g
        "filename": "scrfd_500m.onnx",
        "input_size": (640, 640),
    },
    "scrfd_2.5g": {
        "url": "https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx",
        "filename": "scrfd_2.5g.onnx",
        "input_size": (640, 640),
    },
    "scrfd_10g": {
        "url": "https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx", # Fallback to 2.5g
        "filename": "scrfd_10g.onnx",
        "input_size": (640, 640),
    },
}

DEFAULT_MODEL = "scrfd_2.5g"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    """A single detected face."""
    x: int       # Top-left x
    y: int       # Top-left y
    w: int       # Width
    h: int       # Height
    score: float # Detection confidence


# ---------------------------------------------------------------------------
# Face Detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    ONNX Runtime-based face detector using SCRFD models.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(bgr_frame, confidence=0.5)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model_config = MODELS[model_name]
        self.input_size = self.model_config["input_size"]

        self.model_path = self._ensure_model()
        self.session = create_session(self.model_path)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        gpu_info = detect_gpu()
        active_provider = self.session.get_providers()[0]
        self.backend = _provider_backend(active_provider)
        self.device = gpu_info["device"] if self.backend != "cpu" else "CPU"

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[FaceDetection]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image (numpy array).
            confidence: Minimum detection confidence.
            nms_threshold: NMS IoU threshold.

        Returns:
            List of FaceDetection objects.
        """
        fh, fw = frame.shape[:2]

        # Preprocess
        blob, ratio, (pad_w, pad_h) = self._preprocess(frame)

        # Inference
        outputs = self._run(blob)

        # Post-process
        faces = self._postprocess(
            outputs,
            fw,
            fh,
            confidence,
            nms_threshold,
            ratio,
            pad_w,
            pad_h,
        )

        return faces

    def close(self):
        """Release ONNX session resources."""
        del self.session

    def _run(self, blob: np.ndarray) -> list[np.ndarray]:
        try:
            return self.session.run(self.output_names, {self.input_name: blob})
        except Exception as exc:
            if self.backend == "cpu":
                raise

            print(f"[ClearShot] Face detector GPU inference failed ({exc}); retrying on CPU")
            self.session = create_cpu_session(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.backend = "cpu"
            self.device = "CPU"
            return self.session.run(self.output_names, {self.input_name: blob})

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _ensure_model(self) -> str:
        """Download model if not present, return path."""
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
        """Letterbox resize, normalize, and convert to NCHW format."""
        iw, ih = self.input_size
        fh, fw = frame.shape[:2]

        ratio = min(iw / fw, ih / fh)
        new_w = max(1, int(fw * ratio))
        new_h = max(1, int(fh * ratio))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (iw - new_w) / 2.0
        pad_h = (ih - new_h) / 2.0

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        img = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        img = img.astype(np.float32)

        # Normalize (standard ImageNet-style)
        img = (img - 127.5) / 128.0

        # HWC → CHW → NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        return img, ratio, (pad_w, pad_h)

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        orig_w: int,
        orig_h: int,
        confidence: float,
        nms_threshold: float,
        ratio: float,
        pad_w: float,
        pad_h: float,
    ) -> list[FaceDetection]:
        """
        Decode SCRFD outputs into face detections.

        SCRFD outputs vary by model, but generally produce:
        - Score maps at 3 stride levels (8, 16, 32)
        - Bounding box regressions at each level
        - Optional keypoint regressions
        """
        iw, ih = self.input_size

        all_boxes = []
        all_scores = []

        # SCRFD has 3 stride levels, each with score + bbox outputs
        # Output ordering: [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32]
        # Or without keypoints: [score_8, bbox_8, score_16, bbox_16, score_32, bbox_32]

        num_outputs = len(outputs)

        # Determine if model includes keypoints
        if num_outputs >= 9:
            # Outputs: 0,1,2 (scores), 3,4,5 (bboxes), 6,7,8 (keypoints)
            stride_outputs = [
                (outputs[0], outputs[3], 8),
                (outputs[1], outputs[4], 16),
                (outputs[2], outputs[5], 32),
            ]
        elif num_outputs >= 6:
            # Outputs: 0,1,2 (scores), 3,4,5 (bboxes)
            stride_outputs = [
                (outputs[0], outputs[3], 8),
                (outputs[1], outputs[4], 16),
                (outputs[2], outputs[5], 32),
            ]
        else:
            # Fallback: try to parse whatever we get
            return []

        for scores_raw, bboxes_raw, stride in stride_outputs:
            scores = scores_raw.reshape(-1)
            bboxes = bboxes_raw.reshape(-1, 4)

            if bboxes.shape[0] != scores.shape[0]:
                count = min(bboxes.shape[0], scores.shape[0])
                scores = scores[:count]
                bboxes = bboxes[:count]

            # Filter by confidence
            mask = scores > confidence
            if not mask.any():
                continue

            filtered_scores = scores[mask]
            filtered_indices = np.where(mask)[0]

            # Grid dimensions for this stride level
            grid_h = ih // stride
            grid_w = iw // stride
            grid_cells = grid_h * grid_w
            anchor_count = max(1, len(scores) // grid_cells) if grid_cells else 1

            for idx, score in zip(filtered_indices, filtered_scores):
                # Grid position
                spatial_idx = int(idx) // anchor_count
                if spatial_idx >= grid_cells:
                    continue

                row = spatial_idx // grid_w
                col = spatial_idx % grid_w

                # Anchor center
                cx = (col + 0.5) * stride
                cy = (row + 0.5) * stride

                # Decode bbox (distance from anchor: left, top, right, bottom)
                bbox = bboxes[idx, :]

                x1 = (cx - bbox[0] * stride - pad_w) / ratio
                y1 = (cy - bbox[1] * stride - pad_h) / ratio
                x2 = (cx + bbox[2] * stride - pad_w) / ratio
                y2 = (cy + bbox[3] * stride - pad_h) / ratio

                # Clamp to image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(orig_w, int(x2))
                y2 = min(orig_h, int(y2))

                w = x2 - x1
                h = y2 - y1

                if w > 5 and h > 5:
                    all_boxes.append([x1, y1, w, h])
                    all_scores.append(float(score))

        if not all_boxes:
            return []

        # Apply NMS
        boxes_array = np.array(all_boxes, dtype=np.float32)
        scores_array = np.array(all_scores, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes_array.tolist(),
            scores_array.tolist(),
            confidence,
            nms_threshold,
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = all_boxes[i]
                results.append(FaceDetection(
                    x=x, y=y, w=w, h=h,
                    score=all_scores[i],
                ))

        return results


def _provider_backend(provider: str) -> str:
    if provider == "CUDAExecutionProvider":
        return "cuda"
    if provider == "CoreMLExecutionProvider":
        return "coreml"
    return "cpu"
