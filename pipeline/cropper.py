from __future__ import annotations

"""
Cropping utilities: face crop, body crop, square formatting, and resize.
"""

import cv2
import numpy as np


def _normalize_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return a bbox with positive width/height."""
    x, y, w, h = bbox
    if w < 0:
        x += w
        w = abs(w)
    if h < 0:
        y += h
        h = abs(h)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def expand_bbox(
    bbox: tuple[int, int, int, int],
    padding_pct: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """
    Expand a bounding box by a padding percentage, clamped to frame bounds.

    Args:
        bbox: (x, y, w, h)
        padding_pct: Padding as a fraction (e.g., 0.2 = 20%).
        frame_w: Frame width.
        frame_h: Frame height.

    Returns:
        Expanded (x, y, w, h).
    """
    x, y, w, h = _normalize_bbox(bbox)
    pad_x = int(w * padding_pct)
    pad_y = int(h * padding_pct)

    x1 = max(0, min(frame_w, x - pad_x))
    y1 = max(0, min(frame_h, y - pad_y))
    x2 = max(0, min(frame_w, x + w + pad_x))
    y2 = max(0, min(frame_h, y + h + pad_y))

    return (x1, y1, x2 - x1, y2 - y1)


def crop_region(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a region from a frame given (x, y, w, h)."""
    fh, fw = frame.shape[:2]
    x, y, w, h = _normalize_bbox(bbox)

    x1 = max(0, min(fw, x))
    y1 = max(0, min(fh, y))
    x2 = max(0, min(fw, x + w))
    y2 = max(0, min(fh, y + h))

    if x2 <= x1 or y2 <= y1:
        empty_shape = (0, 0) if frame.ndim == 2 else (0, 0, frame.shape[2])
        return np.empty(empty_shape, dtype=frame.dtype)

    return frame[y1:y2, x1:x2].copy()


def crop_face(
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    padding_pct: float = 0.2,
) -> np.ndarray:
    """
    Crop the face region with padding.

    Args:
        frame: Full BGR frame.
        face_bbox: (x, y, w, h) of the detected face.
        padding_pct: How much to expand the box (0.2 = 20%).

    Returns:
        Cropped face image.
    """
    fh, fw = frame.shape[:2]
    expanded = expand_bbox(face_bbox, padding_pct, fw, fh)
    return crop_region(frame, expanded)


def crop_body(
    frame: np.ndarray,
    landmarks: list,
    padding_pct: float = 0.1,
) -> np.ndarray | None:
    """
    Crop the full body region from MediaPipe Pose landmarks.

    Args:
        frame: Full BGR frame.
        landmarks: List of MediaPipe pose landmarks (normalized coords).
        padding_pct: How much to expand the box.

    Returns:
        Cropped body image, or None if landmarks are insufficient.
    """
    fh, fw = frame.shape[:2]

    # Filter landmarks with reasonable visibility
    visible = [lm for lm in landmarks if lm.visibility > 0.3]
    if len(visible) < 5:
        return None

    x_coords = [lm.x * fw for lm in visible]
    y_coords = [lm.y * fh for lm in visible]

    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))

    w = x_max - x_min
    h = y_max - y_min

    if w < 10 or h < 10:
        return None

    bbox = (x_min, y_min, w, h)
    expanded = expand_bbox(bbox, padding_pct, fw, fh)
    return crop_region(frame, expanded)


def crop_body_from_keypoints(
    frame: np.ndarray,
    keypoints: list[tuple[float, float, float]],
    padding_pct: float = 0.1,
) -> np.ndarray | None:
    """
    Crop the full body region from ONNX detector keypoints.

    Args:
        frame: Full BGR frame.
        keypoints: List of (x, y, confidence) tuples in pixel coordinates.
        padding_pct: How much to expand the box.

    Returns:
        Cropped body image, or None if keypoints are insufficient.
    """
    fh, fw = frame.shape[:2]

    # Filter keypoints with reasonable confidence
    visible = [(x, y) for x, y, c in keypoints if c > 0.3]
    if len(visible) < 5:
        return None

    x_coords = [x for x, y in visible]
    y_coords = [y for x, y in visible]

    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))

    w = x_max - x_min
    h = y_max - y_min

    if w < 10 or h < 10:
        return None

    bbox = (x_min, y_min, w, h)
    expanded = expand_bbox(bbox, padding_pct, fw, fh)
    return crop_region(frame, expanded)


def make_square(
    image: np.ndarray,
    method: str = "center_crop",
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Convert an image to a square aspect ratio.

    Args:
        image: Input BGR image (any aspect ratio).
        method: "center_crop" (cut longer dim) or "letterbox" (pad shorter dim).
        bg_color: Background color for letterbox padding.

    Returns:
        Square image.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot make an empty image square")

    if method == "center_crop":
        size = min(h, w)
        y_off = (h - size) // 2
        x_off = (w - size) // 2
        return image[y_off : y_off + size, x_off : x_off + size].copy()

    elif method == "letterbox":
        size = max(h, w)
        canvas = np.full((size, size, 3), bg_color, dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = image
        return canvas

    else:
        raise ValueError(f"Unknown square method: {method!r}. Use 'center_crop' or 'letterbox'.")


def resize_square(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize a (presumably square) image to size×size."""
    h = image.shape[0]
    # INTER_AREA is mathematically correct and faster for downscaling.
    # INTER_LINEAR is much faster than Lanczos for upscaling with acceptable quality.
    interpolation = cv2.INTER_AREA if size < h else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)
