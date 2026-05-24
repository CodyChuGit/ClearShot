from __future__ import annotations

"""
Blur / sharpness detection using Laplacian variance.
"""

import cv2
import numpy as np


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute a sharpness score for an image region using Laplacian variance.

    Higher values = sharper image. Typical thresholds:
      - < 50:  very blurry
      - 50-100: somewhat blurry
      - > 100: acceptably sharp

    Args:
        image: BGR or grayscale numpy array.

    Returns:
        Laplacian variance (float). Higher = sharper.
    """
    if image is None or image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def is_sharp(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if an image region is sharp enough (above blur threshold)."""
    return compute_blur_score(image) >= threshold


def compute_blur_score_roi(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """
    Compute blur score on a specific region of interest.

    Args:
        frame: Full BGR frame.
        bbox: (x, y, w, h) bounding box of the ROI.

    Returns:
        Laplacian variance of the ROI.
    """
    x, y, w, h = bbox
    fh, fw = frame.shape[:2]

    # Clamp to frame bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = frame[y1:y2, x1:x2]
    return compute_blur_score(roi)
