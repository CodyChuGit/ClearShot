from __future__ import annotations

"""
Blur / sharpness detection using Laplacian variance.
"""

import cv2
import numpy as np


def compute_blur_score(image: np.ndarray, target_size: int = 128) -> float:
    """
    Compute a scale-invariant sharpness score for an image region.

    Higher values = sharper image.
    Args:
        image: BGR or grayscale numpy array.
        target_size: Square dimension to normalize the spatial frequency.

    Returns:
        Normalized Laplacian variance (float).
    """
    if image is None or image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Force Scale Invariance
    resized = cv2.resize(gray, (target_size, target_size))
    
    # Filter out ISO camera grain/noise
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return float(laplacian.var())


def is_sharp(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if an image region is sharp enough (above blur threshold)."""
    return compute_blur_score(image) >= threshold


def compute_blur_score_roi(frame: np.ndarray, bbox: tuple[int, int, int, int], keypoints: list[tuple[float, float]] | None = None) -> float:
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

    # If we have facial keypoints (SCRFD: 0=LeftEye, 1=RightEye)
    if keypoints and len(keypoints) >= 2:
        le = keypoints[0]
        re = keypoints[1]
        
        # Inter-ocular distance
        iod = np.sqrt((re[0] - le[0])**2 + (re[1] - le[1])**2)
        eye_box_size = int(max(iod * 0.6, 10))  # At least 10px box
        
        scores = []
        for ex, ey in [le, re]:
            ex1 = max(0, int(ex - eye_box_size / 2))
            ey1 = max(0, int(ey - eye_box_size / 2))
            ex2 = min(fw, int(ex + eye_box_size / 2))
            ey2 = min(fh, int(ey + eye_box_size / 2))
            
            if ex2 > ex1 and ey2 > ey1:
                eye_roi = frame[ey1:ey2, ex1:ex2]
                # Keep the scale-invariance for the eye crops!
                scores.append(compute_blur_score(eye_roi, target_size=64))
                
        if scores:
            # We want both eyes to be reasonably sharp, so we take the average
            eye_score = sum(scores) / len(scores)
            
            # Combine eye score with overall face score, heavily weighting the eyes (70/30)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2]
                # Keep the scale-invariance for the face crop!
                face_score = compute_blur_score(face_roi, target_size=128)
                return (eye_score * 0.7) + (face_score * 0.3)
            return eye_score

    # Standard bounding box extraction fallback
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = frame[y1:y2, x1:x2]
    return compute_blur_score(roi, target_size=128)
