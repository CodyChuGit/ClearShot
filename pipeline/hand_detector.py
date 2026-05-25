import os
import urllib.request
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_DIR = os.path.join(str(Path.home()), ".clearshot", "models")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

@dataclass
class HandDetection:
    """A single detected hand."""
    landmarks: list[tuple[float, float]] # (x, y) normalized [0.0, 1.0]

class HandDetector:
    """
    MediaPipe Hand Landmarker wrapper.
    """

    def __init__(self, max_hands: int = 2):
        self.model_path = self._ensure_model()
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        """
        Detect hands in a BGR frame.
        
        Args:
            frame: BGR image (numpy array).
            
        Returns:
            List of HandDetection objects.
        """
        # MediaPipe expects RGB
        rgb_frame = frame[:, :, ::-1] # BGR to RGB
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Inference
        result = self.detector.detect(mp_image)
        
        hands = []
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                points = [(lm.x, lm.y) for lm in hand_lms]
                hands.append(HandDetection(landmarks=points))
                
        return hands

    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()

    def _ensure_model(self) -> str:
        """Download model if not present, return path."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

        if not os.path.exists(model_path):
            print(f"[ClearShot] Downloading MediaPipe Hand Landmarker model...")
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print(f"[ClearShot] Model saved to {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model from {MODEL_URL}: {e}\n"
                    f"Please download manually and place at {model_path}"
                )

        return model_path
