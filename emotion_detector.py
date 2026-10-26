"""
emotion_detector.py
-------------------
Face detection and facial-expression estimation for the Real-Time Human
Emotion Detection application.

This module wraps the FER library (a pretrained CNN trained on FER-2013) and
exposes a small, GUI-independent API:

    detector = EmotionDetector()
    faces = detector.analyze(frame_bgr)      # list[FaceResult], largest face first

The output is an ESTIMATE of the visible facial expression only. It is not a
reliable measurement of what a person actually feels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Display names, in the order they are shown in the interface.
EMOTIONS: List[str] = [
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
    "Neutral",
]

# FER uses its own label names; map them to the display names above.
_FER_TO_DISPLAY: Dict[str, str] = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fearful",
    "disgust": "Disgust",
    "surprise": "Surprised",
    "neutral": "Neutral",
}


class EmotionDetectorError(RuntimeError):
    """Raised when the emotion model cannot be created or used."""


@dataclass
class FaceResult:
    """One detected face with its seven emotion scores (sum ~= 1.0)."""

    box: Tuple[int, int, int, int]  # x, y, width, height in original frame pixels
    scores: Dict[str, float]

    @property
    def top_emotion(self) -> str:
        return max(self.scores, key=self.scores.get)

    @property
    def confidence(self) -> float:
        return self.scores[self.top_emotion]


def _import_fer_class():
    """Import the FER class, supporting both known package layouts."""
    errors = []
    try:
        from fer import FER  # type: ignore  # fer==22.x layout
        return FER
    except Exception as exc:  # ImportError, but also TF / moviepy load errors
        errors.append(f"'from fer import FER' failed: {type(exc).__name__}: {exc}")
    try:
        from fer.fer import FER  # type: ignore  # alternative layout
        return FER
    except Exception as exc:
        errors.append(f"'from fer.fer import FER' failed: {type(exc).__name__}: {exc}")

    raise EmotionDetectorError(
        "The FER library could not be imported.\n\n"
        + "\n".join(errors)
        + "\n\nMost common causes:\n"
        "  * Python version is too new (use Python 3.10 or 3.11).\n"
        "  * TensorFlow is not installed or does not match your Python version.\n"
        "  * moviepy 2.x is installed (use moviepy==1.0.3).\n"
        "  * NumPy 2.x is installed (use numpy==1.26.4).\n"
        "See the Troubleshooting section of README.md."
    )


class EmotionDetector:
    """Detects faces and estimates facial expressions using a pretrained model."""

    def __init__(self, max_width: int = 480) -> None:
        """
        max_width: frames wider than this are downscaled before analysis to keep
                   the application responsive. Boxes are scaled back afterwards.
        """
        self.max_width = max_width
        fer_class = _import_fer_class()
        try:
            # mtcnn=False -> use OpenCV's Haar cascade for face detection.
            # It is fast, needs no extra packages and works well on webcam frames.
            self._fer = fer_class(mtcnn=False)
        except Exception as exc:
            raise EmotionDetectorError(
                "The FER model could not be initialised "
                f"({type(exc).__name__}: {exc}).\n"
                "The pretrained model file ships inside the 'fer' package; try "
                "reinstalling it:  pip install --force-reinstall fer==22.5.1"
            ) from exc

    def analyze(self, frame_bgr: np.ndarray) -> List[FaceResult]:
        """Return all faces found in a BGR frame, largest face first."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        height, width = frame_bgr.shape[:2]
        scale = 1.0
        small = frame_bgr
        if width > self.max_width:
            scale = self.max_width / float(width)
            small = cv2.resize(
                frame_bgr, (self.max_width, int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        try:
            raw_results = self._fer.detect_emotions(small)
        except Exception as exc:
            raise EmotionDetectorError(
                f"Emotion prediction failed ({type(exc).__name__}: {exc})"
            ) from exc

        faces: List[FaceResult] = []
        for item in raw_results:
            x, y, w, h = item["box"]
            # Scale the box back to the original frame and clamp to its bounds.
            x = max(0, int(round(x / scale)))
            y = max(0, int(round(y / scale)))
            w = int(round(w / scale))
            h = int(round(h / scale))
            w = min(w, width - x)
            h = min(h, height - y)
            if w <= 0 or h <= 0:
                continue

            scores = {name: 0.0 for name in EMOTIONS}
            for fer_name, value in item["emotions"].items():
                display = _FER_TO_DISPLAY.get(fer_name)
                if display:
                    scores[display] = float(value)

            # Normalise so the seven scores always sum to 1.
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

            faces.append(FaceResult(box=(x, y, w, h), scores=scores))

        faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return faces
