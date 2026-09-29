"""Offline geometric expression estimator used when no trained model is present.

This is **not a machine-learning model**. It measures a handful of cheap
geometric cues from a face crop (mouth curvature, eye aperture, brow energy,
lower-face compression) and maps them onto the seven primary classes with a
hand-written scoring function.

Because it is untrained, the results are weak and are marked as such
everywhere: :attr:`is_heuristic` is true, :attr:`is_trained_model` is false, and
:attr:`confidence_ceiling` caps reported confidence so the UI can never present
it as a real prediction. It exists so the pipeline, the tracker, the overlay and
the tests all have a deterministic offline backend.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

import cv2
import numpy as np

from visionai.ai import taxonomy
from visionai.ai.errors import FrameError
from visionai.ai.models.base import (
    ExpressionClassifier,
    ExpressionResult,
    validate_frame,
)

LOG = logging.getLogger(__name__)

#: Untrained heuristics must never look authoritative next to a real model.
CONFIDENCE_CEILING = 0.55

FEATURE_NAMES = (
    "mouth_curve",
    "mouth_open",
    "eye_open",
    "brow_raise",
    "brow_furrow",
    "lower_face_tension",
    "symmetry",
    "brightness",
)


class HeuristicExpressionClassifier(ExpressionClassifier):
    """Deterministic feature-based expression estimator (no training involved)."""

    name = "heuristic"
    display_name = "Geometric heuristic (untrained)"
    is_heuristic = True
    is_trained_model = False
    requires_weights = False
    confidence_ceiling = CONFIDENCE_CEILING
    input_size = (64, 64)
    class_names = taxonomy.DEFAULT_CLASS_NAMES

    def __init__(self, smooth: float = 0.35) -> None:
        self._ready = False
        self._last_latency_ms = 0.0
        self._smooth = float(max(0.0, min(0.95, smooth)))
        self._previous: dict[str, float] | None = None

    def load(self) -> None:
        self._ready = True
        self._previous = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def close(self) -> None:
        self._ready = False
        self._previous = None

    def predict(self, crop: np.ndarray) -> ExpressionResult:
        start = time.perf_counter()
        image = validate_frame(crop, "face crop")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[0] < 8 or image.shape[1] < 8:
            raise FrameError(f"face crop too small: {image.shape[:2]}")

        features = self.extract_features(image)
        if self._previous is not None:
            features = {
                key: (1.0 - self._smooth) * self._previous.get(key, value)
                + self._smooth * value
                for key, value in features.items()
            }
        self._previous = dict(features)

        scores = self.score(features)
        result = ExpressionResult.from_scores(
            scores, model=self.name, is_heuristic=True, threshold=0.0
        )
        result.confidence = min(result.confidence, self.confidence_ceiling)
        peak = max(result.scores.values()) or 1.0
        result.scores = {k: (v / peak) * result.confidence for k, v in result.scores.items()}
        result.is_confident = False
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        result.alternatives = tuple(k for k, _ in result.top_k(3)[1:])
        return result

    def extract_features(self, image: np.ndarray) -> dict[str, float]:
        """Measure the geometric cues. All outputs are normalised to roughly 0..1."""
        size = self.input_size[0]
        small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        edges_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(edges_x, edges_y)

        def band(y0: float, y1: float) -> slice:
            a, b = int(y0 * size), max(int(y0 * size) + 1, int(y1 * size))
            return slice(min(a, size - 1), min(b, size))

        eye_band = band(0.22, 0.48)
        brow_band = band(0.12, 0.28)
        mouth_band = band(0.58, 0.86)
        lower_band = band(0.62, 0.95)
        left_half = slice(4, size // 2 - 2)
        right_half = slice(size // 2 + 2, size - 4)

        eye_energy = float(edge_mag[eye_band].mean())
        brow_energy = float(edge_mag[brow_band].mean())
        mouth_rows = edge_mag[mouth_band]
        lower_energy = float(edge_mag[lower_band].mean())

        centre_line = size // 2
        half = max(1, size // 8)
        brow_rows = edge_mag[brow_band]
        # A furrowed brow concentrates energy in a narrow strip between the eyes;
        # a raised brow spreads it across the full band. Measuring both gives
        # "angry" and "surprise" genuinely different evidence instead of one
        # signal counted twice.
        central_brow = brow_rows[:, centre_line - half : centre_line + half]
        total_brow = float(brow_rows.sum())
        brow_concentration = (
            float(central_brow.sum()) / total_brow if total_brow > 1e-6 else 0.0
        )
        mouth_slice = slice(int(0.58 * size), int(0.86 * size))
        window = mouth_rows[:, centre_line - half : centre_line + half]
        if window.size:
            curve = float(window[0].mean() - window[-1].mean())
            openness = float(
                np.percentile(window, 95) - np.percentile(window, 5)
            )
        else:
            curve, openness = 0.0, 0.0

        mean_gray = float(gray.mean())
        half_diff = abs(
            float(edge_mag[mouth_slice, left_half].mean() - edge_mag[mouth_slice, right_half].mean())
        )

        return {
            "mouth_curve": _norm(curve, 0.06),
            "mouth_open": _norm(openness, 0.14),
            "eye_open": _norm(eye_energy, 0.10),
            "brow_raise": _norm(brow_energy, 0.11),
            "brow_furrow": _norm(brow_concentration, 0.45),
            "lower_face_tension": _norm(lower_energy, 0.11),
            "symmetry": 1.0 - _norm(half_diff, 0.10),
            "brightness": mean_gray,
        }

    def score(self, features: Mapping[str, float]) -> dict[str, float]:
        """Map measured cues onto the seven primary classes.

        The weights encode the textbook cues for each expression; they are
        intuitive rather than learned, which is exactly why confidence is capped.
        """
        curve = features["mouth_curve"]
        mouth_open = features["mouth_open"]
        eye_open = features["eye_open"]
        brow = features["brow_raise"]
        furrow = features["brow_furrow"]
        tension = features["lower_face_tension"]
        brightness = features["brightness"]

        smile = max(0.0, curve) * (0.6 + 0.4 * (1.0 - mouth_open))
        frown = max(0.0, -curve) * (0.6 + 0.4 * (1.0 - mouth_open))
        neutral_cue = (1.0 - abs(curve)) * (1.0 - mouth_open) * (1.0 - brow) * (1.0 - tension)

        return {
            "happy": 2.4 * smile + 0.5 * max(0.0, 1.0 - brow),
            "sad": 2.0 * frown + 0.4 * (1.0 - eye_open) + 0.3 * furrow,
            "angry": 1.8 * furrow * tension + 1.0 * frown * tension,
            "fear": 1.8 * brow * eye_open + 0.9 * mouth_open * brow,
            "surprise": 2.0 * brow * mouth_open * (0.5 + 0.5 * eye_open),
            "disgust": 1.7 * tension * (0.5 + 0.5 * max(0.0, -curve)),
            "neutral": 2.2 * neutral_cue + 0.3 * brightness,
        }

    def describe(self) -> dict:
        info = super().describe()
        info["warning"] = (
            "UNTRAINED heuristic: no trained expression model is installed, so results "
            "come from a hand-written geometric rule set and are not a real prediction."
        )
        return info


def _norm(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(max(0.0, min(1.0, value / scale)))
