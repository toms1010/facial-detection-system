"""Core data structures and abstract interfaces for the AI layer.

Adding a new model family means implementing :class:`FaceDetector` or
:class:`ExpressionClassifier` and registering it; nothing else in the pipeline
needs to change.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visionai.ai import taxonomy
from visionai.ai.errors import FrameError, InferenceError, ModelLoadError


@dataclass
class Box:
    """An axis-aligned box in pixel coordinates of the source frame."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        self.x1 = float(self.x1)
        self.y1 = float(self.y1)
        self.x2 = float(self.x2)
        self.y2 = float(self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5

    def as_int(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )

    def clamp(self, width: int, height: int) -> Box:
        return Box(
            max(0.0, min(self.x1, float(width))),
            max(0.0, min(self.y1, float(height))),
            max(0.0, min(self.x2, float(width))),
            max(0.0, min(self.y2, float(height))),
        )

    def iou(self, other: Box) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def expanded(self, ratio: float, width: int, height: int) -> Box:
        """Grow the box by ``ratio`` of its size on every side, clamped to the frame."""
        dx = self.width * ratio
        dy = self.height * ratio
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy).clamp(width, height)

    def scaled(self, sx: float, sy: float) -> Box:
        return Box(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def copy(self) -> Box:
        return Box(self.x1, self.y1, self.x2, self.y2)

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> Box:
        return Box(x, y, x + w, y + h)

    @staticmethod
    def from_any(value: Any) -> Box:
        if isinstance(value, Box):
            return value
        if isinstance(value, Mapping):
            return Box(
                float(value.get("x1", 0.0)),
                float(value.get("y1", 0.0)),
                float(value.get("x2", 0.0)),
                float(value.get("y2", 0.0)),
            )
        seq = list(value)
        if len(seq) == 4:
            return Box(*(float(v) for v in seq))
        if len(seq) == 8:
            return Box(*(float(v) for v in seq[:4]))
        raise ValueError(f"cannot interpret {value!r} as a Box")


@dataclass
class FaceDetection:
    """One detected face before tracking assigns it an identity."""

    box: Box
    score: float
    source_class: str = "face"
    crop: np.ndarray | None = field(default=None, repr=False, compare=False)
    track_id: int | None = None
    detection_time: float = field(default_factory=time.perf_counter)
    landmarks: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.box = Box.from_any(self.box)
        self.score = float(max(0.0, min(1.0, self.score)))

    @property
    def area_ratio(self) -> float:
        return self.box.area


@dataclass
class ExpressionResult:
    """A single expression prediction with its full score distribution."""

    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    model: str = "unknown"
    is_heuristic: bool = False
    is_confident: bool = True
    latency_ms: float = 0.0
    box: Box | None = None
    track_id: int | None = None
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.label = taxonomy.normalize_label(self.label) or str(self.label)
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        if self.scores:
            self.scores = {
                taxonomy.normalize_label(k) or str(k): float(v)
                for k, v in self.scores.items()
            }

    @property
    def emoji(self) -> str:
        return taxonomy.emoji_for(self.label)

    @property
    def display_label(self) -> str:
        return taxonomy.title(self.label)

    @property
    def color(self) -> tuple[int, int, int]:
        return taxonomy.color_for(self.label)

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def describe(self) -> str:
        return taxonomy.describe(self.label, self.confidence)

    @classmethod
    def from_scores(
        cls,
        scores: Mapping[Any, float],
        model: str,
        is_heuristic: bool = False,
        threshold: float = 0.0,
    ) -> ExpressionResult:
        normalized = taxonomy.align_scores(scores, scores.keys())
        if not normalized:
            raise InferenceError("classifier returned an empty score distribution")
        label, confidence = max(normalized.items(), key=lambda kv: kv[1])
        result = cls(
            label=label,
            confidence=confidence,
            scores=normalized,
            model=model,
            is_heuristic=is_heuristic,
        )
        result.is_confident = (not is_heuristic) and confidence >= threshold
        return result


class _LatencyMixin:
    """Shared timing helper for model implementations."""

    _last_latency_ms: float = 0.0

    @property
    def last_latency_ms(self) -> float:
        return float(self._last_latency_ms)

    def _timed(self):
        return _LatencyTimer(self)


class _LatencyTimer:
    def __init__(self, owner: _LatencyMixin) -> None:
        self._owner = owner
        self._start = 0.0

    def __enter__(self) -> _LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._owner._last_latency_ms = (time.perf_counter() - self._start) * 1000.0


def validate_frame(frame: np.ndarray | None, name: str = "frame") -> np.ndarray:
    """Reject anything that is not a usable BGR/gray image."""
    if frame is None:
        raise FrameError(f"{name} is None")
    if not isinstance(frame, np.ndarray):
        raise FrameError(f"{name} must be a numpy array, got {type(frame).__name__}")
    if frame.size == 0:
        raise FrameError(f"{name} is empty")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] in (1, 3, 4):
        return frame
    raise FrameError(f"{name} has unsupported shape {frame.shape}")


class FaceDetector(abc.ABC):
    """Locates faces in a frame.

    Implementations must not mutate the frame they are given and must be safe
    to call from a single dedicated worker thread.
    """

    name: str = "detector"
    display_name: str = "Face detector"
    requires_weights: bool = False
    supports_batching: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """Prepare the model. Must be idempotent."""

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when :meth:`detect` may be called."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Return face detections for ``frame`` (BGR or grayscale)."""

    def close(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release native resources. Must be idempotent."""

    @property
    def last_latency_ms(self) -> float:
        return 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "ready": self.is_ready,
            "requires_weights": self.requires_weights,
            "latency_ms": self.last_latency_ms,
        }

    def __enter__(self) -> FaceDetector:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ExpressionClassifier(abc.ABC):
    """Maps a face crop to an expression label plus a score distribution."""

    name: str = "classifier"
    display_name: str = "Expression classifier"
    is_heuristic: bool = False
    is_trained_model: bool = True
    requires_weights: bool = False
    confidence_ceiling: float = 1.0
    input_size: tuple[int, int] = (224, 224)
    class_names: Sequence[str] = taxonomy.DEFAULT_CLASS_NAMES
    supports_batching: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """Prepare the model. Must be idempotent."""

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when :meth:`predict` may be called."""

    @abc.abstractmethod
    def predict(self, crop: np.ndarray) -> ExpressionResult:
        """Classify a single BGR face crop."""

    def predict_batch(self, crops: Sequence[np.ndarray]) -> list[ExpressionResult]:
        """Classify several crops; the default loops over :meth:`predict`."""
        return [self.predict(crop) for crop in crops]

    def close(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release native resources. Must be idempotent."""

    @property
    def last_latency_ms(self) -> float:
        return 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "ready": self.is_ready,
            "is_heuristic": self.is_heuristic,
            "is_trained_model": self.is_trained_model,
            "confidence_ceiling": self.confidence_ceiling,
            "class_names": list(self.class_names),
            "input_size": list(self.input_size),
            "latency_ms": self.last_latency_ms,
        }

    def __enter__(self) -> ExpressionClassifier:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def enforce_confidence_ceiling(
    result: ExpressionResult, classifier: ExpressionClassifier
) -> ExpressionResult:
    """Clamp a heuristic prediction so it can never look as certain as a model."""
    ceiling = float(getattr(classifier, "confidence_ceiling", 1.0))
    if ceiling >= 1.0 or result.confidence <= ceiling:
        return result
    factor = ceiling / max(result.confidence, 1e-6)
    result.confidence = ceiling
    result.scores = {k: v * factor for k, v in result.scores.items()}
    return result


__all__ = [
    "Box",
    "ExpressionClassifier",
    "ExpressionResult",
    "FaceDetection",
    "FaceDetector",
    "ModelLoadError",
    "enforce_confidence_ceiling",
    "validate_frame",
]
