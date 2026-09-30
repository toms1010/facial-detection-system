"""Expression classification stage: crop, preprocess, classify, threshold.

The stage owns everything between a detection box and a
:class:`~visionai.ai.models.base.ExpressionResult`, including the policy that
decides what to do with a low-confidence or heuristic prediction.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from visionai.ai import taxonomy
from visionai.ai.errors import FrameError, ModelError
from visionai.ai.models.base import (
    Box,
    ExpressionClassifier,
    ExpressionResult,
    FaceDetection,
    validate_frame,
)
from visionai.ai.models.registry import LoadReport

LOG = logging.getLogger(__name__)

UNKNOWN_LABEL = "unknown"


@dataclass
class ClassificationStage:
    """Crops each face and runs the expression classifier."""

    classifier: ExpressionClassifier
    report: LoadReport | None = None
    confidence_threshold: float = 0.0
    pad_ratio: float = 0.12
    max_faces: int = 16
    last_latency_ms: float = 0.0
    last_count: int = 0
    error_count: int = 0
    last_error: str = ""
    suppressed: int = 0
    _batchable: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._batchable = bool(getattr(self.classifier, "supports_batching", False))

    def run(
        self, frame: np.ndarray, detections: Sequence[FaceDetection]
    ) -> dict[int, ExpressionResult]:
        """Return ``{detection index: ExpressionResult}``."""
        results: dict[int, ExpressionResult] = {}
        if not detections:
            self.last_count = 0
            # No crops means no classification work, so the reported cost is
            # zero rather than whatever the last populated frame happened to
            # take. Leaving it stale would keep inflating the average after the
            # faces have left.
            self.last_latency_ms = 0.0
            return results

        image = validate_frame(frame, "frame")
        height, width = image.shape[:2]
        start = time.perf_counter()

        usable: list[tuple[int, np.ndarray]] = []
        for index, detection in enumerate(detections[: self.max_faces]):
            crop = self.crop_face(image, detection.box, width, height)
            if crop is None:
                results[index] = self._unknown(detection, "crop unavailable")
                continue
            detection.crop = crop
            usable.append((index, crop))

        if usable:
            try:
                predictions = self._classify([crop for _, crop in usable])
            except ModelError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                LOG.warning("expression classifier failed: %s", exc)
                return results

            for (index, _), prediction in zip(usable, predictions, strict=False):
                if prediction is None:
                    results[index] = self._unknown(detections[index], "no prediction")
                    continue
                prediction.box = detections[index].box
                results[index] = prediction

        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        self.last_count = len(results)
        return results

    def _classify(
        self, crops: Sequence[np.ndarray]
    ) -> list[ExpressionResult | None]:
        if not crops:
            return []
        if self._batchable and len(crops) > 1:
            try:
                return list(self.classifier.predict_batch(list(crops)))
            except Exception as exc:  # noqa: BLE001
                LOG.debug("batched classification unavailable, falling back: %s", exc)
        output: list[ExpressionResult | None] = []
        for crop in crops:
            try:
                output.append(self.classifier.predict(crop))
            except FrameError:
                output.append(None)
            except Exception as exc:  # noqa: BLE001
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                LOG.warning("per-crop classification failed: %s", exc)
                output.append(None)
        return output

    def crop_face(
        self, frame: np.ndarray, box: Box, width: int, height: int
    ) -> np.ndarray | None:
        """Return a padded BGR crop, or ``None`` when the box is degenerate."""
        padded = box.expanded(self.pad_ratio, width, height).clamp(width, height)
        x1, y1, x2, y2 = padded.as_int()
        x2, y2 = max(x1 + 8, min(x2, width)), max(y1 + 8, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(crop)

    def _unknown(self, detection: FaceDetection, reason: str) -> ExpressionResult:
        names = list(getattr(self.classifier, "class_names", ()) or taxonomy.DEFAULT_CLASS_NAMES)
        scores = {name: (1.0 if name == "neutral" else 0.0) for name in names}
        return ExpressionResult(
            label=UNKNOWN_LABEL,
            confidence=0.0,
            scores=scores,
            model=self.classifier.name,
            is_heuristic=bool(getattr(self.classifier, "is_heuristic", False)),
            is_confident=False,
            box=detection.box,
        )

    def passes_threshold(self, result: ExpressionResult) -> bool:
        """Whether a prediction is confident enough to show as a label.

        A trained model must clear the configured threshold. A heuristic result
        never clears it, because it is not a model prediction at all.
        """
        if result is None or result.label == UNKNOWN_LABEL:
            return False
        if getattr(self.classifier, "is_heuristic", False):
            return False
        return result.confidence >= self.confidence_threshold

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.classifier.name,
            "report": self.report.describe() if self.report else None,
            "faces": self.last_count,
            "latency_ms": self.last_latency_ms,
            "errors": self.error_count,
            "last_error": self.last_error or None,
            "threshold": self.confidence_threshold,
            "trained_model": bool(getattr(self.classifier, "is_trained_model", True)),
            **self.classifier.describe(),
        }

    def close(self) -> None:
        self.classifier.close()
