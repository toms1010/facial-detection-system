"""Face detection stage of the pipeline.

This is a thin, logging-friendly façade over whichever detector the registry
built. Keeping the stage separate means the engine never has to know whether it
is talking to a Haar cascade, Ultralytics or ONNX Runtime.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visionai.ai.errors import ModelError
from visionai.ai.models.base import Box, FaceDetection, FaceDetector, validate_frame
from visionai.ai.models.registry import LoadReport

LOG = logging.getLogger(__name__)


@dataclass
class DetectionStage:
    """Runs the detector and reports how long it took."""

    detector: FaceDetector
    report: LoadReport | None = None
    max_faces: int = 16
    frame_scale: float = 1.0
    last_latency_ms: float = 0.0
    last_count: int = 0
    error_count: int = 0
    last_error: str = ""
    _frame_size: tuple[int, int] = field(default=(0, 0), repr=False)

    def run(self, frame: np.ndarray) -> list[FaceDetection]:
        image = validate_frame(frame, "frame")
        height, width = image.shape[:2]
        self._frame_size = (width, height)

        working = image
        if self.frame_scale != 1.0 and max(height, width) * self.frame_scale > 0:
            working = _resize(image, self.frame_scale)

        start = time.perf_counter()
        try:
            detections = self.detector.detect(working)
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            LOG.warning("detector %s failed: %s", self.detector.name, exc)
            return []

        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        scaled = self._rescale(detections, width, height)
        scaled.sort(key=lambda d: d.box.area, reverse=True)
        self.last_count = len(scaled)
        return scaled[: self.max_faces]

    def _rescale(self, detections: Sequence[FaceDetection], width: int, height: int) -> list[FaceDetection]:
        if abs(self.frame_scale - 1.0) < 1e-6:
            return list(detections)
        inverse = 1.0 / self.frame_scale
        out: list[FaceDetection] = []
        for detection in detections:
            detection.box = detection.box.scaled(inverse, inverse).clamp(width, height)
            if detection.box.width >= 8 and detection.box.height >= 8:
                out.append(detection)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.detector.name,
            "report": self.report.describe() if self.report else None,
            "faces": self.last_count,
            "latency_ms": self.last_latency_ms,
            "errors": self.error_count,
            "last_error": self.last_error or None,
            "frame_scale": self.frame_scale,
            **self.detector.describe(),
        }

    def close(self) -> None:
        self.detector.close()


def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    new_w = max(16, int(round(width * scale)))
    new_h = max(16, int(round(height * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def boxes_overlap_ratio(a: Box, b: Box) -> float:
    """Intersection over the smaller box; useful for UI hit-testing."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0
