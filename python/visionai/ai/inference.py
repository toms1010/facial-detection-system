"""Synchronous inference pipeline: detect → classify → track.

This module contains no threading. The async engine in
:mod:`visionai.pipeline.engine` owns the worker thread and calls
:meth:`InferencePipeline.process` once per frame, which keeps the core logic
directly unit-testable without a running Qt event loop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from visionai.ai.emotion_classifier import ClassificationStage
from visionai.ai.errors import ModelError
from visionai.ai.face_tracker import FaceTracker, TrackedFace
from visionai.ai.models.base import ExpressionResult, FaceDetection
from visionai.ai.models.registry import LoadReport, ModelRegistry
from visionai.ai.yolo_detector import DetectionStage
from visionai.config.settings import Settings
from visionai.pipeline.stats import PipelineStats

LOG = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything one inference pass produced."""

    frame: np.ndarray | None
    frame_number: int
    timestamp: float
    detections: list[FaceDetection] = field(default_factory=list)
    tracks: list[TrackedFace] = field(default_factory=list)
    expressions: dict[int, ExpressionResult] = field(default_factory=dict)
    detection_ms: float = 0.0
    classification_ms: float = 0.0
    total_ms: float = 0.0
    error: str = ""

    @property
    def face_count(self) -> int:
        return len(self.tracks) if self.tracks else len(self.detections)

    def top_expression(self) -> ExpressionResult | None:
        ranked = [t.expression for t in self.tracks if t.expression is not None]
        if not ranked:
            return None
        return max(ranked, key=lambda e: e.confidence)

    def summarize(self) -> str:
        from visionai.ai.taxonomy import describe_track

        if not self.tracks:
            return "No faces detected"
        return " | ".join(
            describe_track(t.track_id, t.label, t.confidence) for t in self.tracks
        )

    def summarise(self) -> str:
        """British spelling alias of :meth:`summarize`."""
        return self.summarize()

    def describe(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "faces": self.face_count,
            "detection_ms": self.detection_ms,
            "classification_ms": self.classification_ms,
            "total_ms": self.total_ms,
            "error": self.error or None,
            "tracks": [
                {
                    "id": t.track_id,
                    "expression": t.label,
                    "confidence": t.confidence,
                    "box": t.box.as_int(),
                    "hits": t.hits,
                }
                for t in self.tracks
            ],
        }


class InferencePipeline:
    """Owns the detector, classifier, tracker and per-frame statistics."""

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        detector: Any = None,
        classifier: Any = None,
        detector_report: LoadReport | None = None,
        classifier_report: LoadReport | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.detector_stage = DetectionStage(
            detector=detector,
            report=detector_report,
            max_faces=settings.models.max_faces,
            frame_scale=self._frame_scale(settings),
        )
        self.classifier_stage = ClassificationStage(
            classifier=classifier,
            report=classifier_report,
            confidence_threshold=settings.models.emotion_confidence,
            pad_ratio=settings.models.pad_ratio,
            max_faces=settings.models.max_faces,
        )
        self.tracker = FaceTracker(
            iou_threshold=settings.pipeline.track_iou_threshold,
            max_age=settings.pipeline.track_max_age,
            smooth=settings.pipeline.track_smooth,
            smooth_alpha=settings.pipeline.smooth_alpha,
            hold_labels=settings.pipeline.label_hold_frames,
        )
        self.stats = PipelineStats(inference_fps=settings.pipeline.inference_fps)
        self._frame_number = 0
        self._closed = False

    @staticmethod
    def _frame_scale(settings: Settings) -> float:
        target = settings.pipeline.processing_width
        return 1.0 if target <= 0 else float(target) / 1280.0

    @property
    def uses_trained_expression_model(self) -> bool:
        return bool(getattr(self.classifier_stage.classifier, "is_trained_model", True))

    @property
    def expression_quality_note(self) -> str:
        classifier = self.classifier_stage.classifier
        if getattr(classifier, "is_heuristic", False):
            return (
                "Heuristic fallback: no trained expression model is installed, so "
                "labels come from geometric cues only and are shown as low-confidence."
            )
        return f"Trained model: {getattr(classifier, 'name', 'unknown')}"

    def process(self, frame: np.ndarray) -> PipelineResult:
        """Run one full inference pass over ``frame``."""
        started = time.perf_counter()
        self._frame_number += 1
        result = PipelineResult(
            frame=frame,
            frame_number=self._frame_number,
            timestamp=time.time(),
        )

        try:
            detections = self.detector_stage.run(frame)
            result.detections = detections
            expressions = self.classifier_stage.run(frame, detections)
            result.expressions = expressions
        except ModelError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            LOG.warning("pipeline model error: %s", exc)
            self.stats.record_error(str(exc))
            return result

        result.tracks = self.tracker.update(detections, expressions)
        result.detection_ms = self.detector_stage.last_latency_ms
        result.classification_ms = self.classifier_stage.last_latency_ms
        result.total_ms = (time.perf_counter() - started) * 1000.0

        self.stats.record(
            total_ms=result.total_ms,
            detection_ms=result.detection_ms,
            classification_ms=result.classification_ms,
            face_count=result.face_count,
        )
        return result

    def reset(self) -> None:
        self.tracker.reset()
        self.stats.reset()
        self._frame_number = 0

    def describe(self) -> dict[str, Any]:
        return {
            "detector": self.detector_stage.describe(),
            "classifier": self.classifier_stage.describe(),
            "tracker": self.tracker.describe(),
            "stats": self.stats.snapshot(),
            "quality": self.expression_quality_note,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stage in (self.detector_stage, self.classifier_stage):
            try:
                stage.close()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("error closing %s: %s", type(stage).__name__, exc)

    def __enter__(self) -> InferencePipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def render_preview(frame: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Return a display-ready copy of ``frame`` without mutating the original.

    Overlays are drawn separately by :mod:`visionai.ui.overlay`; this only
    normalises the frame to BGR and applies the optional mirror.
    """
    view = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if mirror:
        view = cv2.flip(view, 1)
    return view


def clamp_detections(
    detections: Sequence[FaceDetection], limit: int
) -> list[FaceDetection]:
    return list(detections)[: max(0, limit)]
