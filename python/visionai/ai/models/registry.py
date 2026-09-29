"""Model registry: turns settings into concrete, loaded model instances.

The registry is the single place that knows which backends exist. It also owns
the fallback policy: a requested backend that cannot load is downgraded with a
recorded reason rather than crashing the application.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from visionai.ai import taxonomy
from visionai.ai.errors import ModelError
from visionai.ai.models.base import ExpressionClassifier, FaceDetector
from visionai.ai.models.descriptor import discover_models
from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier
from visionai.ai.models.yunet_detector import YuNetFaceDetector

if TYPE_CHECKING:
    from visionai.config.settings import Settings

LOG = logging.getLogger(__name__)

#: Preferred first. YuNet ships as a small ONNX file in models/face, so the
#: application is useful on a fresh checkout with no downloads. Haar is kept for
#: OpenCV builds that still expose CascadeClassifier (removed in OpenCV 5.0).
DETECTOR_ORDER = ("yunet", "haar", "yolo", "onnx")
CLASSIFIER_ORDER = ("heuristic", "onnx", "torchscript")


def _haar_supported() -> bool:
    """Haar needs CascadeClassifier, which OpenCV 5.0 no longer ships."""
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return False
    return hasattr(cv2, "CascadeClassifier")


@dataclass
class LoadReport:
    """What was requested, what was actually built, and why."""

    requested: str
    active: str
    degraded: bool
    reason: str = ""

    def describe(self) -> str:
        if self.degraded:
            return f"{self.requested} unavailable, using {self.active} ({self.reason})"
        return f"{self.active} active"


class ModelRegistry:
    """Builds detectors and classifiers, resolving availability lazily."""

    def __init__(self, face_dir: Path, emotion_dir: Path) -> None:
        self.face_dir = Path(face_dir)
        self.emotion_dir = Path(emotion_dir)
        self._detectors: dict[str, Callable[[Settings], FaceDetector]] = {
            "yunet": self._build_yunet,
            "haar": self._build_haar,
            "yolo": self._build_yolo,
            "onnx": self._build_onnx,
        }
        self._classifiers: dict[str, Callable[[Settings], ExpressionClassifier]] = {
            "heuristic": self._build_heuristic,
            "onnx": self._build_onnx_classifier,
            "torchscript": self._build_torch_classifier,
        }
        if not _haar_supported():
            del self._detectors["haar"]
            LOG.debug("OpenCV build has no CascadeClassifier; the haar detector is hidden")

    def available_detectors(self) -> tuple[str, ...]:
        return tuple(name for name in DETECTOR_ORDER if name in self._detectors)

    def available_classifiers(self) -> tuple[str, ...]:
        return tuple(name for name in CLASSIFIER_ORDER if name in self._classifiers)

    def available_emotion_models(self) -> list[Path]:
        return discover_models(self.emotion_dir)

    def available_face_models(self) -> list[Path]:
        return discover_models(self.face_dir)

    def _build_yunet(self, settings) -> FaceDetector:
        weights = settings.models.detector_weights
        if not (weights and weights.endswith(".onnx")):
            weights = None
        return YuNetFaceDetector(
            confidence=max(settings.models.detection_confidence, 0.3),
            weights=weights,
        )

    def _build_haar(self, settings) -> FaceDetector:
        from visionai.ai.models.haar_detector import HaarFaceDetector

        return HaarFaceDetector(confidence=settings.models.detection_confidence)

    def _build_yolo(self, settings) -> FaceDetector:
        from visionai.ai.models.yolo_detector import YoloFaceDetector

        return YoloFaceDetector(
            weights=settings.models.detector_weights,
            confidence=settings.models.detection_confidence,
            iou=settings.models.detection_iou,
            imgsz=settings.models.imgsz,
            device=settings.models.device,
            half=settings.models.half_precision,
        )

    def _build_onnx(self, settings) -> FaceDetector:
        from visionai.ai.models.onnx_detector import OnnxYoloFaceDetector

        return OnnxYoloFaceDetector(
            weights=settings.models.detector_weights,
            confidence=settings.models.detection_confidence,
            iou=settings.models.detection_iou,
            imgsz=settings.models.imgsz,
            device=settings.models.device,
        )

    def _build_heuristic(self, settings) -> ExpressionClassifier:
        return HeuristicExpressionClassifier(smooth=settings.pipeline.smooth_alpha)

    def _build_onnx_classifier(self, settings) -> ExpressionClassifier:
        from visionai.ai.models.onnx_classifier import OnnxExpressionClassifier

        return OnnxExpressionClassifier(
            model_path=settings.models.classifier_weights,
            device=settings.models.device,
            confidence_threshold=settings.models.emotion_confidence,
        )

    def _build_torch_classifier(self, settings) -> ExpressionClassifier:
        from visionai.ai.models.torch_classifier import TorchScriptExpressionClassifier

        return TorchScriptExpressionClassifier(
            model_path=settings.models.classifier_weights,
            device=settings.models.device,
            confidence_threshold=settings.models.emotion_confidence,
        )

    def _resolve(
        self,
        kind: str,
        requested: str,
        order: tuple[str, ...],
        builders: dict[str, Callable[[Settings], Any]],
        settings: Settings,
        weights: str | None,
    ) -> tuple[Any, LoadReport]:
        attempts: list[str] = []
        for name in self._candidate_order(requested, order, weights):
            builder = builders.get(name)
            if builder is None:
                continue
            try:
                instance = builder(settings)
                instance.load()
            except ModelError as exc:
                attempts.append(f"{name}: {exc}")
                LOG.warning("%s backend %r unavailable: %s", kind, name, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{name}: unexpected {exc}")
                LOG.exception("%s backend %r raised during load", kind, name)
                continue
            reason = "; ".join(attempts)
            return instance, LoadReport(
                requested=requested,
                active=name,
                degraded=name != requested and bool(attempts),
                reason=reason,
            )
        raise ModelError(
            f"no usable {kind} backend. Tried: " + ("; ".join(attempts) or "none")
        )

    @staticmethod
    def _candidate_order(
        requested: str,
        order: tuple[str, ...],
        weights: str | None,
    ) -> tuple[str, ...]:
        """Order the backends to try, most preferred first.

        The requested backend is always tried, because the user asked for it by
        name. Any *fallback* that needs weights is skipped unless weights have
        actually been configured. Two reasons: importing ultralytics and torch
        costs over a gigabyte of resident memory, and silently switching to a
        different model family would change the results without the user asking
        for it.

        ``weights`` is the path the user configured, or ``None``.
        """
        needs_weights = {"yolo", "onnx", "torchscript"}
        candidates: list[str] = []
        if requested in order:
            candidates.append(requested)
        for name in order:
            if name in candidates:
                continue
            if name in needs_weights and not weights:
                continue
            candidates.append(name)
        return tuple(candidates) or (requested,)

    def build_detector(self, settings: Settings) -> tuple[FaceDetector, LoadReport]:
        return self._resolve(
            "detector",
            settings.models.detector,
            DETECTOR_ORDER,
            self._detectors,
            settings,
            settings.models.detector_weights,
        )

    def build_classifier(self, settings: Settings) -> tuple[ExpressionClassifier, LoadReport]:
        return self._resolve(
            "classifier",
            settings.models.classifier,
            CLASSIFIER_ORDER,
            self._classifiers,
            settings,
            settings.models.classifier_weights,
        )

    def class_choices(self) -> tuple[tuple[str, str], ...]:
        return (
            ("7 primary classes", "happy, sad, angry, fear, surprise, disgust, neutral"),
            ("7 primary + 40 extended", "adds joy, love, calm, relaxed, tired, ..."),
        )

    def active_class_names(self, classifier: ExpressionClassifier) -> tuple[str, ...]:
        """The class names the loaded classifier can actually emit."""
        raw = tuple(getattr(classifier, "class_names", ()) or taxonomy.DEFAULT_CLASS_NAMES)
        return tuple(entry.name for entry in taxonomy.resolve(raw))
