"""Model backends for face detection and expression classification.

Import concrete classes from their modules; this package only re-exports the
stable names that the rest of the application depends on.
"""

from __future__ import annotations

from visionai.ai.models.base import (
    Box,
    ExpressionClassifier,
    ExpressionResult,
    FaceDetection,
    FaceDetector,
    validate_frame,
)
from visionai.ai.models.haar_detector import HaarFaceDetector
from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier
from visionai.ai.models.registry import LoadReport, ModelRegistry

__all__ = [
    "Box",
    "ExpressionClassifier",
    "ExpressionResult",
    "FaceDetection",
    "FaceDetector",
    "HaarFaceDetector",
    "HeuristicExpressionClassifier",
    "LoadReport",
    "ModelRegistry",
    "validate_frame",
]
