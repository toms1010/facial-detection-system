"""AI layer: detection, expression classification, tracking and the pipeline.

YOLO is used for detection. Expression classification is a separate model and
is labelled an *AI estimate* everywhere it is surfaced.
"""

from __future__ import annotations

from visionai.ai.errors import (
    CameraError,
    ConfigError,
    DeviceError,
    FrameError,
    HardwareError,
    InferenceError,
    ModelError,
    ModelLoadError,
    ModelMissingError,
    NativeLibraryError,
    SensorUnavailableError,
    TrackingError,
    VisionAIError,
)
from visionai.ai.taxonomy import (
    DISCLAIMER,
    ESTIMATE_PREFIX,
    EmotionClass,
    describe,
    describe_track,
)

__all__ = [
    "CameraError",
    "ConfigError",
    "DISCLAIMER",
    "ESTIMATE_PREFIX",
    "DeviceError",
    "EmotionClass",
    "FrameError",
    "HardwareError",
    "InferenceError",
    "ModelError",
    "ModelLoadError",
    "ModelMissingError",
    "NativeLibraryError",
    "SensorUnavailableError",
    "TrackingError",
    "VisionAIError",
    "describe",
    "describe_track",
]
