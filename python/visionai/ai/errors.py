"""Exception hierarchy shared by the AI layer.

Every recoverable failure the application anticipates has a type here, so the
UI and the CLI can present a specific message instead of a traceback.
"""

from __future__ import annotations


class VisionAIError(Exception):
    """Base class for every error raised by this project."""


class ConfigError(VisionAIError):
    """Settings could not be loaded or are internally inconsistent."""


class ModelError(VisionAIError):
    """Base class for model-related failures."""


class ModelMissingError(ModelError):
    """A required weights file or descriptor is not present on disk."""


class ModelLoadError(ModelError):
    """Weights exist but could not be loaded or have an unusable format."""


class InferenceError(ModelError):
    """A model ran but produced an unusable result."""


class DeviceError(ModelError):
    """A compute device was requested but is not available."""


class CameraError(VisionAIError):
    """Base class for capture-device failures."""


class CameraUnavailableError(CameraError):
    """No capture device could be opened."""


class CameraPermissionError(CameraError):
    """The OS refused access to the capture device."""


class InvalidCameraError(CameraError):
    """The requested device index or path does not exist."""


class FrameError(CameraError):
    """A frame was empty, malformed or the wrong shape."""


class HardwareError(VisionAIError):
    """Base class for native hardware monitoring failures."""


class NativeLibraryError(HardwareError):
    """The compiled C++ layer is missing or could not be loaded."""


class SensorUnavailableError(HardwareError):
    """A specific sensor (GPU, temperature, NIC) is not present."""


class TrackingError(VisionAIError):
    """Face tracking could not associate the current detections."""
