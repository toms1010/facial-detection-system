"""Camera capture layer."""

from __future__ import annotations

from visionai.camera.camera_manager import (
    CameraDevice,
    CameraManager,
    CaptureStats,
    enumerate_cameras,
    open_source,
)
from visionai.camera.frame_source import (
    FrameSource,
    SyntheticFrameSource,
    VideoFileSource,
)

__all__ = [
    "CameraDevice",
    "CameraManager",
    "CaptureStats",
    "FrameSource",
    "SyntheticFrameSource",
    "VideoFileSource",
    "enumerate_cameras",
    "open_source",
]
