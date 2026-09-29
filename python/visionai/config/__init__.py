"""Configuration for linux-ai-vision."""

from __future__ import annotations

from visionai.config.paths import (
    PackagePaths,
    default_config_path,
    default_log_dir,
    project_root,
)
from visionai.config.settings import (
    CameraSettings,
    HardwareSettings,
    LoggingSettings,
    ModelSettings,
    PipelineSettings,
    PrivacySettings,
    Settings,
    UISettings,
    load_settings,
    save_settings,
)

__all__ = [
    "CameraSettings",
    "HardwareSettings",
    "LoggingSettings",
    "ModelSettings",
    "PackagePaths",
    "PipelineSettings",
    "PrivacySettings",
    "Settings",
    "UISettings",
    "default_config_path",
    "default_log_dir",
    "load_settings",
    "project_root",
    "save_settings",
]
