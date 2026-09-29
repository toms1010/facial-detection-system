"""Qt application bootstrap and window construction."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from visionai.ai.errors import VisionAIError
from visionai.config.settings import Settings
from visionai.hardware.hardware_bridge import HardwareBridge
from visionai.pipeline.engine import PipelineEngine
from visionai.ui.main_window import MainWindow
from visionai.ui.theme import get_palette, stylesheet

LOG = logging.getLogger(__name__)


def qt_available() -> bool:
    try:
        import PySide6  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def create_window(
    settings: Settings,
    source_spec: str = "",
    engine: PipelineEngine | None = None,
    bridge: HardwareBridge | None = None,
) -> MainWindow:
    """Build the main window, loading models eagerly so the first frame is fast."""
    engine = engine or PipelineEngine(settings)
    bridge = bridge or HardwareBridge(
        settings.hardware, anonymize=settings.privacy.anonymize_log_payloads
    )
    if settings.hardware.enabled:
        try:
            bridge.start()
        except VisionAIError as exc:
            LOG.warning("hardware monitoring unavailable: %s", exc)
    return MainWindow(settings, engine, bridge, source_spec=source_spec)


def run(
    settings: Settings,
    source_spec: str = "",
    start_immediately: bool | None = None,
    app: Any = None,
) -> int:
    """Run the desktop application. Returns the Qt exit code."""
    from PySide6.QtWidgets import QApplication

    owns_app = app is None
    application = app or QApplication(sys.argv)
    application.setApplicationName("Linux AI Vision")
    application.setApplicationVersion(_version())
    application.setStyleSheet(stylesheet(get_palette(settings.ui.theme)))

    try:
        window = create_window(settings, source_spec=source_spec)
    except VisionAIError as exc:
        LOG.error("could not start the application: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if start_immediately is None:
        start_immediately = settings.ui.start_with_camera
    if start_immediately:
        window.toggle_pipeline()

    window.show()
    if not owns_app:
        return 0
    return int(application.exec())


def _version() -> str:
    from visionai import __version__

    return __version__
