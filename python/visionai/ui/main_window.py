"""Main application window.

A single Qt timer drives every panel. The window never runs inference: it asks
the engine for the newest completed result and paints it, so a slow model slows
the label refresh but never freezes the interface.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from visionai import __version__
from visionai.ai import taxonomy
from visionai.ai.errors import VisionAIError
from visionai.ai.inference import PipelineResult
from visionai.config.settings import Settings
from visionai.hardware.hardware_bridge import HardwareBridge
from visionai.pipeline.engine import PipelineEngine, PipelineState
from visionai.ui.ai_panel import AIPanel
from visionai.ui.dashboard import DashboardPanel
from visionai.ui.hardware_panel import HardwarePanel
from visionai.ui.overlay import blank_overlay
from visionai.ui.settings_panel import SettingsPanel
from visionai.ui.theme import Palette, get_palette, stylesheet

LOG = logging.getLogger(__name__)

REFRESH_MS = 66
HARDWARE_REFRESH_MS = 1000
RESTART_REQUIRED_NOTICE = (
    "Settings saved. Camera, model and resolution changes apply the next time the "
    "pipeline starts."
)


class MainWindow(QMainWindow):
    """The desktop shell: dashboard, AI, hardware and settings tabs."""

    def __init__(
        self,
        settings: Settings,
        engine: PipelineEngine,
        bridge: HardwareBridge,
        source_spec: str = "",
    ) -> None:
        super().__init__()
        self.settings = settings
        self.engine = engine
        self.bridge = bridge
        self.source_spec = source_spec
        self._palette: Palette = get_palette(settings.ui.theme)
        self._last_result: PipelineResult | None = None
        self._camera_enabled = True
        self._needs_restart = False

        self.setWindowTitle("Linux AI Vision - facial expression estimation")
        self.resize(1360, 900)

        self._build_ui()
        self._build_menu()
        self._start_timers()
        self._refresh_panels()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.dashboard = DashboardPanel(self._palette)
        self.ai_panel = AIPanel(self._palette)
        self.hardware_panel = HardwarePanel(self.bridge, self._palette)
        self.settings_panel = SettingsPanel(self.settings, self._palette)
        self.tabs.addTab(self.dashboard, "Dashboard")
        self.tabs.addTab(self.ai_panel, "AI")
        self.tabs.addTab(self.hardware_panel, "Hardware")
        self.tabs.addTab(self.settings_panel, "Settings")
        layout.addWidget(self.tabs, stretch=1)

        self._connect_settings_panel()
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.status_left = QLabel("Ready")
        self.status_right = QLabel("")
        self.statusBar().addWidget(self.status_left, stretch=1)
        self.statusBar().addPermanentWidget(self.status_right)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("Card")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        title = QLabel("Linux AI Vision")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title)

        self.camera_indicator = QLabel("Camera: off")
        self.camera_indicator.setObjectName("MetricMuted")
        layout.addWidget(self.camera_indicator)

        self.notice = QLabel("")
        self.notice.setObjectName("MetricMuted")
        self.notice.setWordWrap(True)
        self.notice.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.notice, stretch=1)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.toggle_pipeline)
        layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        layout.addWidget(self.pause_button)

        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart_pipeline)
        self.restart_button.setEnabled(False)
        layout.addWidget(self.restart_button)
        return header

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        start = QAction("Start / Stop", self)
        start.setShortcut(QKeySequence("Ctrl+R"))
        start.triggered.connect(self.toggle_pipeline)
        file_menu.addAction(start)
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("&View")
        for index, name in enumerate(("Dashboard", "AI", "Hardware", "Settings")):
            action = QAction(name, self)
            action.setShortcut(QKeySequence(f"Ctrl+{index + 1}"))
            action.triggered.connect(lambda _checked=False, i=index: self.tabs.setCurrentIndex(i))
            view_menu.addAction(action)

        help_menu = menu.addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)
        limitations = QAction("AI limitations", self)
        limitations.triggered.connect(self.show_limitations)
        help_menu.addAction(limitations)
        privacy = QAction("Privacy notice", self)
        privacy.triggered.connect(self.show_privacy)
        help_menu.addAction(privacy)

    def _connect_settings_panel(self) -> None:
        self.settings_panel.applyRequested.connect(self.on_settings_applied)
        self.settings_panel.restartRequested.connect(self.restart_pipeline)
        self.settings_panel.cameraToggleRequested.connect(self.set_camera_enabled)
        self.settings_panel.clearPrivacyDataRequested.connect(self.clear_stored_data)

    def _start_timers(self) -> None:
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)

        self._hardware_timer = QTimer(self)
        self._hardware_timer.timeout.connect(self.refresh_hardware)
        self._hardware_timer.start(HARDWARE_REFRESH_MS)

    def _refresh_panels(self) -> None:
        if self.engine.pipeline is not None:
            self.ai_panel.set_models(
                self.engine.pipeline.detector_stage.describe(),
                self.engine.pipeline.classifier_stage.describe(),
            )
            self.ai_panel.set_quality_note(self.engine.pipeline.expression_quality_note)
        self.settings_panel.set_registry_note()

    def refresh(self) -> None:
        """Poll the engine and repaint. Runs on the Qt thread, never blocks."""
        self.engine.refresh_stats()
        result = self.engine.latest()
        status = self.engine.status
        self._last_result = result

        if result is not None:
            frame = result.frame
            display = self._display_frame(frame)
            heuristic = bool(
                self.engine.pipeline is not None
                and not self.engine.pipeline.uses_trained_expression_model
            )
            model_name = (
                self.engine.pipeline.classifier_stage.classifier.name
                if self.engine.pipeline is not None
                else ""
            )
            self.dashboard.update_result(
                result, display, model_name, heuristic, self._labels_accepted()
            )
            self.ai_panel.update(result, self.settings.models)

        state = self.engine.state
        self.dashboard.set_status(status.describe(), self._status_kind(state))
        self.status_left.setText(
            f"{status.describe()}  |  {status.frames_processed} frames  |  "
            f"{status.frames_dropped} dropped"
        )
        self.status_right.setText(
            f"backend: {self.bridge.backend_name}  |  v{__version__}"
        )

        if state is PipelineState.RUNNING and not self.start_button.text().startswith("Stop"):
            self.start_button.setText("Stop")
        elif state is not PipelineState.RUNNING and self.start_button.text() != "Start":
            self.start_button.setText("Start")
        self._sync_controls()

    def _sync_controls(self) -> None:
        """Keep the header buttons consistent with the engine state."""
        state = self.engine.state
        running = state is PipelineState.RUNNING
        self.start_button.setText("Stop" if running else "Start")
        self.pause_button.setEnabled(running)
        self.restart_button.setEnabled(state in (PipelineState.IDLE, PipelineState.ERROR))
        self._update_camera_indicator()

    def _display_frame(self, frame: np.ndarray | None) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return blank_overlay(960, 540, "Waiting for camera frames")
        if self.settings.camera.mirror:
            import cv2

            return cv2.flip(frame, 1)
        return frame

    def _labels_accepted(self) -> bool:
        return bool(self.settings.ui.privacy_notice_accepted)

    @staticmethod
    def _status_kind(state: PipelineState) -> str:
        return {
            PipelineState.RUNNING: "running",
            PipelineState.PAUSED: "paused",
            PipelineState.ERROR: "error",
        }.get(state, "muted")

    def _update_camera_indicator(self) -> None:
        source = self.engine.source
        if source is None or not getattr(source, "is_open", False):
            self.camera_indicator.setText("Camera: off")
            return
        name = getattr(source, "display_name", "source")
        if isinstance(source, str) or source.__class__.__name__ != "CameraManager":
            self.camera_indicator.setText(f"Source: {name}")
        else:
            target = self.settings.camera.capture_target
            self.camera_indicator.setText(f"Camera: live ({target})")
        self.camera_indicator.setObjectName("StatusRunning")
        self.camera_indicator.style().unpolish(self.camera_indicator)
        self.camera_indicator.style().polish(self.camera_indicator)

    def refresh_hardware(self) -> None:
        snapshot = self.bridge.latest
        if snapshot is not None:
            self.hardware_panel.update_snapshot(snapshot)

    def toggle_pipeline(self) -> None:
        if self.engine.state is PipelineState.RUNNING:
            self.engine.stop()
            self._sync_controls()
            return
        try:
            self.engine.start()
        except VisionAIError as exc:
            self._report_error(exc)
            return
        if self.engine.state is PipelineState.ERROR:
            self._sync_controls()
            self._report_error(RuntimeError(self.engine.status.last_error))
            return
        self.dashboard.show_placeholder("Starting the pipeline")
        self._sync_controls()

    def toggle_pause(self) -> None:
        self.engine.toggle_pause()

    def restart_pipeline(self) -> None:
        """Rebuild the engine so changed model and camera settings take effect."""
        self.engine.stop()
        self.engine.close()
        self.engine = PipelineEngine(self.settings)
        self._refresh_panels()
        self.toggle_pipeline()
        self._needs_restart = False
        self.notice.setText("")

    def set_camera_enabled(self, enabled: bool) -> None:
        self._camera_enabled = bool(enabled)
        self.settings_panel.set_camera_toggle_state(enabled)
        if not enabled:
            self.engine.stop()
            self.dashboard.show_placeholder("Camera disabled. Enable it in Settings to resume.")
        self.notice.setText(
            "" if enabled else "Camera disabled - no frames are being captured or analysed."
        )

    def clear_stored_data(self) -> None:
        from visionai.utils.storage import clear_stored_data

        removed = clear_stored_data()
        self.notice.setText(f"Deleted {removed} stored file(s).")

    def on_settings_applied(self, settings: Settings) -> None:
        self.settings = settings
        theme_changed = settings.ui.theme != self._palette.name
        self._palette = get_palette(settings.ui.theme)
        self.dashboard.set_settings(settings.ui)
        self.dashboard.set_palette(self._palette)
        self.ai_panel.set_palette(self._palette)
        self.hardware_panel.set_palette(self._palette)
        self.dashboard.update_overlay_only()
        if theme_changed:
            application = QApplication.instance()
            if application is not None:
                application.setStyleSheet(stylesheet(self._palette))
        if settings.hardware.enabled:
            self.bridge.settings = settings.hardware
            self.bridge.start()
        else:
            self.bridge.stop()
        self._needs_restart = True
        self.notice.setText(RESTART_REQUIRED_NOTICE)

    def show_about(self) -> None:
        from visionai.hardware.native_loader import describe_environment

        env = describe_environment()
        QMessageBox.about(
            self,
            "About Linux AI Vision",
            f"<b>linux-ai-vision {__version__}</b><br><br>"
            "Real-time face detection with AI-estimated facial expression labels.<br><br>"
            f"Python {env['python']} &middot; OpenCV {env['opencv']}<middot> "
            f"NumPy {env['numpy']}<br>"
            f"Hardware backend: {self.bridge.backend_name}<br>"
            f"Native library: {env['native_library']}<br><br>"
            f"<i>{taxonomy.DISCLAIMER}</i>",
        )

    def show_limitations(self) -> None:
        QMessageBox.information(
            self,
            "What this output means",
            f"<b>{taxonomy.ESTIMATE_PREFIX}</b><br><br>"
            "The label shown is what a model inferred from visible facial features in "
            "one frame. It is not a statement about a person's internal state, and it "
            "cannot detect things a camera cannot see: tone of voice, posture in "
            "context, thoughts, or anything masked by a face covering.<br><br>"
            "Do not use these labels to make decisions about people. See PRIVACY.md "
            "and AI_MODEL.md for the full limitations.",
        )

    def show_privacy(self) -> None:
        QMessageBox.information(
            self,
            "Privacy",
            self.settings_panel.privacy_notice(),
        )

    def _report_error(self, exc: BaseException) -> None:
        LOG.error("pipeline failure: %s", exc)
        self.dashboard.set_status(str(exc), "error")
        QMessageBox.critical(
            self,
            "Pipeline error",
            f"{type(exc).__name__}: {exc}\n\nThe application is still running. "
            "Adjust the settings and press Start to try again.",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop every thread cleanly before the window goes away."""
        self._timer.stop()
        self._hardware_timer.stop()
        try:
            self.engine.close()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("error closing engine: %s", exc)
        try:
            self.bridge.stop()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("error stopping hardware bridge: %s", exc)
        if self.settings.privacy.wipe_on_exit:
            try:
                from visionai.utils.storage import clear_stored_data

                clear_stored_data()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("privacy wipe failed: %s", exc)
        super().closeEvent(event)

    def describe(self) -> dict[str, Any]:
        """State summary used by the UI tests."""
        return {
            "state": self.engine.state.value,
            "tabs": [self.tabs.tabText(i) for i in range(self.tabs.count())],
            "dashboard": self.dashboard.summary_text(),
            "hardware": self.hardware_panel.summary_text(),
            "camera_indicator": self.camera_indicator.text(),
        }
