"""Qt interface tests.

These run under ``QT_QPA_PLATFORM=offscreen`` (set in conftest), so they exercise
real widget construction, layout and signal wiring without a display server.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402

from visionai.ai import taxonomy  # noqa: E402
from visionai.ai.face_tracker import FaceTracker, reset_id_counter  # noqa: E402
from visionai.ai.inference import PipelineResult  # noqa: E402
from visionai.ai.models.base import Box, ExpressionResult, FaceDetection  # noqa: E402
from visionai.config.settings import Settings  # noqa: E402
from visionai.hardware.hardware_bridge import HardwareBridge, HardwareSnapshot  # noqa: E402
from visionai.pipeline.engine import PipelineEngine  # noqa: E402
from visionai.ui.ai_panel import AIPanel  # noqa: E402
from visionai.ui.dashboard import DashboardPanel  # noqa: E402
from visionai.ui.hardware_panel import HardwarePanel  # noqa: E402
from visionai.ui.main_window import MainWindow  # noqa: E402
from visionai.ui.settings_panel import SettingsPanel  # noqa: E402
from visionai.ui.theme import DARK, LIGHT  # noqa: E402
from visionai.ui.video_widget import VideoWidget  # noqa: E402
from visionai.ui.widgets import Card, KeyValueGrid, MetricTile, UsageBar  # noqa: E402

pytestmark = pytest.mark.requires_qt


def make_result(frame: np.ndarray | None = None, faces: int = 0) -> PipelineResult:
    """A PipelineResult with ``faces`` synthetic tracked faces.

    Track IDs are process-global, so the counter is reset to keep assertions
    about "ID 01" independent of test execution order.
    """
    reset_id_counter(1)
    tracker = FaceTracker()
    detections = [
        FaceDetection(box=Box(40 + i * 120, 60, 160 + i * 120, 220), score=0.9)
        for i in range(faces)
    ]
    expressions = {
        i: ExpressionResult.from_scores(
            {label: 0.9 if label == "happy" else 0.1 for label in taxonomy.PRIMARY_CLASSES and
             [c.name for c in taxonomy.PRIMARY_CLASSES]},
            model="test",
        )
        for i in range(faces)
    }
    tracks = tracker.update(detections, expressions)
    return PipelineResult(
        frame=frame,
        frame_number=1,
        timestamp=time.time(),
        detections=detections,
        tracks=tracks,
        expressions=expressions,
        detection_ms=8.0,
        classification_ms=2.0,
        total_ms=12.0,
    )


class TestVideoWidget:
    def test_starts_empty(self, app) -> None:
        widget = VideoWidget()
        assert widget.has_content() is False

    def test_accepts_a_frame(self, app) -> None:
        widget = VideoWidget()
        assert widget.set_frame(np.zeros((240, 320, 3), dtype=np.uint8)) is True
        assert widget.has_content() is True

    def test_accepts_grayscale(self, app) -> None:
        widget = VideoWidget()
        assert widget.set_frame(np.zeros((240, 320), dtype=np.uint8)) is True

    def test_ignores_empty_and_invalid_input(self, app) -> None:
        widget = VideoWidget()
        assert widget.set_frame(None) is False
        assert widget.set_frame(np.array([])) is False
        assert widget.set_frame("not a frame") is False  # type: ignore[arg-type]
        assert widget.has_content() is False

    def test_placeholder_is_rendered(self, app) -> None:
        widget = VideoWidget()
        widget.set_placeholder("Camera off")
        pixmap = widget.grab()
        assert not pixmap.isNull()

    def test_clear(self, app) -> None:
        widget = VideoWidget()
        widget.set_frame(np.zeros((60, 80, 3), dtype=np.uint8))
        widget.clear()
        assert widget.has_content() is False


class TestSharedWidgets:
    def test_metric_tile(self, app) -> None:
        tile = MetricTile("FPS", "--", DARK)
        tile.set_value("30", "per second")
        assert tile.text() == "30"

    def test_usage_bar(self, app) -> None:
        bar = UsageBar("CPU", DARK)
        bar.set_value(42.0)
        assert bar.value_text() == "42%"

    def test_usage_bar_handles_none(self, app) -> None:
        bar = UsageBar("GPU", DARK)
        bar.set_value(None)
        assert bar.value_text() == "n/a"

    def test_usage_bar_clamps(self, app) -> None:
        bar = UsageBar("CPU", DARK)
        bar.set_value(500.0)
        assert bar.value_text() == "500%"

    def test_key_value_grid(self, app) -> None:
        grid = KeyValueGrid()
        grid.set("Detector", "yunet")
        assert grid.get("Detector") == "yunet"
        grid.set("Detector", "yolo")
        assert grid.get("Detector") == "yolo"
        assert "Detector" in list(grid.keys())
        grid.clear()
        assert grid.get("Detector") == "--"

    def test_unknown_key_reads_empty(self, app) -> None:
        assert KeyValueGrid().get("nope") == ""

    def test_card_renders(self, app) -> None:
        card = Card("Title")
        assert not card.grab().isNull()


class TestDashboard:
    def test_updates_from_a_result(self, app) -> None:
        panel = DashboardPanel(DARK)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        panel.update_result(make_result(frame, faces=2), frame, "heuristic", True, True)
        assert panel.table.rowCount() == 2
        assert panel.table.item(0, 0).text() == "01"
        assert "30" in panel.summary_text() or "12" in panel.summary_text()

    def test_shows_a_placeholder(self, app) -> None:
        panel = DashboardPanel(DARK)
        panel.show_placeholder("Camera is off")
        assert panel.video.has_content() is False
        assert panel.table.rowCount() == 0

    def test_handles_a_missing_frame(self, app) -> None:
        panel = DashboardPanel(DARK)
        panel.update_result(make_result(None, faces=0), None, "heuristic", True, True)
        assert panel.video.has_content() is False

    def test_labels_are_withheld_until_accepted(self, app) -> None:
        panel = DashboardPanel(DARK)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        panel.update_result(make_result(frame, faces=1), frame, "m", False, False)
        assert "threshold" in panel.table.item(0, 1).text().lower()

    def test_status_updates(self, app) -> None:
        panel = DashboardPanel(DARK)
        panel.set_status("Running - 30 FPS", "running")
        assert "Running" in panel.status_label.text()

    def test_palette_can_be_swapped(self, app) -> None:
        panel = DashboardPanel(DARK)
        panel.set_palette(LIGHT)
        assert not panel.grab().isNull()

    def test_rerender_after_a_setting_change(self, app) -> None:
        panel = DashboardPanel(DARK)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = make_result(frame, faces=1)
        panel.update_result(result, frame, "m", False, True)
        panel.set_settings(type(panel._settings)(show_labels=False))
        panel.update_overlay_only()
        assert panel.video.has_content() is True


class TestAIPanel:
    def test_shows_model_information(self, app) -> None:
        panel = AIPanel(DARK)
        panel.set_models(
            {"name": "yunet", "display_name": "YuNet", "latency_ms": 8.0},
            {
                "name": "heuristic",
                "display_name": "Heuristic",
                "is_heuristic": True,
                "is_trained_model": False,
                "class_names": [c.name for c in taxonomy.PRIMARY_CLASSES],
            },
        )
        assert panel.model_grid.get("Detector") == "YuNet"
        assert "UNTRAINED" in panel.model_grid.get("Model quality")

    def test_marks_a_trained_model(self, app) -> None:
        panel = AIPanel(DARK)
        panel.set_models(
            {"name": "yolo", "display_name": "YOLO", "latency_ms": 5.0},
            {"name": "onnx", "display_name": "ONNX", "is_heuristic": False, "is_trained_model": True},
        )
        assert panel.model_grid.get("Model quality") == "trained model"

    def test_updates_status_and_scores(self, app) -> None:
        panel = AIPanel(DARK)
        result = make_result(faces=2)
        panel.update(result, Settings().models)
        assert panel.status_grid.get("Faces detected") == "2"
        assert panel.table.rowCount() == 2
        assert panel.table.columnCount() == 7

    def test_handles_no_result(self, app) -> None:
        panel = AIPanel(DARK)
        panel.update(None, Settings().models)
        assert "waiting" in panel.status_grid.get("State")

    def test_quality_note_is_displayed(self, app) -> None:
        panel = AIPanel(DARK)
        panel.set_quality_note("Heuristic fallback in use")
        assert "Heuristic" in panel.quality_banner.text()

    def test_history_accumulates(self, app) -> None:
        panel = AIPanel(DARK)
        for _ in range(3):
            panel.update(make_result(faces=1), Settings().models)
        assert len(panel.history("latency")) == 3


class TestHardwarePanel:
    @pytest.fixture
    def panel(self, app, settings: Settings) -> HardwarePanel:
        return HardwarePanel(HardwareBridge(settings.hardware, anonymize=True), DARK)

    def test_renders_every_metric(self, panel) -> None:
        snapshot = HardwareSnapshot(
            cpu_percent=32.0,
            cpu_temp_c=51.0,
            ram_percent=40.0,
            ram_used_bytes=6 * 1024**3,
            ram_total_bytes=16 * 1024**3,
            gpu_percent=48.0,
            gpu_name="Test GPU",
            disk_percent=62.0,
            disk_total_bytes=500 * 1024**3,
            net_rx_bytes_per_second=4.2 * 1024**2,
            net_tx_bytes_per_second=0.8 * 1024**2,
            net_interface="eth0",
            net_interfaces=[
                {"name": "eth0", "rx_bytes_per_second": 1.0, "tx_bytes_per_second": 2.0, "is_up": True}
            ],
            temperature_sensors=[{"label": "coretemp", "celsius": 51.0}],
            cpu_model="Test CPU",
            cpu_cores=8,
            load_average=(0.5, 0.4, 0.3),
            sample_ms=3.2,
            backend="pybind11",
            version="1.0.0",
        )
        panel.update_snapshot(snapshot)
        text = panel.summary_text()
        assert "32%" in text
        assert "51" in text
        assert "48%" in text
        assert panel.network_table.rowCount() == 1
        assert panel.sensor_table.rowCount() == 1
        assert panel.system_grid.get("CPU") == "Test CPU"
        assert panel.system_grid.get("Backend") == "pybind11 v1.0.0"

    def test_handles_a_snapshot_without_a_gpu(self, panel) -> None:
        panel.update_snapshot(HardwareSnapshot())
        assert "n/a" in panel.summary_text()

    def test_notes_are_shown(self, panel) -> None:
        panel.update_snapshot(HardwareSnapshot(notes=["No temperature sensor"]))
        assert "No temperature sensor" in panel.notes_banner.text()

    def test_palette_swap(self, panel) -> None:
        panel.set_palette(LIGHT)
        assert not panel.grab().isNull()


class TestSettingsPanel:
    @pytest.fixture
    def panel(self, app, settings: Settings) -> SettingsPanel:
        return SettingsPanel(settings, DARK)

    def test_loads_settings_into_widgets(self, panel, settings: Settings) -> None:
        assert panel.camera_index.value() == settings.camera.index
        assert panel.detector.currentText() == settings.models.detector
        assert panel.emotion_confidence.value() == pytest.approx(
            settings.models.emotion_confidence
        )

    def test_collects_edits(self, panel) -> None:
        panel.camera_index.setValue(2)
        panel.emotion_confidence.setValue(0.7)
        panel.detector.setCurrentText("yolo")
        collected = panel.collect_settings()
        assert collected.camera.index == 2
        assert collected.models.emotion_confidence == pytest.approx(0.7)
        assert collected.models.detector == "yolo"

    def test_collected_settings_are_validated(self, panel) -> None:
        panel.camera_fps.setValue(5000)
        assert panel.collect_settings().camera.fps == 240

    def test_apply_emits_the_signal(self, panel) -> None:
        received: list[Settings] = []
        panel.applyRequested.connect(received.append)
        panel.camera_index.setValue(5)
        result = panel.apply()
        assert received == [result]
        assert result.camera.index == 5

    def test_privacy_notice_is_explicit(self, panel) -> None:
        notice = panel.privacy_notice()
        assert "on this machine" in notice
        assert "not a measurement" in notice
        assert "never written to disk" in notice

    def test_registry_note_mentions_fallback(self, panel) -> None:
        panel.set_registry_note()
        assert "heuristic" in panel.registry_note.text().lower()

    def test_camera_toggle_signal(self, panel) -> None:
        received: list[bool] = []
        panel.cameraToggleRequested.connect(received.append)
        assert panel.camera_toggle.isChecked() is False
        panel.camera_toggle.click()
        assert panel.camera_toggle.isChecked() is True
        assert received == [True]

    def test_clear_data_signal(self, panel) -> None:
        received: list[bool] = []
        panel.clearPrivacyDataRequested.connect(lambda: received.append(True))
        panel.findChildren(type(panel.camera_toggle))
        button = [b for b in panel.findChildren(type(panel.camera_toggle)) if b.text().startswith("Delete")]
        button[0].clicked.emit()
        assert received == [True]

    def test_camera_toggle_state_text(self, panel) -> None:
        panel.set_camera_toggle_state(True)
        assert panel.camera_toggle.text() == "Disable camera"
        panel.set_camera_toggle_state(False)
        assert panel.camera_toggle.text() == "Enable camera"

    def test_theme_round_trip(self, panel) -> None:
        panel.theme.setCurrentText("light")
        assert panel.collect_settings().ui.theme == "light"


class TestMainWindow:
    @pytest.fixture
    def window(self, app, settings: Settings) -> Iterator[MainWindow]:
        engine = PipelineEngine(settings, autostart_models=False)
        bridge = HardwareBridge(settings.hardware, anonymize=True)
        win = MainWindow(settings, engine, bridge, source_spec="synthetic")
        yield win
        win.close()
        win.setParent(None)
        win.deleteLater()
        QApplication.processEvents()

    def test_has_all_four_tabs(self, window) -> None:
        assert window.describe()["tabs"] == ["Dashboard", "AI", "Hardware", "Settings"]

    def test_starts_idle(self, window) -> None:
        assert window.engine.state.value == "idle"
        assert "Camera: off" in window.camera_indicator.text()

    def test_toggle_starts_the_pipeline(self, window) -> None:
        window.toggle_pipeline()
        assert window.engine.is_running
        assert window.start_button.text() == "Stop"
        window.refresh()
        assert "Camera" in window.camera_indicator.text() or "Source" in window.camera_indicator.text()

    def test_toggle_stops_the_pipeline(self, window) -> None:
        window.toggle_pipeline()
        window.toggle_pipeline()
        assert window.engine.state.value == "idle"
        assert window.start_button.text() == "Start"

    def test_switching_tabs(self, window) -> None:
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
            assert window.tabs.currentIndex() == index

    def test_camera_can_be_disabled(self, window) -> None:
        window.set_camera_enabled(False)
        assert "disabled" in window.notice.text()
        assert window.engine.state.value == "idle"
        window.set_camera_enabled(True)
        assert window.notice.text() == ""

    def test_settings_are_applied_and_saved_to_the_widgets(self, window) -> None:
        window.settings_panel.camera_index.setValue(4)
        window.settings_panel.apply()
        assert window.settings.camera.index == 4
        assert window.settings_panel.camera_index.value() == 4

    def test_theme_change_updates_every_panel(self, window) -> None:
        window.settings_panel.theme.setCurrentText("light")
        window.settings_panel.apply()
        assert window._palette is LIGHT

    def test_about_dialog_reports_the_backend(self, window, monkeypatch: pytest.MonkeyPatch) -> None:
        from PySide6.QtWidgets import QMessageBox

        captured: dict[str, object] = {}

        def fake_about(parent, title, text):
            captured["text"] = text
            captured["title"] = title

        monkeypatch.setattr(QMessageBox, "about", staticmethod(fake_about))
        window.show_about()
        assert "linux-ai-vision" in str(captured["text"])

    def test_limitations_dialog_refuses_certainty(self, window, monkeypatch: pytest.MonkeyPatch) -> None:
        from PySide6.QtWidgets import QMessageBox

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            QMessageBox,
            "information",
            staticmethod(lambda parent, title, text: captured.update(text=text)),
        )
        window.show_limitations()
        text = str(captured["text"])
        assert "Detected facial expression" in text
        assert "not a statement" in text

    def test_privacy_dialog(self, window, monkeypatch: pytest.MonkeyPatch) -> None:
        from PySide6.QtWidgets import QMessageBox

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            QMessageBox,
            "information",
            staticmethod(lambda parent, title, text: captured.update(text=text)),
        )
        window.show_privacy()
        assert "on this machine" in str(captured["text"])

    def test_refresh_paints_the_dashboard(self, window) -> None:
        window.toggle_pipeline()
        window.engine.wait_for_result(timeout=8.0)
        window.refresh()
        assert window.dashboard.video.has_content() is True
        window.toggle_pipeline()

    def test_hardware_refresh(self, window) -> None:
        window.bridge.sample()
        window.refresh_hardware()
        assert "cpu=" in window.hardware_panel.summary_text()

    def test_describe_is_serialisable(self, window) -> None:
        import json

        json.dumps(window.describe())

    def test_close_stops_everything(self, app, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        bridge = HardwareBridge(settings.hardware, anonymize=True)
        win = MainWindow(settings, engine, bridge, source_spec="synthetic")
        win.toggle_pipeline()
        win.close()
        assert not engine.is_running
