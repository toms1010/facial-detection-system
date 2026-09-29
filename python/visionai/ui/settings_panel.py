"""Settings tab: camera, models, thresholds, hardware, privacy and logging.

Edits are held in the widgets and only pushed into the :class:`Settings` object
when Apply is pressed, so a half-finished change never reaches the pipeline.
Camera-related changes additionally require a restart to take effect, which the
panel states explicitly rather than silently ignoring.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from visionai.ai.models.registry import CLASSIFIER_ORDER, DETECTOR_ORDER
from visionai.config.paths import PackagePaths
from visionai.config.settings import CAMERA_BACKENDS, Settings
from visionai.ui.theme import Palette
from visionai.ui.widgets import Banner, Card

LOG = logging.getLogger(__name__)


class SettingsPanel(QWidget):
    """Edits every settings group and reports the changes back."""

    applyRequested = Signal(object)
    restartRequested = Signal()
    cameraToggleRequested = Signal(bool)
    clearPrivacyDataRequested = Signal()

    def __init__(
        self,
        settings: Settings,
        palette: Palette,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._palette = palette
        self._registry_note: str = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        scroll.setWidget(container)

        layout.addWidget(self._build_camera_card())
        layout.addWidget(self._build_model_card())
        layout.addWidget(self._build_pipeline_card())
        layout.addWidget(self._build_privacy_card())
        layout.addWidget(self._build_hardware_card())
        layout.addWidget(self._build_logging_card())
        layout.addStretch(1)

        self.load_settings(settings)

    def _build_camera_card(self) -> QWidget:
        card = Card("Camera")
        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        grid.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 32)
        self.camera_path = QLineEdit()
        self.camera_path.setPlaceholderText("/dev/video0 (optional)")
        self.camera_width = QSpinBox()
        self.camera_width.setRange(160, 7680)
        self.camera_width.setSingleStep(160)
        self.camera_height = QSpinBox()
        self.camera_height.setRange(120, 4320)
        self.camera_height.setSingleStep(120)
        self.camera_fps = QSpinBox()
        self.camera_fps.setRange(1, 240)
        self.backend = QComboBox()
        self.backend.addItems(list(CAMERA_BACKENDS))
        self.fourcc = QLineEdit()
        self.fourcc.setMaxLength(4)
        self.mirror = QCheckBox("Mirror the preview")

        grid.addRow("Device index", self.camera_index)
        grid.addRow("Device path", self.camera_path)
        grid.addRow("Resolution width", self.camera_width)
        grid.addRow("Resolution height", self.camera_height)
        grid.addRow("Capture FPS", self.camera_fps)
        grid.addRow("Backend", self.backend)
        grid.addRow("FOURCC", self.fourcc)
        grid.addRow("", self.mirror)
        card.add(_wrap(grid))

        buttons = QHBoxLayout()
        self.camera_toggle = QPushButton("Disable camera")
        self.camera_toggle.setObjectName("Danger")
        self.camera_toggle.setCheckable(True)
        self.camera_toggle.setToolTip(
            "Checked means the camera is enabled. Turning it off stops capture "
            "and analysis immediately."
        )
        self.camera_toggle.clicked.connect(
            lambda: self.cameraToggleRequested.emit(self.camera_toggle.isChecked())
        )
        buttons.addWidget(self.camera_toggle)
        buttons.addStretch(1)
        card.add(_wrap(buttons))

        card.add(
            Banner(
                "Changing the camera, resolution or backend takes effect the next time "
                "the pipeline starts. Press Restart to apply now."
            )
        )
        return card

    def _build_model_card(self) -> QWidget:
        card = Card("Models and inference")
        self.registry_note = Banner("")
        card.add(self.registry_note)

        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.detector = QComboBox()
        self.detector.addItems(list(DETECTOR_ORDER))
        self.classifier = QComboBox()
        self.classifier.addItems(list(CLASSIFIER_ORDER))
        self.detector_weights = QLineEdit()
        self.detector_weights.setPlaceholderText("models/face/<weights>.pt or .onnx")
        self.classifier_weights = QLineEdit()
        self.classifier_weights.setPlaceholderText("models/emotion/<model>.onnx")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda", "cuda:0"])
        self.half_precision = QCheckBox("Use FP16 where supported")
        self.detection_confidence = _double_spin(0.01, 1.0, 0.01, 2)
        self.detection_iou = _double_spin(0.0, 1.0, 0.01, 2)
        self.emotion_confidence = _double_spin(0.0, 1.0, 0.01, 2)
        self.max_faces = QSpinBox()
        self.max_faces.setRange(1, 64)

        grid.addRow("Face detector", self.detector)
        grid.addRow("Detector weights", self.detector_weights)
        grid.addRow("Expression classifier", self.classifier)
        grid.addRow("Classifier weights", self.classifier_weights)
        grid.addRow("Compute device", self.device)
        grid.addRow("", self.half_precision)
        grid.addRow("Detection confidence", self.detection_confidence)
        grid.addRow("Detection NMS IoU", self.detection_iou)
        grid.addRow("Expression confidence", self.emotion_confidence)
        grid.addRow("Max faces", self.max_faces)
        card.add(_wrap(grid))

        banner = Banner(
            "YOLO detects faces; it does not classify expressions. Expression labels "
            "come from the separate classifier selected above. Without a trained "
            "classifier the app falls back to an untrained geometric heuristic and "
            "says so on the AI tab."
        )
        card.add(banner)
        return card

    def _build_pipeline_card(self) -> QWidget:
        card = Card("Pipeline and display")
        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.inference_fps = _double_spin(0.5, 120.0, 1.0, 1)
        self.processing_width = QSpinBox()
        self.processing_width.setRange(160, 7680)
        self.processing_width.setSingleStep(160)
        self.track_iou = _double_spin(0.0, 1.0, 0.01, 2)
        self.track_max_age = QSpinBox()
        self.track_max_age.setRange(0, 600)
        self.smooth_alpha = _double_spin(0.0, 1.0, 0.05, 2)
        self.theme = QComboBox()
        self.theme.addItems(["dark", "light", "system"])
        self.show_boxes = QCheckBox("Bounding boxes")
        self.show_labels = QCheckBox("Expression labels")
        self.show_confidence = QCheckBox("Confidence values")
        self.show_track_ids = QCheckBox("Tracking IDs")
        self.show_fps = QCheckBox("Performance overlay")

        grid.addRow("Inference FPS", self.inference_fps)
        grid.addRow("Processing width", self.processing_width)
        grid.addRow("Tracking IoU", self.track_iou)
        grid.addRow("Track max age", self.track_max_age)
        grid.addRow("Smoothing", self.smooth_alpha)
        grid.addRow("Theme", self.theme)
        for box in (
            self.show_boxes,
            self.show_labels,
            self.show_confidence,
            self.show_track_ids,
            self.show_fps,
        ):
            grid.addRow("", box)
        card.add(_wrap(grid))
        return card

    def _build_privacy_card(self) -> QWidget:
        card = Card("Privacy")
        card.add(Banner(self.privacy_notice()))
        self.store_frames = QCheckBox("Store video frames on disk")
        self.store_snapshots = QCheckBox("Save snapshots of detected faces")
        self.store_face_crops = QCheckBox("Save cropped face images")
        self.allow_upload = QCheckBox("Allow network upload of camera data")
        self.anonymize_logs = QCheckBox("Anonymise machine identifiers in logs")
        self.wipe_on_exit = QCheckBox("Delete stored data when the app exits")
        boxes = QVBoxLayout()
        boxes.setSpacing(4)
        for box in (
            self.store_frames,
            self.store_snapshots,
            self.store_face_crops,
            self.allow_upload,
            self.anonymize_logs,
            self.wipe_on_exit,
        ):
            boxes.addWidget(box)
        card.add(_wrap(boxes))

        wipe = QPushButton("Delete stored camera data now")
        wipe.setObjectName("Danger")
        wipe.clicked.connect(self.clearPrivacyDataRequested.emit)
        card.add(wipe)
        return card

    def _build_hardware_card(self) -> QWidget:
        card = Card("Hardware monitoring")
        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.hardware_enabled = QCheckBox("Enable hardware monitoring")
        self.hardware_interval = _double_spin(0.2, 30.0, 0.1, 1)
        self.disk_path = QLineEdit()
        self.network_interface = QLineEdit()
        self.network_interface.setPlaceholderText("auto (busiest interface)")
        self.prefer_native = QCheckBox("Prefer the compiled C++ layer")

        grid.addRow("", self.hardware_enabled)
        grid.addRow("Sample interval (s)", self.hardware_interval)
        grid.addRow("Disk to watch", self.disk_path)
        grid.addRow("Network interface", self.network_interface)
        grid.addRow("", self.prefer_native)
        card.add(_wrap(grid))
        return card

    def _build_logging_card(self) -> QWidget:
        card = Card("Logging")
        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_console = QCheckBox("Log to the terminal")
        self.log_file = QCheckBox("Log to a rotating file")
        self.log_directory = QLineEdit()
        self.log_directory.setPlaceholderText("default: ~/.local/share/linux-ai-vision/logs")
        grid.addRow("Level", self.log_level)
        grid.addRow("", self.log_console)
        grid.addRow("", self.log_file)
        grid.addRow("Log directory", self.log_directory)
        card.add(_wrap(grid))
        return card

    def privacy_notice(self) -> str:
        from visionai.ai.taxonomy import DISCLAIMER

        return (
            "All processing happens on this machine. The application contains no "
            "network upload path, and camera frames are never written to disk unless "
            "you explicitly enable storage below. Detected expressions are an AI "
            "estimate from visible facial features, not a measurement of how anyone "
            f"feels. {DISCLAIMER}"
        )

    def load_settings(self, settings: Settings) -> None:
        """Push a settings object into the widgets."""
        self._settings = settings
        camera, models = settings.camera, settings.models
        self.camera_index.setValue(camera.index)
        self.camera_path.setText(camera.device_path or "")
        self.camera_width.setValue(camera.width)
        self.camera_height.setValue(camera.height)
        self.camera_fps.setValue(camera.fps)
        self.backend.setCurrentText(camera.backend)
        self.fourcc.setText(camera.fourcc)
        self.mirror.setChecked(camera.mirror)

        self.detector.setCurrentText(models.detector)
        self.classifier.setCurrentText(models.classifier)
        self.detector_weights.setText(models.detector_weights or "")
        self.classifier_weights.setText(models.classifier_weights or "")
        self.device.setCurrentText(models.device)
        self.half_precision.setChecked(models.half_precision)
        self.detection_confidence.setValue(models.detection_confidence)
        self.detection_iou.setValue(models.detection_iou)
        self.emotion_confidence.setValue(models.emotion_confidence)
        self.max_faces.setValue(models.max_faces)

        pipeline, ui = settings.pipeline, settings.ui
        self.inference_fps.setValue(pipeline.inference_fps)
        self.processing_width.setValue(pipeline.processing_width)
        self.track_iou.setValue(pipeline.track_iou_threshold)
        self.track_max_age.setValue(pipeline.track_max_age)
        self.smooth_alpha.setValue(pipeline.smooth_alpha)
        self.theme.setCurrentText(ui.theme)
        self.show_boxes.setChecked(ui.show_boxes)
        self.show_labels.setChecked(ui.show_labels)
        self.show_confidence.setChecked(ui.show_confidence)
        self.show_track_ids.setChecked(ui.show_track_ids)
        self.show_fps.setChecked(ui.show_fps)

        privacy = settings.privacy
        self.store_frames.setChecked(privacy.store_frames)
        self.store_snapshots.setChecked(privacy.store_snapshots)
        self.store_face_crops.setChecked(privacy.store_face_crops)
        self.allow_upload.setChecked(privacy.allow_network_upload)
        self.anonymize_logs.setChecked(privacy.anonymize_log_payloads)
        self.wipe_on_exit.setChecked(privacy.wipe_on_exit)

        hardware = settings.hardware
        self.hardware_enabled.setChecked(hardware.enabled)
        self.hardware_interval.setValue(hardware.interval)
        self.disk_path.setText(hardware.disk_path)
        self.network_interface.setText(hardware.network_interface or "")
        self.prefer_native.setChecked(hardware.prefer_native)

        logging_settings = settings.logging
        self.log_level.setCurrentText(logging_settings.level)
        self.log_console.setChecked(logging_settings.console)
        self.log_file.setChecked(logging_settings.file)
        self.log_directory.setText(logging_settings.directory or "")

        self.set_registry_note()

    def collect_settings(self) -> Settings:
        """Read every widget into a validated settings object."""
        settings = Settings()
        camera = settings.camera
        camera.index = self.camera_index.value()
        camera.device_path = self.camera_path.text().strip() or None
        camera.width = self.camera_width.value()
        camera.height = self.camera_height.value()
        camera.fps = self.camera_fps.value()
        camera.backend = self.backend.currentText()
        camera.fourcc = self.fourcc.text().strip() or "MJPG"
        camera.mirror = self.mirror.isChecked()

        models = settings.models
        models.detector = self.detector.currentText()
        models.classifier = self.classifier.currentText()
        models.detector_weights = self.detector_weights.text().strip() or None
        models.classifier_weights = self.classifier_weights.text().strip() or None
        models.device = self.device.currentText()
        models.half_precision = self.half_precision.isChecked()
        models.detection_confidence = self.detection_confidence.value()
        models.detection_iou = self.detection_iou.value()
        models.emotion_confidence = self.emotion_confidence.value()
        models.max_faces = self.max_faces.value()

        pipeline = settings.pipeline
        pipeline.inference_fps = self.inference_fps.value()
        pipeline.processing_width = self.processing_width.value()
        pipeline.track_iou_threshold = self.track_iou.value()
        pipeline.track_max_age = self.track_max_age.value()
        pipeline.smooth_alpha = self.smooth_alpha.value()

        ui = settings.ui
        ui.theme = self.theme.currentText()
        ui.show_boxes = self.show_boxes.isChecked()
        ui.show_labels = self.show_labels.isChecked()
        ui.show_confidence = self.show_confidence.isChecked()
        ui.show_track_ids = self.show_track_ids.isChecked()
        ui.show_fps = self.show_fps.isChecked()

        privacy = settings.privacy
        privacy.store_frames = self.store_frames.isChecked()
        privacy.store_snapshots = self.store_snapshots.isChecked()
        privacy.store_face_crops = self.store_face_crops.isChecked()
        privacy.allow_network_upload = self.allow_upload.isChecked()
        privacy.anonymize_log_payloads = self.anonymize_logs.isChecked()
        privacy.wipe_on_exit = self.wipe_on_exit.isChecked()

        hardware = settings.hardware
        hardware.enabled = self.hardware_enabled.isChecked()
        hardware.interval = self.hardware_interval.value()
        hardware.disk_path = self.disk_path.text().strip() or "/"
        hardware.network_interface = self.network_interface.text().strip() or None
        hardware.prefer_native = self.prefer_native.isChecked()

        logging_settings = settings.logging
        logging_settings.level = self.log_level.currentText()
        logging_settings.console = self.log_console.isChecked()
        logging_settings.file = self.log_file.isChecked()
        logging_settings.directory = self.log_directory.text().strip() or None

        return settings.validate()

    def apply(self) -> Settings:
        settings = self.collect_settings()
        self.applyRequested.emit(settings)
        self.load_settings(settings)
        return settings

    def set_registry_note(self, note: str = "") -> None:
        """Describe which model files were found on disk."""
        if note:
            self._registry_note_text = note
        else:
            paths = PackagePaths.discover()
            face_models = sorted(p.name for p in paths.face_models.glob("*") if p.is_file())
            emotion_models = sorted(p.name for p in paths.emotion_models.glob("*") if p.is_file())
            self._registry_note_text = (
                f"Face models found: {', '.join(face_models) or 'none'}    |    "
                f"Expression models found: {', '.join(emotion_models) or 'none - the heuristic fallback will be used'}"
            )
        self.registry_note.setText(self._registry_note_text)

    def set_camera_toggle_state(self, enabled: bool) -> None:
        self.camera_toggle.setChecked(enabled)
        self.camera_toggle.setText("Disable camera" if enabled else "Enable camera")

    def error_banner(self, message: str) -> Banner:
        return Banner(message)


def _double_spin(low: float, high: float, step: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    return spin


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget
