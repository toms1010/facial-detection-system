"""AI panel: model identity, detection status, per-face score breakdown."""

from __future__ import annotations

import logging
from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from visionai.ai import taxonomy
from visionai.ai.inference import PipelineResult
from visionai.config.settings import ModelSettings
from visionai.ui.theme import Palette
from visionai.ui.widgets import Banner, Card, KeyValueGrid, UsageBar

LOG = logging.getLogger(__name__)

HISTORY = 240


class AIPanel(QWidget):
    """Shows which models are loaded and what they currently report."""

    def __init__(self, palette: Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._latency_history: deque[float] = deque(maxlen=HISTORY)
        self._fps_history: deque[float] = deque(maxlen=HISTORY)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.quality_banner = Banner("")
        layout.addWidget(self.quality_banner)

        model_card = Card("Models")
        self.model_grid = KeyValueGrid()
        model_card.add(self.model_grid)
        layout.addWidget(model_card)

        status_card = Card("Detection status")
        self.status_grid = KeyValueGrid()
        status_card.add(self.status_grid)
        self.latency_bar = UsageBar("Inference latency", palette, maximum=100.0)
        status_card.add(self.latency_bar)
        layout.addWidget(status_card)

        self.expressions_card = Card("Score distribution by tracked face")
        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.expressions_card.add(self.table)
        layout.addWidget(self.expressions_card, stretch=1)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.latency_bar.set_palette_colors(palette)

    def set_models(self, detector_info: dict, classifier_info: dict) -> None:
        """Populate the model card from the pipeline descriptions."""
        self.model_grid.set("Detector", str(detector_info.get("display_name", "--")))
        self.model_grid.set(
            "Detector backend", f"{detector_info.get('name', '--')} ({detector_info.get('latency_ms', 0):.1f} ms)"
        )
        weights = detector_info.get("weights") or detector_info.get("cascade")
        if weights:
            self.model_grid.set("Detector weights", str(weights))
        report = detector_info.get("report")
        if report:
            self.model_grid.set("Detector note", str(report))

        self.model_grid.set(
            "Expression classifier", str(classifier_info.get("display_name", "--"))
        )
        classes = classifier_info.get("class_names") or taxonomy.DEFAULT_CLASS_NAMES
        self.model_grid.set("Classes", f"{len(classes)} ({', '.join(classes[:7])}...)")
        heuristic = bool(classifier_info.get("is_heuristic", False))
        if heuristic:
            self.model_grid.set("Model quality", "UNTRAINED HEURISTIC - not a real prediction")
        else:
            self.model_grid.set("Model quality", "trained model")
        descriptor = classifier_info.get("descriptor")
        if descriptor:
            self.model_grid.set("Model name", str(descriptor.get("name", "--")))
            if descriptor.get("license"):
                self.model_grid.set("Model licence", str(descriptor["license"]))
        classifier_report = classifier_info.get("report")
        if classifier_report:
            self.model_grid.set("Classifier note", str(classifier_report))

    def set_quality_note(self, note: str) -> None:
        self.quality_banner.setText(note)

    def update(self, result: PipelineResult | None, settings: ModelSettings) -> None:
        """Refresh the status card and the per-face score table."""
        if result is None:
            self.status_grid.set("State", "waiting for the first frame")
            return

        self._latency_history.append(result.total_ms)
        observed = result.total_ms and 1000.0 / max(result.total_ms, 1e-3) or 0.0
        if observed:
            self._fps_history.append(observed)
        self.status_grid.set("Faces detected", str(len(result.detections)))
        self.status_grid.set("Faces tracked", str(len(result.tracks)))
        self.status_grid.set("Detection time", f"{result.detection_ms:.1f} ms")
        self.status_grid.set("Classification time", f"{result.classification_ms:.1f} ms")
        self.status_grid.set("Total inference", f"{result.total_ms:.1f} ms")
        self.status_grid.set("Frame number", str(result.frame_number))
        if result.error:
            self.status_grid.set("Last error", result.error)
        self.latency_bar.set_value(
            min(result.total_ms, 100.0), f"{result.total_ms:.1f} ms"
        )
        self._populate_expressions(result, settings)

    def _populate_expressions(self, result: PipelineResult, settings: ModelSettings) -> None:
        entries: list[tuple[int, str, dict[str, float], bool, bool]] = []
        for track in result.tracks:
            expression = track.expression
            if expression is None:
                continue
            entries.append(
                (
                    track.track_id,
                    expression.label,
                    expression.scores,
                    expression.is_confident and expression.confidence >= settings.emotion_confidence,
                    expression.is_heuristic,
                )
            )

        if self.table.columnCount() == 0:
            self.table.setColumnCount(len(entries[0][2]) if entries else 1)
            self.table.setHorizontalHeaderLabels(
                list(entries[0][2].keys()) if entries else ["no faces"]
            )
            self.table.horizontalHeader().setStretchLastSection(True)

        self.table.setRowCount(len(entries))
        for row, (track_id, _label, scores, _accepted, _heuristic) in enumerate(entries):
            header = self.table.verticalHeaderItem(row)
            if header is None:
                header = QTableWidgetItem()
                self.table.setVerticalHeaderItem(row, header)
            header.setText(f"ID {track_id:02d}")
            for column, (name, score) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1])):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, column, item)
                item.setText(f"{score * 100:.0f}%")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
                red, green, blue = taxonomy.color_for(name)
                item.setForeground(QColor(red, green, blue))

    def history(self, attribute: str = "latency") -> list[float]:
        source = self._latency_history if attribute == "latency" else self._fps_history
        return list(source)
