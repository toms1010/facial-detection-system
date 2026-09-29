"""Dashboard tab: live camera preview, face table and pipeline metrics."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from visionai.ai.emotion_classifier import UNKNOWN_LABEL
from visionai.ai.face_tracker import TrackedFace
from visionai.ai.inference import PipelineResult
from visionai.config.settings import UISettings
from visionai.ui.overlay import OverlayMetrics, RenderOptions, render_overlay
from visionai.ui.theme import Palette
from visionai.ui.video_widget import VideoWidget
from visionai.ui.widgets import Card, MetricTile, UsageBar

LOG = logging.getLogger(__name__)

COLUMNS = ("ID", "Detected expression", "Confidence", "Model note")


class DashboardPanel(QWidget):
    """The main view: preview plus a live table of tracked faces."""

    def __init__(self, palette: Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._settings = UISettings()
        self._last_result: PipelineResult | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.video = VideoWidget()
        self.video.set_placeholder("Camera is off. Press Start to begin.")
        layout.addWidget(self.video, stretch=1)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("MetricMuted")
        layout.addWidget(self.status_label)

        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.tile_fps = MetricTile("FPS", "--", palette)
        self.tile_latency = MetricTile("Inference", "--", palette)
        self.tile_faces = MetricTile("Faces", "0", palette)
        self.tile_model = MetricTile("Expression model", "--", palette)
        for tile in (self.tile_fps, self.tile_latency, self.tile_faces, self.tile_model):
            tiles.addWidget(tile)
        layout.addLayout(tiles)

        self.latency_bar = UsageBar("Inference time vs 33 ms budget", palette, maximum=33.0)
        layout.addWidget(self.latency_bar)

        table_card = Card("Detected facial expressions (AI estimates)")
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setMinimumHeight(150)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(1, header.ResizeMode.Stretch)
        table_card.add(self.table)
        layout.addWidget(table_card)

        self._apply_palette(palette)

    def _apply_palette(self, palette: Palette) -> None:
        for tile in (self.tile_fps, self.tile_latency, self.tile_faces, self.tile_model):
            tile.set_palette_colors(palette)
        self.latency_bar.set_palette_colors(palette)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self._apply_palette(palette)

    def set_settings(self, settings: UISettings) -> None:
        self._settings = settings

    def set_status(self, text: str, kind: str = "muted") -> None:
        self.status_label.setText(text)
        object_name = {
            "running": "StatusRunning",
            "paused": "StatusPaused",
            "error": "StatusError",
        }.get(kind, "MetricMuted")
        self.status_label.setObjectName(object_name)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def show_placeholder(self, message: str) -> None:
        self.video.set_placeholder(message)
        self._clear_table()

    def update_result(
        self,
        result: PipelineResult | None,
        display_frame: np.ndarray | None = None,
        model_name: str = "",
        heuristic: bool = False,
        label_accepted: bool = True,
    ) -> None:
        """Push the newest pipeline result into the preview and the table."""
        if result is None:
            return
        self._last_result = result
        tracks = result.tracks

        options = RenderOptions(settings=self._settings, palette=self._palette)
        metrics = OverlayMetrics(
            fps=result.total_ms and 1000.0 / max(result.total_ms, 1e-3) or 0.0,
            inference_ms=result.total_ms,
            detection_ms=result.detection_ms,
            classification_ms=result.classification_ms,
            face_count=result.face_count,
            model=model_name,
            heuristic=heuristic,
        )
        if display_frame is not None:
            rendered = render_overlay(display_frame, tracks, options, metrics, label_accepted)
            self.video.set_frame(rendered)
        else:
            self.video.set_placeholder("Waiting for a camera frame")

        self._populate_table(tracks, label_accepted)
        self.tile_fps.set_value(
            f"{metrics.fps:.0f}" if metrics.fps else "--", "inference rate"
        )
        self.tile_latency.set_value(
            f"{result.total_ms:.0f} ms",
            f"detect {result.detection_ms:.0f} / classify {result.classification_ms:.0f} ms",
        )
        self.tile_faces.set_value(str(result.face_count), "tracked faces")
        self.tile_model.set_value(
            model_name or "--", "heuristic fallback" if heuristic else "trained model"
        )
        self.latency_bar.set_value(
            min(result.total_ms, 33.0) / 33.0 * 100.0,
            f"{result.total_ms:.1f} ms",
        )

    def update_overlay_only(self) -> None:
        """Re-render the stored result after a display setting changed."""
        if self._last_result is None:
            return
        frame = self._last_result.frame
        if frame is not None:
            options = RenderOptions(settings=self._settings, palette=self._palette)
            self.video.set_frame(render_overlay(frame, self._last_result.tracks, options))

    def _populate_table(self, tracks: list[TrackedFace], label_accepted: bool = True) -> None:
        labelled = [t for t in tracks if t.expression is not None]
        if self.table.rowCount() != len(labelled):
            self.table.setRowCount(len(labelled))
        for row, track in enumerate(labelled):
            expression = track.expression
            label = expression.label if expression else UNKNOWN_LABEL
            accepted = bool(expression and expression.is_confident) and label_accepted
            shown = (
                f"{expression.emoji} {label.title()}"
                if expression and accepted
                else "Below confidence threshold"
            )
            note = "AI estimate"
            if expression is not None and expression.is_heuristic:
                note = "heuristic, untrained"
            if not label_accepted:
                note = "accept the privacy notice to show labels"
            self._set_cell(row, 0, f"{track.track_id:02d}")
            self._set_cell(row, 1, shown)
            self._set_cell(
                row, 2, f"{expression.confidence * 100:.0f}%" if expression and accepted else "--"
            )
            self._set_cell(row, 3, note)

    def _set_cell(self, row: int, column: int, text: str) -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(text)
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignLeft
            if column in (1, 3)
            else Qt.AlignmentFlag.AlignCenter
        )

    def _clear_table(self) -> None:
        self.table.setRowCount(0)

    def summary_text(self) -> str:
        """Text used by the UI tests to assert the panel updated."""
        rows = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            rows.append("" if item is None else item.text())
        return f"{self.tile_fps.text()} fps | {self.tile_latency.text()} | faces={rows}"
