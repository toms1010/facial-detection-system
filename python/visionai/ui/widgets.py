"""Reusable small widgets shared by the dashboard and panels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from visionai.ui.theme import Palette


class Card(QFrame):
    """A titled panel container."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setObjectName("CardTitle")
            self._layout.addWidget(label)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget


class MetricTile(QFrame):
    """A large value with a caption, used across the dashboard."""

    def __init__(
        self,
        caption: str,
        value: str = "--",
        palette: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._caption = caption
        self._palette = palette
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self._caption_label = QLabel(caption)
        self._caption_label.setObjectName("CardTitle")
        self._value_label = QLabel(value)
        self._value_label.setObjectName("Metric")
        self._detail_label = QLabel("")
        self._detail_label.setObjectName("MetricMuted")
        self._detail_label.setWordWrap(True)

        layout.addWidget(self._caption_label)
        layout.addWidget(self._value_label)
        layout.addWidget(self._detail_label)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: str, detail: str = "") -> None:
        self._value_label.setText(value)
        self._detail_label.setText(detail)

    def set_palette_colors(self, palette: Palette) -> None:
        self._palette = palette
        color = QColor(palette.accent)
        self._value_label.setStyleSheet(f"color: {palette.text}; font-size: 22px; font-weight: 600;")
        self._value_label.setTextFormat(Qt.TextFormat.PlainText)
        del color

    def text(self) -> str:
        return self._value_label.text()


class UsageBar(QWidget):
    """A labelled progress bar that also shows the numeric percentage."""

    def __init__(
        self,
        caption: str,
        palette: Palette | None = None,
        maximum: float = 100.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._maximum = max(0.001, maximum)
        self._palette = palette
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._caption = QLabel(caption)
        self._value = QLabel("--")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._caption)
        header.addStretch(1)
        header.addWidget(self._value)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)

        layout.addLayout(header)
        layout.addWidget(self._bar)

    def set_palette_colors(self, palette: Palette) -> None:
        self._palette = palette
        chunk = palette.accent
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {chunk}; border-radius: 4px; }}"
        )

    def set_value(self, value: float | None, text: str | None = None) -> None:
        if value is None:
            self._bar.setValue(0)
            self._value.setText("n/a")
            return
        fraction = max(0.0, min(1.0, value / self._maximum))
        self._bar.setValue(int(fraction * 1000))
        self._value.setText(text if text is not None else f"{value:.0f}%")

    def set_unknown(self, reason: str) -> None:
        self._bar.setValue(0)
        self._value.setText(reason)

    def value_text(self) -> str:
        return self._value.text()


class KeyValueGrid(QWidget):
    """A two-column label/value grid used by the AI and info panels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(1, 1)
        self._rows: dict[str, QLabel] = {}

    def set(self, key: str, value: str) -> None:
        label = self._rows.get(key)
        if label is None:
            caption = QLabel(key)
            caption.setObjectName("MetricMuted")
            label = QLabel(value)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setWordWrap(True)
            row = self._grid.rowCount()
            self._grid.addWidget(caption, row, 0)
            self._grid.addWidget(label, row, 1)
            self._rows[key] = label
            return
        label.setText(value)

    def get(self, key: str) -> str:
        label = self._rows.get(key)
        return label.text() if label is not None else ""

    def keys(self) -> Iterable[str]:
        return self._rows.keys()

    def clear(self) -> None:
        for label in self._rows.values():
            label.setText("--")


class Banner(QLabel):
    """A static notice line, used for privacy and model-quality messages."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Banner")
        self.setWordWrap(True)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        super().setText(text)


def elide(items: Sequence[str], limit: int = 6) -> str:
    """Join items, adding an ellipsis marker when the list is long."""
    if len(items) <= limit:
        return ", ".join(items)
    remaining = len(items) - limit
    return ", ".join(items[:limit]) + f" (+{remaining} more)"
