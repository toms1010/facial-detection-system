"""Qt video surface.

Displays a BGR frame as a scaled QImage, converting only when the frame
reference actually changes. Re-converting an unchanged frame on every timer
tick is the single easiest way to make an OpenCV + Qt UI stutter.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

LOG = logging.getLogger(__name__)

FORMAT_BGR888 = QImage.Format.Format_BGR888


class VideoWidget(QWidget):
    """Scales a BGR frame to the widget size while preserving aspect ratio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = "Camera is off"
        self._aspect = 16.0 / 9.0
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    def set_frame(self, frame: np.ndarray | None) -> bool:
        """Display ``frame``. Returns True when the content changed."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            return False

        contiguous = np.ascontiguousarray(frame[:, :, :3])
        height, width = contiguous.shape[:2]
        image = QImage(contiguous.data, width, height, 3 * width, FORMAT_BGR888).copy()
        self._aspect = width / max(1, height)
        self._pixmap = QPixmap.fromImage(image)
        self.update()
        return True

    def set_placeholder(self, message: str) -> None:
        self._placeholder = message
        self._pixmap = None
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self.update()

    def has_content(self) -> bool:
        return self._pixmap is not None

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pixmap is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
