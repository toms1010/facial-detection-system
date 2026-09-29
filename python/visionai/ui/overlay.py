"""Frame overlay renderer.

Draws bounding boxes, tracking IDs, expression labels, confidence bars and the
live metrics strip directly onto a frame with OpenCV. Keeping this independent
of Qt means the visual output can be asserted in tests without a display
server, and the same renderer backs both the desktop UI and the headless mode.

Every label is phrased as an estimate. The renderer never states a conclusion
about a person; see :mod:`visionai.ai.taxonomy`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from visionai.ai import taxonomy
from visionai.ai.emotion_classifier import UNKNOWN_LABEL
from visionai.ai.face_tracker import TrackedFace
from visionai.config.settings import UISettings
from visionai.ui.theme import Palette, get_palette

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.42
FONT_SCALE_LABEL = 0.52
FONT_THICKNESS = 1
LABEL_PADDING = 6
BAR_HEIGHT = 5
MIN_BOX_SIDE = 3


@dataclass
class OverlayMetrics:
    """Values shown in the metrics strip along the bottom of the frame."""

    fps: float = 0.0
    inference_ms: float = 0.0
    detection_ms: float = 0.0
    classification_ms: float = 0.0
    face_count: int = 0
    source: str = ""
    model: str = ""
    heuristic: bool = False

    def text(self) -> str:
        parts = [
            f"{self.fps:.0f} FPS",
            f"{self.inference_ms:.0f} ms",
            f"{self.faces_text(self.face_count)}",
        ]
        if self.model:
            marker = " (heuristic)" if self.heuristic else ""
            parts.append(f"{self.model}{marker}")
        return "   ".join(parts)

    @staticmethod
    def faces_text(count: int) -> str:
        return f"{count} face{'' if count == 1 else 's'}"


@dataclass
class RenderOptions:
    """Everything the renderer needs that is not the frame itself."""

    settings: UISettings = field(default_factory=UISettings)
    palette: Palette | None = None
    show_disclaimer: bool = True
    scale: float = 1.0

    def resolved_palette(self) -> Palette:
        return self.palette or get_palette(self.settings.theme)


def render_overlay(
    frame: np.ndarray,
    tracks: Sequence[TrackedFace] = (),
    options: RenderOptions | None = None,
    metrics: OverlayMetrics | None = None,
    label_accepted: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` with boxes, labels and metrics drawn on it.

    The input frame is never modified.
    """
    options = options or RenderOptions()
    palette = options.resolved_palette()
    canvas = _as_bgr(frame).copy()

    scale = max(0.1, options.scale)
    for track in tracks:
        _draw_track(canvas, track, options, palette, scale, label_accepted)

    if metrics is not None and options.settings.show_fps:
        _draw_metrics_strip(canvas, metrics, palette, scale)

    if options.show_disclaimer:
        _draw_disclaimer(canvas, palette, scale)

    return canvas


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    """Normalise any supported frame layout to a 3-channel BGR array."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        raise ValueError(f"unsupported frame shape {frame.shape}")
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _draw_track(
    canvas: np.ndarray,
    track: TrackedFace,
    options: RenderOptions,
    palette: Palette,
    scale: float,
    label_accepted: bool,
) -> None:
    settings = options.settings
    x1, y1, x2, y2 = track.box.clamp(canvas.shape[1], canvas.shape[0]).as_int()
    thickness = max(1, int(round(2 * scale)))
    color = taxonomy.bgr_for(track.label) if track.label != UNKNOWN_LABEL else (140, 140, 140)

    if settings.show_boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        corner = max(8, int(18 * scale))
        for cx, cy, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(canvas, (cx, cy), (cx + corner * dx, cy), color, thickness, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + corner * dy), color, thickness, cv2.LINE_AA)

    if not settings.show_labels:
        return

    lines: list[str] = []
    if settings.show_track_ids:
        lines.append(f"ID {track.track_id:02d}")
    label = taxonomy.title(track.label) if label_accepted else "Analyzing"
    line = f"{taxonomy.emoji_for(track.label)} {label}" if label_accepted else label
    if settings.show_confidence and label_accepted:
        line += f"  {track.confidence * 100:.0f}%"
    lines.append(line)
    if track.expression is not None and track.expression.is_heuristic:
        lines.append("heuristic, not a model prediction")

    _draw_label_block(canvas, lines, (x1, y1), color, palette, scale)

    if settings.show_confidence and label_accepted:
        _draw_confidence_bar(canvas, track, (x1, y1, x2, y2), color, scale)


def _draw_label_block(
    canvas: np.ndarray,
    lines: Sequence[str],
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    palette: Palette,
    scale: float,
) -> None:
    font_scale = FONT_SCALE_LABEL * scale
    padding = int(LABEL_PADDING * scale)
    line_height = int(cv2.getTextSize("Ag", FONT, font_scale, FONT_THICKNESS)[0][1] * 1.45)
    block_width = 0
    for text in lines:
        (text_width, _), _ = cv2.getTextSize(text, FONT, font_scale, FONT_THICKNESS)
        block_width = max(block_width, text_width)
    block_width += padding * 2
    block_height = line_height * len(lines) + padding

    x, y = anchor
    box_top = max(0, y - block_height)
    box_left = max(0, min(x, canvas.shape[1] - block_width))
    if box_left + block_width > canvas.shape[1]:
        box_left = max(0, canvas.shape[1] - block_width)

    cv2.rectangle(
        canvas,
        (box_left, box_top),
        (box_left + block_width, box_top + block_height),
        palette.overlay_backdrop,
        -1,
    )
    cv2.rectangle(
        canvas,
        (box_left, box_top),
        (box_left + block_width, box_top + block_height),
        color,
        max(1, int(round(scale))),
    )

    baseline = box_top + padding + line_height
    for text in lines:
        cv2.putText(
            canvas,
            text,
            (box_left + padding, baseline - int(line_height * 0.25)),
            FONT,
            font_scale,
            palette.overlay_text,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        baseline += line_height


def _draw_confidence_bar(
    canvas: np.ndarray,
    track: TrackedFace,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    x1, y1, x2, y2 = bounds
    width = max(1, x2 - x1)
    bar_y = min(canvas.shape[0] - int(BAR_HEIGHT * scale) - 1, y2 + int(2 * scale))
    bar_height = max(2, int(BAR_HEIGHT * scale))
    cv2.rectangle(
        canvas, (x1, bar_y), (x1 + width, bar_y + bar_height), (60, 60, 60), -1
    )
    filled = int(width * max(0.0, min(1.0, track.confidence)))
    if filled > 0:
        cv2.rectangle(canvas, (x1, bar_y), (x1 + filled, bar_y + bar_height), color, -1)


def _draw_metrics_strip(
    canvas: np.ndarray,
    metrics: OverlayMetrics,
    palette: Palette,
    scale: float,
) -> None:
    text = metrics.text()
    font_scale = FONT_SCALE_SMALL * scale
    thickness = max(1, int(round(scale)))
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    padding = int(10 * scale)
    height = text_height + baseline + padding * 2
    width = min(canvas.shape[1], text_width + padding * 2)
    top = canvas.shape[0] - height

    strip = canvas[top : top + height, 0:width]
    if strip.size:
        backdrop = np.full_like(strip, palette.overlay_backdrop)
        cv2.addWeighted(strip, 0.25, backdrop, 0.75, 0, dst=strip)
        cv2.putText(
            canvas,
            text,
            (padding, top + padding + text_height),
            FONT,
            font_scale,
            palette.overlay_text,
            thickness,
            cv2.LINE_AA,
        )


def _draw_disclaimer(canvas: np.ndarray, palette: Palette, scale: float) -> None:
    text = "AI-estimated expression - not a reading of a person's actual feelings"
    font_scale = FONT_SCALE_SMALL * 0.95 * scale
    thickness = max(1, int(round(scale)))
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    padding = int(8 * scale)
    top = max(0, canvas.shape[0] - int((text_height + baseline + padding * 2) * 1.9))
    left = max(0, (canvas.shape[1] - text_width - padding * 2) // 2)

    region = canvas[top : top + text_height + baseline + padding * 2, left : left + text_width + padding * 2]
    if region.size:
        backdrop = np.full_like(region, palette.overlay_backdrop)
        cv2.addWeighted(region, 0.2, backdrop, 0.8, 0, dst=region)
        cv2.putText(
            canvas,
            text,
            (left + padding, top + padding + text_height),
            FONT,
            font_scale,
            (150, 150, 150),
            thickness,
            cv2.LINE_AA,
        )


def blank_overlay(width: int = 640, height: int = 360, text: str = "") -> np.ndarray:
    """A placeholder frame used when no camera is active."""
    frame = np.full((height, width, 3), 18, dtype=np.uint8)
    if text:
        font_scale = max(0.4, min(1.0, width / 1400.0))
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, font_scale, 1)
        cv2.putText(
            frame,
            text,
            ((width - text_width) // 2, (height + text_height) // 2),
            FONT,
            font_scale,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    return frame


def overlay_timestamp() -> str:
    return time.strftime("%H:%M:%S")
