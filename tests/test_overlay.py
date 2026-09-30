"""Frame overlay rendering, exercised without a display server."""

from __future__ import annotations

import numpy as np
import pytest

from visionai.ai.emotion_classifier import UNKNOWN_LABEL
from visionai.ai.face_tracker import FaceTracker
from visionai.ai.models.base import Box, ExpressionResult, FaceDetection
from visionai.config.settings import UISettings
from visionai.ui.overlay import (
    OverlayMetrics,
    RenderOptions,
    blank_overlay,
    render_overlay,
)
from visionai.ui.theme import DARK, LIGHT, get_palette, stylesheet


def make_track(track_id: int = 1, label: str = "happy", confidence: float = 0.91, heuristic: bool = False):
    tracker = FaceTracker()
    result = ExpressionResult.from_scores(
        {label: confidence, "neutral": 1 - confidence}, model="test"
    )
    result.is_heuristic = heuristic
    detection = FaceDetection(box=Box(100, 80, 260, 280), score=0.9)
    tracks = tracker.update([detection], {0: result})
    tracks[0].track_id = track_id
    return tracks[0]


class TestRenderOverlay:
    def test_does_not_modify_the_input(self, frame: np.ndarray) -> None:
        original = frame.copy()
        render_overlay(frame, [make_track()])
        assert np.array_equal(frame, original)

    def test_preserves_shape_and_dtype(self, frame: np.ndarray) -> None:
        rendered = render_overlay(frame, [make_track()])
        assert rendered.shape == frame.shape
        assert rendered.dtype == frame.dtype

    def test_draws_something(self, frame: np.ndarray) -> None:
        assert not np.array_equal(render_overlay(frame, [make_track()]), frame)

    def test_handles_grayscale_input(self, gray_frame: np.ndarray) -> None:
        rendered = render_overlay(gray_frame, [])
        assert rendered.shape[:2] == gray_frame.shape[:2]
        assert rendered.shape[2] == 3

    def test_normalises_bgra_to_bgr(self, frame: np.ndarray) -> None:
        bgra = np.dstack([frame, np.full(frame.shape[:2], 255, np.uint8)])
        assert render_overlay(bgra, []).shape[2] == 3

    def test_normalises_single_channel(self, frame: np.ndarray) -> None:
        assert render_overlay(frame[:, :, :1], []).shape[2] == 3

    def test_rejects_impossible_shapes(self) -> None:
        with pytest.raises(ValueError):
            render_overlay(np.zeros((2, 2, 2, 2), dtype=np.uint8), [])

    def test_with_no_tracks_only_draws_the_disclaimer(self, frame: np.ndarray) -> None:
        rendered = render_overlay(frame, [], options=RenderOptions(show_disclaimer=False))
        assert np.array_equal(rendered, frame)

    def test_disclaimer_can_be_disabled(self, frame: np.ndarray) -> None:
        with_notice = render_overlay(frame, [], options=RenderOptions(show_disclaimer=True))
        without = render_overlay(frame, [], options=RenderOptions(show_disclaimer=False))
        assert not np.array_equal(with_notice, without)

    def test_multiple_tracks(self, frame: np.ndarray) -> None:
        tracks = [make_track(1, "happy"), make_track(2, "sad"), make_track(3, "surprise")]
        rendered = render_overlay(frame, tracks)
        assert rendered.shape == frame.shape

    def test_track_outside_the_frame_is_clamped(self) -> None:
        small = np.zeros((120, 160, 3), dtype=np.uint8)
        tracker = FaceTracker()
        detection = FaceDetection(box=Box(1000, 1000, 2000, 2000), score=0.9)
        tracks = tracker.update([detection], {0: ExpressionResult(label="happy", confidence=0.9)})
        assert render_overlay(small, tracks).shape == small.shape

    def test_label_acceptance_controls_the_text(self, frame: np.ndarray) -> None:
        accepted = render_overlay(frame, [make_track()], label_accepted=True)
        pending = render_overlay(frame, [make_track()], label_accepted=False)
        assert not np.array_equal(accepted, pending)

    def test_unknown_label_is_drawn(self, frame: np.ndarray) -> None:
        tracker = FaceTracker()
        result = ExpressionResult(label=UNKNOWN_LABEL, confidence=0.0, scores={})
        detection = FaceDetection(box=Box(50, 50, 150, 150), score=0.5)
        tracks = tracker.update([detection], {0: result})
        assert render_overlay(frame, tracks).shape == frame.shape

    def test_smoothing_heuristic_note(self, frame: np.ndarray) -> None:
        plain = render_overlay(frame, [make_track()])
        heuristic = render_overlay(frame, [make_track(heuristic=True)])
        assert not np.array_equal(plain, heuristic)


class TestDisplaySettings:
    @pytest.mark.parametrize(
        "field", ["show_boxes", "show_labels", "show_confidence", "show_track_ids"]
    )
    def test_each_toggle_changes_the_output(self, frame: np.ndarray, field: str) -> None:
        enabled = UISettings(**{field: True})  # type: ignore[arg-type]
        disabled = UISettings(**{field: False})  # type: ignore[arg-type]
        with_on = render_overlay(frame, [make_track()], RenderOptions(settings=enabled))
        with_off = render_overlay(frame, [make_track()], RenderOptions(settings=disabled))
        assert not np.array_equal(with_on, with_off)

    def test_fps_overlay_toggle(self, frame: np.ndarray) -> None:
        metrics = OverlayMetrics(fps=30, inference_ms=12, face_count=1)
        shown = render_overlay(frame, [], RenderOptions(settings=UISettings(show_fps=True)), metrics)
        hidden = render_overlay(frame, [], RenderOptions(settings=UISettings(show_fps=False)), metrics)
        assert not np.array_equal(shown, hidden)

    def test_light_and_dark_palettes_differ(self, frame: np.ndarray) -> None:
        dark = render_overlay(frame, [make_track()], RenderOptions(palette=DARK))
        light = render_overlay(frame, [make_track()], RenderOptions(palette=LIGHT))
        assert not np.array_equal(dark, light)

    def test_theme_setting_selects_the_palette(self) -> None:
        assert RenderOptions(settings=UISettings(theme="light")).resolved_palette() is LIGHT
        assert RenderOptions(settings=UISettings(theme="dark")).resolved_palette() is DARK

    def test_unknown_theme_falls_back_to_dark(self) -> None:
        assert get_palette("neon") is DARK


class TestOverlayMetrics:
    def test_text_contains_the_key_numbers(self) -> None:
        text = OverlayMetrics(fps=30.0, inference_ms=24.0, face_count=2, model="onnx").text()
        assert "30 FPS" in text
        assert "24 ms" in text
        assert "2 faces" in text
        assert "onnx" in text

    def test_singular_face(self) -> None:
        assert "1 face" in OverlayMetrics(face_count=1).text()

    def test_heuristic_is_marked(self) -> None:
        assert "heuristic" in OverlayMetrics(model="heuristic", heuristic=True).text()

    def test_model_is_omitted_when_empty(self) -> None:
        assert "()" not in OverlayMetrics().text()


class TestPlaceholder:
    def test_creates_a_frame(self) -> None:
        assert blank_overlay(320, 240).shape == (240, 320, 3)

    def test_optional_text(self) -> None:
        assert not np.array_equal(blank_overlay(320, 240), blank_overlay(320, 240, "hello"))


class TestStylesheet:
    def test_generates_valid_qss(self) -> None:
        text = stylesheet(DARK)
        assert "QWidget" in text
        assert DARK.background in text
        assert "{" in text and "}" in text

    def test_braces_are_balanced(self) -> None:
        assert stylesheet(LIGHT).count("{") == stylesheet(LIGHT).count("}")

    def test_palette_exposes_a_dict(self) -> None:
        assert DARK.as_dict()["name"] == "dark"
        assert DARK.qcolor("accent") == DARK.accent
