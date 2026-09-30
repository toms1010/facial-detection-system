"""Face detection backends: contracts, geometry and error handling."""

from __future__ import annotations

import numpy as np
import pytest

from visionai.ai.errors import FrameError, ModelLoadError, ModelMissingError
from visionai.ai.models.base import Box, FaceDetection, validate_frame
from visionai.ai.models.haar_detector import HaarFaceDetector, resolve_cascade_path
from visionai.ai.models.registry import ModelRegistry
from visionai.ai.models.yunet_detector import YuNetFaceDetector, yunet_available


def make_detection(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> FaceDetection:
    return FaceDetection(box=Box(x1, y1, x2, y2), score=score)


class TestBox:
    def test_geometry(self) -> None:
        box = Box(10, 20, 60, 100)
        assert box.width == 50
        assert box.height == 80
        assert box.area == 4000
        assert box.center == (35.0, 60.0)

    def test_degenerate_box_has_no_area(self) -> None:
        assert Box(10, 10, 10, 10).area == 0.0

    def test_iou_of_identical_boxes_is_one(self) -> None:
        box = Box(0, 0, 10, 10)
        assert box.iou(Box(0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self) -> None:
        assert Box(0, 0, 10, 10).iou(Box(50, 50, 60, 60)) == 0.0

    def test_iou_of_half_overlap(self) -> None:
        assert Box(0, 0, 10, 10).iou(Box(5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_iou_with_degenerate_box_is_zero(self) -> None:
        assert Box(0, 0, 10, 10).iou(Box(5, 5, 5, 5)) == 0.0

    def test_clamp_constrains_to_frame(self) -> None:
        clamped = Box(-20, -10, 900, 700).clamp(640, 480)
        assert clamped.x1 == 0 and clamped.y1 == 0
        assert clamped.x2 == 640 and clamped.y2 == 480

    def test_expanded_grows_by_ratio(self) -> None:
        expanded = Box(100, 100, 200, 200).expanded(0.5, 640, 480)
        assert (expanded.x1, expanded.y1, expanded.x2, expanded.y2) == (50, 50, 250, 250)

    def test_expanded_respects_frame_bounds(self) -> None:
        expanded = Box(0, 0, 20, 20).expanded(1.0, 100, 100)
        assert expanded.x1 == 0 and expanded.y1 == 0

    def test_scaled_scales_coordinates(self) -> None:
        assert Box(10, 10, 20, 20).scaled(2.0, 3.0).as_int() == (20, 30, 40, 60)

    def test_from_xywh(self) -> None:
        assert Box.from_xywh(10, 20, 30, 40).as_int() == (10, 20, 40, 60)

    def test_from_mapping(self) -> None:
        assert Box.from_any({"x1": 1, "y1": 2, "x2": 3, "y2": 4}).as_int() == (1, 2, 3, 4)

    def test_from_sequence(self) -> None:
        assert Box.from_any([1, 2, 3, 4]).as_int() == (1, 2, 3, 4)

    def test_from_unsupported_raises(self) -> None:
        with pytest.raises(ValueError):
            Box.from_any([1, 2, 3])

    def test_as_int_rounds(self) -> None:
        assert Box(1.4, 1.6, 2.5, 3.5).as_int() == (1, 2, 2, 4)


class TestValidateFrame:
    def test_accepts_colour(self, frame: np.ndarray) -> None:
        assert validate_frame(frame) is frame

    def test_accepts_grayscale(self, gray_frame: np.ndarray) -> None:
        assert validate_frame(gray_frame) is gray_frame

    @pytest.mark.parametrize(
        "bad",
        [None, np.array([]), "not an array", np.zeros((2, 2, 7)), np.zeros((2, 2, 2, 2))],
    )
    def test_rejects_unusable(self, bad: object) -> None:
        with pytest.raises(FrameError):
            validate_frame(bad)  # type: ignore[arg-type]


class TestFaceDetection:
    def test_score_is_clamped(self) -> None:
        assert make_detection(0, 0, 10, 10, score=5.0).score == 1.0
        assert make_detection(0, 0, 10, 10, score=-1.0).score == 0.0

    def test_box_is_coerced_from_any(self) -> None:
        detection = FaceDetection(box=[0, 0, 10, 10], score=0.5)  # type: ignore[arg-type]
        assert detection.box.as_int() == (0, 0, 10, 10)


@pytest.mark.skipif(not yunet_available(), reason="OpenCV has no FaceDetectorYN")
class TestYuNetDetector:
    def test_loads_and_reports_ready(self) -> None:
        detector = YuNetFaceDetector()
        detector.load()
        assert detector.is_ready
        detector.close()
        assert not detector.is_ready

    def test_load_is_idempotent(self) -> None:
        detector = YuNetFaceDetector()
        detector.load()
        path = detector._path
        detector.load()
        assert detector._path == path
        detector.close()

    def test_detect_returns_valid_detections(self, frame: np.ndarray) -> None:
        detector = YuNetFaceDetector(confidence=0.3)
        detections = detector.detect(frame)
        assert isinstance(detections, list)
        for detection in detections:
            assert isinstance(detection, FaceDetection)
            assert 0.0 <= detection.score <= 1.0
            assert detection.box.width > 0 and detection.box.height > 0
            assert detector.last_latency_ms > 0

    def test_detect_accepts_grayscale(self, gray_frame: np.ndarray) -> None:
        assert isinstance(YuNetFaceDetector().detect(gray_frame), list)

    def test_detect_rejects_empty_frame(self) -> None:
        with pytest.raises(FrameError):
            YuNetFaceDetector().detect(np.array([]))

    def test_detect_rejects_none(self) -> None:
        with pytest.raises(FrameError):
            YuNetFaceDetector().detect(None)  # type: ignore[arg-type]

    def test_missing_weights_raise(self) -> None:
        with pytest.raises(ModelMissingError):
            YuNetFaceDetector(weights="/nonexistent/yunet.onnx").load()

    def test_context_manager(self) -> None:
        with YuNetFaceDetector() as detector:
            assert detector.is_ready
        assert not detector.is_ready

    def test_describe_includes_metadata(self) -> None:
        detector = YuNetFaceDetector()
        info = detector.describe()
        assert info["name"] == "yunet"
        assert info["ready"] is False
        detector.load()
        assert detector.describe()["weights"]

    def test_detect_after_close_reloads_transparently(self, frame: np.ndarray) -> None:
        """A use after close() must reload rather than dereference None."""
        detector = YuNetFaceDetector(confidence=0.3)
        assert isinstance(detector.detect(frame), list)
        detector.close()
        assert not detector.is_ready
        assert isinstance(detector.detect(frame), list)
        assert detector.is_ready
        detector.close()

    def test_ensure_input_size_after_close_does_not_crash(self) -> None:
        detector = YuNetFaceDetector()
        detector.load()
        detector.close()
        detector._ensure_input_size(640, 480)
        assert detector.is_ready
        detector.close()


class TestYuNetDownscale:
    """A large frame must not be fed to the network at its native size.

    YuNet is trained at 320x320, so detecting on a 1280x720 frame costs roughly
    nine times the work for no gain. This was the largest single contributor to
    the application feeling slow.
    """

    def test_long_side_is_capped_at_the_input_size(self) -> None:
        detector = YuNetFaceDetector(confidence=0.3)
        detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert max(detector._last_size) <= 320
        detector.close()

    def test_aspect_ratio_is_preserved(self) -> None:
        detector = YuNetFaceDetector(confidence=0.3)
        detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        width, height = detector._last_size
        assert width / height == pytest.approx(1280 / 720, rel=0.05)
        detector.close()

    def test_a_small_frame_is_passed_through_untouched(self) -> None:
        detector = YuNetFaceDetector(confidence=0.3)
        detector.detect(np.zeros((120, 160, 3), dtype=np.uint8))
        assert detector._last_size == (160, 120)
        detector.close()

    def test_boxes_stay_inside_the_frame(self) -> None:
        detector = YuNetFaceDetector(confidence=0.3)
        detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        for detection in detections:
            assert detection.box.x1 >= -1.0
            assert detection.box.y1 >= -1.0
            assert detection.box.x2 <= 1281.0
            assert detection.box.y2 <= 721.0
        detector.close()

    def test_landmarks_are_mapped_back_to_frame_space(self) -> None:
        """Landmarks are rescaled with the boxes, not left in network space."""
        detector = YuNetFaceDetector(confidence=0.3)
        detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        for detection in detections:
            landmarks = detection.landmarks
            if landmarks is None:
                continue
            assert landmarks.shape == (5, 2)
            assert landmarks[:, 0].max() <= 1281.0
            assert landmarks[:, 1].max() <= 721.0
        detector.close()

    def test_downscale_is_faster_than_detecting_at_native_size(self) -> None:
        detector = YuNetFaceDetector(confidence=0.3)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        detector.detect(frame)
        fast = detector.last_latency_ms
        detector.close()

        native = YuNetFaceDetector(confidence=0.3, input_size=1280)
        native.detect(frame)
        assert fast < native.last_latency_ms
        native.close()

    @pytest.mark.parametrize(
        ("frame_w", "frame_h"),
        [(1280, 720), (1920, 1080), (640, 480), (320, 240)],
    )
    def test_network_coordinates_are_mapped_back_into_frame_space(
        self, monkeypatch: pytest.MonkeyPatch, frame_w: int, frame_h: int
    ) -> None:
        """A box in network pixels must land where it belongs in the frame.

        The scale has to be applied in the right direction and to the
        landmarks as well as the box. Getting it backwards still returns boxes
        of plausible size, so it passes every shape assertion while silently
        breaking tracking and the overlay.
        """
        import visionai.ai.models.yunet_detector as module

        class StubNetwork:
            """Emits one face covering a known fraction of the network input.

            Uses OpenCV's real row layout:
            ``[x, y, w, h, rx, ry, lx, ly, nx, ny, mlx, mly, mrx, mry, score]``.
            """

            size: tuple[int, int] | None = None

            def setInputSize(self, size) -> None:
                StubNetwork.size = tuple(size)

            def detect(self, image):
                w, h = StubNetwork.size
                # box across the middle 50% of the width, middle 25% of the height
                row = [w * 0.25, h * 0.5, w * 0.5, h * 0.25]
                # ten landmark coordinates, the first at 30% of the width
                row += [w * 0.3, h * 0.3] * 5
                row += [0.87]
                return None, np.array([row], dtype=np.float32)

        monkeypatch.setattr(
            module.cv2.FaceDetectorYN, "create", staticmethod(lambda *a, **k: StubNetwork())
        )

        detector = YuNetFaceDetector(confidence=0.1)
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        box = detections[0].box
        assert box.x1 == pytest.approx(frame_w * 0.25, abs=1.0)
        assert box.y1 == pytest.approx(frame_h * 0.5, abs=1.0)
        assert box.x2 == pytest.approx(frame_w * 0.75, abs=1.0)
        assert box.y2 == pytest.approx(frame_h * 0.75, abs=1.0)

        # The first landmark sits at 30% of the network width, so it must land
        # at 30% of the frame width.
        landmarks = detections[0].landmarks
        assert landmarks is not None
        assert landmarks[0, 0] == pytest.approx(frame_w * 0.3, abs=1.0)
        assert landmarks[0, 1] == pytest.approx(frame_h * 0.3, abs=1.0)

    def test_confidence_is_read_from_the_last_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The score is column 14; column 4 is a landmark coordinate.

        Reading column 4 yields a value like 86.3, which the dataclass clamps to
        1.0, so every detection was reported at 100% confidence.
        """
        import visionai.ai.models.yunet_detector as module

        class StubNetwork:
            def setInputSize(self, size) -> None: ...

            def detect(self, image):
                row = [100.0, 100.0, 60.0, 60.0]
                row += [125.0, 4.0] * 5  # landmarks, one value deliberately large
                row += [0.62]
                return None, np.array([row], dtype=np.float32)

        monkeypatch.setattr(
            module.cv2.FaceDetectorYN, "create", staticmethod(lambda *a, **k: StubNetwork())
        )

        detections = YuNetFaceDetector(confidence=0.1).detect(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        assert len(detections) == 1
        assert detections[0].score == pytest.approx(0.62)

    def test_landmarks_land_inside_their_own_box(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Landmarks are five points in the face, so they must sit within the box.

        Reading one column too far shifts every pair, which puts the points
        outside the box entirely.
        """
        import visionai.ai.models.yunet_detector as module

        class StubNetwork:
            def setInputSize(self, size) -> None: ...

            def detect(self, image):
                x, y, w, h = 100.0, 100.0, 120.0, 150.0
                # eyes, nose and mouth corners, all within the box
                row = [x, y, w, h, 125, 140, 195, 140, 160, 165, 140, 205, 180, 205]
                row += [0.9]
                return None, np.array([row], dtype=np.float32)

        monkeypatch.setattr(
            module.cv2.FaceDetectorYN, "create", staticmethod(lambda *a, **k: StubNetwork())
        )

        detection = YuNetFaceDetector(confidence=0.1).detect(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )[0]
        landmarks = detection.landmarks
        assert landmarks is not None
        assert landmarks.shape == (5, 2)
        box = detection.box
        assert (landmarks[:, 0] >= box.x1).all() and (landmarks[:, 0] <= box.x2).all()
        assert (landmarks[:, 1] >= box.y1).all() and (landmarks[:, 1] <= box.y2).all()


class TestCascadeResolution:
    def test_resolves_a_bundled_cascade(self) -> None:
        assert resolve_cascade_path().is_file()

    def test_explicit_name_is_honoured(self) -> None:
        assert resolve_cascade_path("haarcascade_frontalface_default.xml").is_file()

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ModelMissingError):
            resolve_cascade_path("haarcascade_nothing.xml")

    def test_model_directory_contains_detection_weights(self, paths) -> None:
        assert (paths.face_models / "face_detection_yunet_2023mar.onnx").is_file()


class TestHaarDetector:
    def test_reports_a_clear_error_on_opencv5(self) -> None:
        import cv2

        detector = HaarFaceDetector()
        if hasattr(cv2, "CascadeClassifier"):
            detector.load()
            assert detector.is_ready
            detector.close()
        else:
            with pytest.raises((AttributeError, ModelLoadError, ModelMissingError)):
                detector.load()

    def test_unknown_cascade_raises(self) -> None:
        with pytest.raises(ModelMissingError):
            HaarFaceDetector(cascade="nope.xml").load()


class TestRegistry:
    def test_lists_backends(self, registry: ModelRegistry) -> None:
        assert "yunet" in registry.available_detectors()
        assert "heuristic" in registry.available_classifiers()

    def test_hides_haar_when_opencv_lacks_it(self, registry: ModelRegistry) -> None:
        import cv2

        expected = hasattr(cv2, "CascadeClassifier")
        assert ("haar" in registry.available_detectors()) is expected

    def test_builds_a_working_default_detector(self, settings, registry: ModelRegistry) -> None:
        detector, report = registry.build_detector(settings)
        assert detector.is_ready
        assert report.active in registry.available_detectors()
        detector.close()

    def test_falls_back_when_requested_backend_is_unusable(
        self, settings, registry: ModelRegistry
    ) -> None:
        settings.models.detector = "onnx"
        detector, report = registry.build_detector(settings)
        assert detector.is_ready
        assert report.active != "onnx"
        assert report.degraded is True
        assert report.reason
        detector.close()

    def test_discovers_model_files(self, registry: ModelRegistry) -> None:
        assert registry.available_face_models()
        for found in registry.available_emotion_models():
            assert found.suffix in {".onnx", ".pt", ".pth"}
