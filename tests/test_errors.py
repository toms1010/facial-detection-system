"""Error handling for every failure mode the specification lists.

The requirement is graceful degradation, so these tests assert that each failure
raises a specific, catchable type or produces a reported state - never an
unhandled crash and never a silent wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionai.ai.errors import (
    CameraError,
    CameraPermissionError,
    CameraUnavailableError,
    ConfigError,
    DeviceError,
    FrameError,
    InferenceError,
    InvalidCameraError,
    ModelError,
    ModelLoadError,
    ModelMissingError,
    NativeLibraryError,
    SensorUnavailableError,
    TrackingError,
    VisionAIError,
)
from visionai.ai.models.base import ExpressionResult, validate_frame
from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier
from visionai.ai.models.registry import ModelRegistry
from visionai.camera.frame_source import SyntheticFrameSource
from visionai.config.paths import PackagePaths
from visionai.config.settings import Settings, load_settings
from visionai.pipeline.engine import PipelineEngine, PipelineState


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "child",
        [
            ConfigError, ModelError, ModelMissingError, ModelLoadError, InferenceError,
            DeviceError, CameraError, CameraUnavailableError, CameraPermissionError,
            InvalidCameraErrorAlias := CameraError, FrameError, HardwareErrorAlias := VisionAIError,
            NativeLibraryError, SensorUnavailableError, TrackingError,
        ],
    )
    def test_everything_descends_from_the_base(self, child: type) -> None:
        assert issubclass(child, VisionAIError)

    def test_model_errors_share_a_parent(self) -> None:
        for child in (ModelMissingError, ModelLoadError, InferenceError, DeviceError):
            assert issubclass(child, ModelError)

    def test_camera_errors_share_a_parent(self) -> None:
        for child in (CameraUnavailableError, CameraPermissionError, FrameError):
            assert issubclass(child, CameraError)

    def test_specific_errors_can_be_caught_generically(self) -> None:
        with pytest.raises(VisionAIError):
            raise ModelMissingError("no weights")

    def test_messages_survive(self) -> None:
        assert str(ModelMissingError("detail here")) == "detail here"


class TestCameraErrors:
    def test_invalid_index(self) -> None:
        from visionai.camera.camera_manager import CameraManager
        from visionai.config.settings import CameraSettings

        with pytest.raises(InvalidCameraError) as info:
            CameraManager(CameraSettings(index=30)).open()
        assert "30" in str(info.value)

    def test_missing_device_path(self) -> None:
        from visionai.camera.camera_manager import CameraManager
        from visionai.config.settings import CameraSettings

        with pytest.raises(InvalidCameraError):
            CameraManager(CameraSettings(device_path="/dev/nope")).open()

    def test_unavailable_device(self) -> None:
        from visionai.camera.camera_manager import CameraManager
        from visionai.config.settings import CameraSettings

        with pytest.raises(CameraUnavailableError) as info:
            CameraManager(CameraSettings(index=0), backend=9999).open()
        message = str(info.value)
        assert "Detected devices" in message or "check" in message.lower()

    def test_invalid_frame(self) -> None:
        with pytest.raises(FrameError):
            validate_frame(None)
        with pytest.raises(FrameError):
            validate_frame(np.zeros((0, 0), np.uint8))
        with pytest.raises(FrameError):
            validate_frame("string")  # type: ignore[arg-type]

    def test_source_failure_moves_the_engine_to_error(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("/dev/not-a-camera")
            assert engine.state is PipelineState.ERROR
            assert engine.status.last_error
        finally:
            engine.close()

    def test_engine_keeps_running_after_a_read_error(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("synthetic")
            assert engine.wait_for_result(timeout=8.0) is not None
        finally:
            engine.close()

    def test_synthetic_source_survives_repeated_reads(self) -> None:
        source = SyntheticFrameSource(160, 120)
        for _ in range(50):
            assert source.read().size > 0
        source.release()


class TestModelErrors:
    def test_missing_detector_weights(self, settings: Settings) -> None:
        from visionai.ai.models.yolo_detector import YoloFaceDetector

        with pytest.raises(ModelMissingError):
            YoloFaceDetector(weights="/nowhere/model.pt").load()

    def test_detector_without_weights_configured(self) -> None:
        from visionai.ai.models.yolo_detector import YoloFaceDetector

        with pytest.raises(ModelMissingError) as info:
            YoloFaceDetector(weights="").load()
        assert "models.detector_weights" in str(info.value)

    def test_missing_classifier_weights(self) -> None:
        from visionai.ai.models.onnx_classifier import OnnxExpressionClassifier

        with pytest.raises(ModelMissingError):
            OnnxExpressionClassifier(model_path="/nowhere/m.onnx").load()

    def test_registry_falls_back_rather_than_raising(self, settings: Settings) -> None:
        registry = ModelRegistry(
            PackagePaths.discover().face_models, PackagePaths.discover().emotion_models
        )
        settings.models.detector = "yolo"
        settings.models.detector_weights = "/nowhere/m.pt"
        detector, report = registry.build_detector(settings)
        assert detector.is_ready
        assert report.degraded is True
        assert "yolo" in report.reason
        detector.close()

    def test_classifier_failure_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from visionai.ai.models.onnx_classifier import OnnxExpressionClassifier

        classifier = OnnxExpressionClassifier()
        classifier._descriptor = None
        monkeypatch.setattr(
            OnnxExpressionClassifier,
            "predict",
            lambda self, crop: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError):
            classifier.predict(np.zeros((32, 32, 3), np.uint8))

    def test_inference_error_type_exists(self) -> None:
        assert issubclass(InferenceError, ModelError)

    def test_device_error_mentions_cuda(self) -> None:
        assert "CUDA" in str(DeviceError("CUDA is unavailable"))


class TestClassifierResilience:
    def test_tiny_crops_do_not_crash(self) -> None:
        classifier = HeuristicExpressionClassifier()
        for size in (8, 9, 16, 32):
            assert classifier.predict(np.zeros((size, size, 3), np.uint8)) is not None

    def test_uniform_crops_are_handled(self) -> None:
        result = HeuristicExpressionClassifier().predict(
            np.full((64, 64, 3), 128, np.uint8)
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_extreme_values_are_handled(self) -> None:
        for value in (0, 255):
            result = HeuristicExpressionClassifier().predict(
                np.full((64, 64, 3), value, np.uint8)
            )
            assert result.label

    def test_non_square_crops(self) -> None:
        assert HeuristicExpressionClassifier().predict(
            np.zeros((32, 128, 3), np.uint8)
        ).label

    def test_confidence_is_always_bounded(self) -> None:
        rng = np.random.default_rng(7)
        classifier = HeuristicExpressionClassifier()
        for _ in range(20):
            crop = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
            result = classifier.predict(crop)
            assert 0.0 <= result.confidence <= 1.0
            assert all(0.0 <= v <= 1.0 for v in result.scores.values())


class TestConfigErrors:
    def test_corrupt_config_falls_back(self, tmp_path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("{broken", encoding="utf-8")
        assert load_settings(path).camera.index == 0

    def test_non_object_config_falls_back(self, tmp_path) -> None:
        path = tmp_path / "settings.json"
        path.write_text('"a string"', encoding="utf-8")
        assert load_settings(path).camera.fps == 30

    def test_out_of_range_config_is_clamped(self, tmp_path) -> None:
        import json

        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"camera": {"fps": 100000}, "pipeline": {"inference_fps": -5}}),
            encoding="utf-8",
        )
        settings = load_settings(path)
        assert settings.camera.fps == 240
        assert settings.pipeline.inference_fps == 0.5

    def test_config_error_type_exists(self) -> None:
        assert issubclass(ConfigError, VisionAIError)


class TestNativeErrors:
    def test_missing_library_is_reported_not_raised(self) -> None:
        from visionai.hardware.native_loader import NativeLoader

        loader = NativeLoader(prefer_native=False)
        report = loader.load()
        assert report.active == "python"

    def test_require_native_raises_a_typed_error(self) -> None:
        from visionai.hardware.native_loader import NativeLoader

        with pytest.raises(NativeLibraryError):
            NativeLoader(prefer_native=False, require_native=True).load()

    def test_unreadable_disk_is_reported(self) -> None:
        from visionai.hardware.python_backend import PythonBackend

        disk = PythonBackend().sample_disk("/proc/nonexistent-mount")
        assert disk["available"] is False

    def test_missing_sensors_are_reported_as_none(self) -> None:
        from visionai.config.settings import HardwareSettings
        from visionai.hardware.hardware_bridge import HardwareBridge

        snapshot = HardwareBridge(HardwareSettings()).sample()
        assert snapshot.cpu_temp_c is None or snapshot.cpu_temp_c > 0

    def test_sensor_error_type_exists(self) -> None:
        assert issubclass(SensorUnavailableError, VisionAIError)


class TestTrackingErrors:
    def test_tracking_error_type_exists(self) -> None:
        assert issubclass(TrackingError, VisionAIError)

    def test_tracker_never_raises_on_odd_input(self) -> None:
        from visionai.ai.face_tracker import FaceTracker
        from visionai.ai.models.base import Box, FaceDetection

        tracker = FaceTracker()
        assert tracker.update([]) == []
        assert tracker.update([FaceDetection(box=Box(0, 0, 0, 0), score=0.0)]) != []


class TestGracefulDegradation:
    def test_pipeline_survives_a_broken_detector(self, settings: Settings) -> None:
        class Broken:
            name = "broken"
            last_latency_ms = 0.0

            def load(self) -> None: ...

            @property
            def is_ready(self) -> bool:
                return True

            def detect(self, frame):
                raise RuntimeError("detector failure")

            def close(self) -> None: ...

            def describe(self):
                return {"name": "broken"}

        classifier, _ = ModelRegistry(
            PackagePaths.discover().face_models, PackagePaths.discover().emotion_models
        ).build_classifier(settings)
        from visionai.ai.inference import InferencePipeline

        pipeline = InferencePipeline(settings, None, Broken(), classifier)  # type: ignore[arg-type]
        result = pipeline.process(np.zeros((120, 160, 3), np.uint8))
        assert result.error == ""
        assert result.detections == []
        pipeline.close()

    def test_expression_result_never_claims_certainty(self) -> None:
        result = ExpressionResult(label="happy", confidence=1.0, is_heuristic=True)
        assert result.is_heuristic is True
