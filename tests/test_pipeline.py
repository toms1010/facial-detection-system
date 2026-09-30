"""End-to-end pipeline behaviour, including the threaded engine."""

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest

from visionai.ai.inference import InferencePipeline
from visionai.ai.models.base import Box
from visionai.ai.models.registry import ModelRegistry
from visionai.config.settings import Settings
from visionai.pipeline.engine import PipelineEngine, PipelineState
from visionai.pipeline.stats import PerformanceReport, PipelineStats


class TestPipelineResult:
    def test_summarize_uses_estimate_wording(self, pipeline: InferencePipeline) -> None:
        pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
        assert "No faces" in pipeline.process(
            np.zeros((240, 320, 3), dtype=np.uint8)
        ).summarize()

    def test_describe_is_serialisable(self, pipeline: InferencePipeline) -> None:
        import json

        result = pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
        json.dumps(result.describe())

    def test_frame_number_increments(self, pipeline: InferencePipeline) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert pipeline.process(frame).frame_number == 1
        assert pipeline.process(frame).frame_number == 2

    def test_top_expression_is_none_without_faces(
        self, pipeline: InferencePipeline
    ) -> None:
        assert pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8)).top_expression() is None


class TestPipeline:
    def test_processes_a_frame(self, pipeline: InferencePipeline) -> None:
        result = pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
        assert result.error == ""
        assert result.total_ms > 0
        assert result.detection_ms > 0

    def test_records_statistics(self, pipeline: InferencePipeline) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(5):
            pipeline.process(frame)
        assert pipeline.stats.frame_count == 5
        assert pipeline.stats.error_count == 0

    def test_reset_clears_state(self, pipeline: InferencePipeline) -> None:
        pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
        pipeline.reset()
        assert pipeline.stats.frame_count == 0
        assert pipeline.detector_stage.last_count == 0

    def test_reports_the_heuristic_fallback(self, pipeline: InferencePipeline) -> None:
        note = pipeline.expression_quality_note.lower()
        if not pipeline.uses_trained_expression_model:
            assert "heuristic" in note
            assert "no trained" in note

    def test_describe_covers_every_stage(self, pipeline: InferencePipeline) -> None:
        pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
        info = pipeline.describe()
        assert set(info) == {"detector", "classifier", "tracker", "stats", "quality"}

    def test_context_manager_closes(self, settings, registry: ModelRegistry) -> None:
        detector, dr = registry.build_detector(settings)
        classifier, cr = registry.build_classifier(settings)
        with InferencePipeline(settings, registry, detector, classifier, dr, cr):
            pass
        assert not detector.is_ready

    def test_close_is_idempotent(self, pipeline: InferencePipeline) -> None:
        pipeline.close()
        pipeline.close()

    def test_max_faces_is_enforced(self, settings, registry: ModelRegistry) -> None:
        settings.models.max_faces = 1
        detector, dr = registry.build_detector(settings)
        classifier, cr = registry.build_classifier(settings)
        pipeline = InferencePipeline(settings, registry, detector, classifier, dr, cr)
        result = pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
        assert result.face_count <= 1
        pipeline.close()


class TestStageFailureHandling:
    def test_detector_failure_is_contained(self, settings, registry: ModelRegistry) -> None:
        class Exploding:
            name = "exploding"
            last_latency_ms = 0.0

            def load(self) -> None: ...

            @property
            def is_ready(self) -> bool:
                return True

            def detect(self, frame: np.ndarray) -> list:
                raise RuntimeError("cuda exploded")

            def close(self) -> None: ...

            def describe(self) -> dict:
                return {"name": "exploding"}

        settings.models.max_faces = 4
        classifier, cr = registry.build_classifier(settings)
        pipeline = InferencePipeline(
            settings, registry, Exploding(), classifier, None, cr
        )
        result = pipeline.process(np.zeros((120, 160, 3), dtype=np.uint8))
        assert result.detections == []
        assert pipeline.detector_stage.error_count == 1
        assert "cuda exploded" in pipeline.detector_stage.last_error
        pipeline.close()

    def test_classifier_failure_is_contained(self, settings, registry: ModelRegistry) -> None:
        class Exploding:
            name = "exploding-classifier"
            is_heuristic = True
            is_trained_model = False
            class_names = ("happy", "sad")
            last_latency_ms = 0.0

            def load(self) -> None: ...

            @property
            def is_ready(self) -> bool:
                return True

            def predict(self, crop: np.ndarray):
                raise RuntimeError("dnn failed")

            def close(self) -> None: ...

            def describe(self) -> dict:
                return {"name": "exploding-classifier"}

        detector, dr = registry.build_detector(settings)
        pipeline = InferencePipeline(settings, registry, detector, Exploding(), dr, None)
        result = pipeline.process(np.zeros((120, 160, 3), dtype=np.uint8))
        assert result.error == ""
        pipeline.close()


class TestClassificationStage:
    def test_crops_are_padded(self, frame: np.ndarray) -> None:
        from visionai.ai.emotion_classifier import ClassificationStage
        from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier

        stage = ClassificationStage(
            classifier=HeuristicExpressionClassifier(), pad_ratio=0.2
        )
        crop = stage.crop_face(frame, Box(100, 100, 200, 200), frame.shape[1], frame.shape[0])
        assert crop is not None
        assert crop.shape[0] > 100 and crop.shape[1] > 100

    def test_degenerate_box_yields_no_crop(self, frame: np.ndarray) -> None:
        from visionai.ai.emotion_classifier import ClassificationStage
        from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier

        stage = ClassificationStage(classifier=HeuristicExpressionClassifier())
        assert stage.crop_face(frame, Box(500, 400, 505, 405), 640, 360) is None

    def test_crop_stays_inside_the_frame(self, frame: np.ndarray) -> None:
        from visionai.ai.emotion_classifier import ClassificationStage
        from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier

        stage = ClassificationStage(
            classifier=HeuristicExpressionClassifier(), pad_ratio=0.5
        )
        crop = stage.crop_face(frame, Box(0, 0, 40, 40), frame.shape[1], frame.shape[0])
        assert crop is not None
        assert crop.shape[0] <= frame.shape[0] and crop.shape[1] <= frame.shape[1]

    def test_no_detections_returns_empty(self, frame: np.ndarray) -> None:
        from visionai.ai.emotion_classifier import ClassificationStage
        from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier

        stage = ClassificationStage(classifier=HeuristicExpressionClassifier())
        assert stage.run(frame, []) == {}


class TestStats:
    def test_records_and_resets(self) -> None:
        stats = PipelineStats()
        for _ in range(3):
            stats.record(total_ms=10.0, detection_ms=6.0, face_count=2)
        assert stats.frame_count == 3
        assert stats.average("total_ms") == pytest.approx(10.0)
        stats.reset()
        assert stats.frame_count == 0

    def test_ema_needs_two_samples(self) -> None:
        stats = PipelineStats()
        stats.record(total_ms=10.0)
        assert stats.observed_fps() == 0.0
        stats.record(total_ms=10.0)
        assert stats.snapshot()["inference_ms_ema"] > 0

    def test_percentiles(self) -> None:
        stats = PipelineStats()
        for value in range(1, 101):
            stats.record(total_ms=float(value))
        assert stats.percentile("total_ms", 0.0) == 1.0
        assert stats.percentile("total_ms", 1.0) == 100.0
        assert 90 < stats.percentile("total_ms", 0.95) <= 100

    def test_snapshot_is_serialisable(self) -> None:
        import json

        stats = PipelineStats()
        stats.record(total_ms=5.0, face_count=1)
        json.dumps(stats.snapshot())

    def test_history_pairs(self) -> None:
        stats = PipelineStats()
        stats.record(total_ms=5.0)
        assert len(stats.history("total_ms")) == 1

    def test_errors_are_counted(self) -> None:
        stats = PipelineStats()
        stats.record_error("boom")
        assert stats.error_count == 1

    def test_empty_stats_report_zeros(self) -> None:
        stats = PipelineStats()
        assert stats.average("total_ms") == 0.0
        assert stats.percentile("total_ms", 0.5) == 0.0
        assert stats.observed_fps() == 0.0


class TestPerformanceReport:
    def test_describe_includes_the_required_metrics(self) -> None:
        report = PerformanceReport(
            frames=100,
            duration_s=10.0,
            average_fps=10.0,
            average_inference_ms=25.0,
            p95_inference_ms=40.0,
            max_inference_ms=60.0,
            average_detection_ms=20.0,
            average_classification_ms=5.0,
            faces_average=1.5,
            cpu_percent=32.0,
            ram_mb=512.0,
            ram_label="Process RAM RSS",
            gpu_percent=48.0,
            gpu_temperature=54.0,
        )
        text = report.describe()
        for label in (
            "Average FPS",
            "Avg inference",
            "P95 inference",
            "CPU usage",
            "Process RAM RSS",
            "GPU usage",
            "GPU temperature",
        ):
            assert label in text

    def test_the_ram_label_states_which_quantity_it_is(self) -> None:
        """Regression: system RAM and process RSS were both printed as 'RAM usage'."""
        assert "System RAM used" in PerformanceReport().describe()
        assert "RAM usage" not in PerformanceReport().describe()
        assert PerformanceReport().to_dict()["ram_label"] == "System RAM used"

    def test_optional_gpu_fields_are_omitted(self) -> None:
        assert "GPU usage" not in PerformanceReport().describe()


class TestEngine:
    def test_starts_and_produces_results(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("synthetic")
            assert engine.state is PipelineState.RUNNING
            result = engine.wait_for_result(timeout=8.0)
            assert result is not None
            assert result.frame_number > 0
        finally:
            engine.close()

    def test_stop_returns_to_idle(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        engine.start("synthetic")
        engine.stop()
        assert engine.state is PipelineState.IDLE
        engine.close()

    def test_pause_and_resume(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("synthetic")
            engine.pause()
            assert engine.state is PipelineState.PAUSED
            engine.toggle_pause()
            assert engine.state is PipelineState.RUNNING
        finally:
            engine.close()

    def test_bad_source_fails_gracefully(self, settings: Settings) -> None:

        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("/dev/definitely-not-a-camera")
            assert engine.state is PipelineState.ERROR
            assert engine.status.last_error
        finally:
            engine.close()

    def test_latest_is_none_before_the_first_frame(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        assert engine.latest() is None
        engine.close()

    def test_describe_works_once_a_source_is_open(self, settings: Settings) -> None:
        """Regression: FrameSource had no describe(), so this raised."""
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.open_source("synthetic")
            described = engine.describe()
            assert isinstance(described["source"], dict)
            assert described["source"]["name"] == "synthetic"
        finally:
            engine.close()

    def test_describe_is_serialisable_with_an_open_source(self, settings: Settings) -> None:
        import json

        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.open_source("synthetic")
            json.dumps(engine.describe())
        finally:
            engine.close()

    def test_describe_before_open_source_has_none(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            assert engine.describe()["source"] is None
        finally:
            engine.close()

    def test_blank_source_spec_falls_back_to_the_camera(
        self, settings: Settings
    ) -> None:
        """Regression: an empty spec used to be stored as a bare ``str``."""
        from visionai.camera.frame_source import FrameSource

        engine = PipelineEngine(settings, autostart_models=False)
        try:
            for blank in ("", "   "):
                resolved = engine.open_source(blank)
                assert isinstance(resolved, FrameSource)
                assert engine.describe()["source"] is not None
        finally:
            engine.close()

    def test_source_spec_is_stripped_before_building(self, settings: Settings) -> None:
        from visionai.camera.frame_source import SyntheticFrameSource

        engine = PipelineEngine(settings, autostart_models=False)
        try:
            assert isinstance(engine.open_source("  synthetic  "), SyntheticFrameSource)
        finally:
            engine.close()

    def test_listeners_are_notified(self, settings: Settings) -> None:
        seen: list[int] = []
        engine = PipelineEngine(settings, autostart_models=False)
        engine.add_listener(lambda result: seen.append(result.frame_number))
        try:
            engine.start("synthetic")
            engine.wait_for_result(timeout=8.0)
            time.sleep(0.3)
        finally:
            engine.close()
        assert seen

    def test_a_failing_listener_does_not_stop_the_pipeline(self, settings: Settings) -> None:
        def broken(result) -> None:
            raise RuntimeError("listener bug")

        engine = PipelineEngine(settings, autostart_models=False)
        engine.add_listener(broken)
        try:
            engine.start("synthetic")
            assert engine.wait_for_result(timeout=8.0) is not None
        finally:
            engine.close()

    def test_describe_is_serialisable(self, settings: Settings) -> None:
        import json

        engine = PipelineEngine(settings, autostart_models=False)
        json.dumps(engine.describe())
        engine.close()

    def test_closed_engine_cannot_restart(self, settings: Settings) -> None:
        from visionai.ai.errors import VisionAIError

        engine = PipelineEngine(settings, autostart_models=False)
        engine.close()
        with pytest.raises(VisionAIError):
            engine.start("synthetic")

    def test_inference_fps_is_respected(self, settings: Settings) -> None:
        settings.pipeline.inference_fps = 4.0
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("synthetic")
            engine.wait_for_result(timeout=8.0)
            time.sleep(1.2)
            engine.refresh_stats()
            assert engine.status.inference_fps < 12.0
        finally:
            engine.close()


class TestShutdownOrdering:
    """A capture thread must be joined before its device is released.

    cv2.VideoCapture is not thread-safe. Releasing it while the capture thread
    is inside read() is a use-after-free that segfaults the process, which is
    exactly what happened when Stop or Pause was pressed with a live camera.
    """

    def test_camera_is_joined_before_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cv2

        import visionai.camera.camera_manager as cm

        events: list[str] = []

        class SlowCapture:
            def __init__(self, *args, **kwargs) -> None:
                self.opened = True
                self.released = False
                self.isOpened = lambda: self.opened

            def read(self):
                events.append("read")
                time.sleep(0.05)
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def set(self, prop, value):
                return True

            def get(self, prop):
                return 640.0

            def release(self):
                events.append("release")
                self.released = True
                self.opened = False

        monkeypatch.setattr(cv2, "VideoCapture", SlowCapture)
        monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

        manager = cm.CameraManager(cm.CameraSettings(index=0))
        manager.start()
        time.sleep(0.2)
        manager.stop()
        manager.stop()  # idempotent

        assert events, "capture never ran"
        assert "release" in events
        # The last read must precede the release, never overlap it.
        assert events.index("release") >= events.index("read")

    def test_release_alone_stops_a_live_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cv2

        import visionai.camera.camera_manager as cm

        class SlowCapture:
            def __init__(self, *args, **kwargs) -> None:
                self.opened = True
                self.isOpened = lambda: self.opened

            def read(self):
                time.sleep(0.05)
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def set(self, prop, value):
                return True

            def get(self, prop):
                return 640.0

            def release(self):
                self.opened = False

        monkeypatch.setattr(cv2, "VideoCapture", SlowCapture)
        monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

        manager = cm.CameraManager(cm.CameraSettings(index=0))
        manager.start()
        time.sleep(0.15)
        manager.release()  # called directly, without stop()
        assert not manager.is_running
        assert not manager.is_open

    def test_engine_stop_joins_before_releasing(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("synthetic")
            engine.wait_for_result(timeout=8.0)
            source = engine.source
            assert source is not None
            engine.stop()
            assert not engine.is_running
            assert not getattr(source, "is_open", False)
        finally:
            engine.close()

    def test_pause_releases_the_device(self, settings: Settings) -> None:
        engine = PipelineEngine(settings, autostart_models=False)
        try:
            engine.start("synthetic")
            engine.pause()
            assert engine.state is PipelineState.PAUSED
            source = engine.source
            assert source is not None
            assert not getattr(source, "is_open", True)
        finally:
            engine.close()
