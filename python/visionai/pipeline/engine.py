"""Threaded pipeline engine.

Three concerns run independently so a slow model never stalls the interface:

* the **capture** thread reads the device and publishes the newest frame only,
  dropping stale ones;
* the **inference** worker consumes frames at the configured inference FPS;
* the **UI** thread polls :meth:`PipelineEngine.latest` on a Qt timer and never
  performs inference itself.

Consumers therefore always get the freshest completed result rather than a
queue, and back-pressure shows up as dropped frames instead of growing latency.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visionai.ai.errors import CameraError, ModelError, VisionAIError
from visionai.ai.inference import InferencePipeline, PipelineResult
from visionai.ai.models.registry import LoadReport, ModelRegistry
from visionai.camera.camera_manager import CameraManager
from visionai.camera.frame_source import FrameSource
from visionai.config.paths import PackagePaths
from visionai.config.settings import Settings
from visionai.pipeline.stats import PipelineStats

LOG = logging.getLogger(__name__)


class PipelineState(enum.Enum):
    """Lifecycle of the engine, used to drive the UI status line."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineStatus:
    """Everything the UI needs to describe the engine without touching it."""

    state: PipelineState = PipelineState.IDLE
    message: str = ""
    started_at: float = 0.0
    inference_fps: float = 0.0
    capture_fps: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    last_error: str = ""
    source_name: str = ""

    @property
    def uptime(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0

    def describe(self) -> str:
        if self.state is PipelineState.RUNNING:
            return f"Running - {self.capture_fps:.0f} FPS capture, {self.inference_fps:.0f} FPS inference"
        return self.message or self.state.value.title()


@dataclass
class _Worker:
    """Bookkeeping for the inference thread."""

    thread: threading.Thread | None = None
    stop: threading.Event = field(default_factory=threading.Event)


class PipelineEngine:
    """Owns capture, inference and the latest result."""

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry | None = None,
        source: FrameSource | None = None,
        pipeline: InferencePipeline | None = None,
        autostart_models: bool = True,
    ) -> None:
        self.settings = settings.validate()
        self.registry = registry or ModelRegistry(
            PackagePaths.discover().face_models, PackagePaths.discover().emotion_models
        )
        self.pipeline = pipeline
        self._owns_pipeline = pipeline is None
        self._source = source
        self._owns_source = source is None
        self._autostart_models = autostart_models
        self._detector_report: LoadReport | None = None
        self._classifier_report: LoadReport | None = None

        self._worker = _Worker()
        self._state = PipelineState.IDLE
        self._status = EngineStatus()
        self._lock = threading.Lock()
        self._latest: PipelineResult | None = None
        self._listeners: list[Callable[[PipelineResult], None]] = []
        self._closed = False
        self._pending_frame: np.ndarray | None = None
        self._pending_seq = -1

        if self._autostart_models and self.pipeline is None:
            self.load_models()

    @property
    def source(self) -> FrameSource | None:
        return self._source

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._state is PipelineState.RUNNING

    def load_models(self) -> tuple[LoadReport | None, LoadReport | None]:
        """Build and load the detector and classifier, with fallbacks applied."""
        if self.pipeline is not None:
            return self._detector_report, self._classifier_report
        detector, detector_report = self.registry.build_detector(self.settings)
        classifier, classifier_report = self.registry.build_classifier(self.settings)
        self._detector_report = detector_report
        self._classifier_report = classifier_report
        self.pipeline = InferencePipeline(
            settings=self.settings,
            registry=self.registry,
            detector=detector,
            classifier=classifier,
            detector_report=detector_report,
            classifier_report=classifier_report,
        )
        self.pipeline.stats = PipelineStats(inference_fps=self.settings.pipeline.inference_fps)
        LOG.info(
            "models ready: detector=%s classifier=%s",
            detector_report.describe(),
            classifier_report.describe(),
        )
        return detector_report, classifier_report

    def open_source(self, source: FrameSource | str | None = None) -> FrameSource:
        """Acquire the frame source, defaulting to the configured camera.

        ``source`` may be a :class:`FrameSource` or a CLI-style spec string
        such as ``"0"``, ``"/dev/video0"``, ``"synthetic"`` or a video path.
        """
        from visionai.camera.camera_manager import open_source as build

        resolved: FrameSource
        if isinstance(source, str) and source:
            resolved = build(source, self.settings.camera)
            self._owns_source = True
        elif source is not None:
            resolved = source
            self._owns_source = True
        elif self._source is not None:
            resolved = self._source
        else:
            resolved = build("", self.settings.camera)
            self._owns_source = True
        self._source = resolved
        self._status.source_name = getattr(resolved, "display_name", "source")
        return resolved

    def start(self, source: FrameSource | str | None = None) -> None:
        """Open the source and run the inference worker."""
        if self._closed:
            raise VisionAIError("engine has been closed")
        if self._state is PipelineState.RUNNING:
            return
        self._set_state(PipelineState.STARTING, "Starting")

        try:
            if self.pipeline is None:
                self.load_models()
            self.open_source(source)
            assert self._source is not None
            self._source.open()
        except (VisionAIError, OSError) as exc:
            self._fail(exc)
            return

        if isinstance(self._source, CameraManager):
            self._source.start()

        self._status.started_at = time.time()
        self._worker.stop.clear()
        self._worker.thread = threading.Thread(
            target=self._run_inference, name="visionai-inference", daemon=True
        )
        self._worker.thread.start()
        self._set_state(PipelineState.RUNNING, "Running")
        self._status.message = self.pipeline.expression_quality_note if self.pipeline else ""

    def _run_inference(self) -> None:
        interval = 1.0 / max(0.5, self.settings.pipeline.inference_fps)
        next_run = time.perf_counter()
        while not self._worker.stop.is_set():
            now = time.perf_counter()
            if now < next_run:
                self._worker.stop.wait(min(next_run - now, interval))
                continue
            next_run = now + interval
            if self._state is PipelineState.PAUSED:
                continue
            frame = self._next_frame()
            if frame is None:
                continue
            self._process(frame)

    def _next_frame(self) -> np.ndarray | None:
        source = self._source
        if source is None:
            return None
        if isinstance(source, CameraManager):
            frame, sequence = source.latest()
            if frame is None or sequence == self._pending_seq:
                return None
            self._pending_seq = sequence
            return frame
        try:
            return source.read()
        except CameraError as exc:
            LOG.warning("source read failed: %s", exc)
            self._fail(exc)
            return None

    def _process(self, frame: np.ndarray) -> PipelineResult | None:
        if self.pipeline is None:
            return None
        try:
            result = self.pipeline.process(frame)
        except ModelError as exc:
            LOG.error("model error during inference: %s", exc)
            self._fail(exc)
            return None

        with self._lock:
            self._latest = result
        self._status.frames_processed += 1
        for listener in list(self._listeners):
            try:
                listener(result)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("pipeline listener failed: %s", exc)
        return result

    def pause(self) -> None:
        """Stop consuming frames and release the camera, keeping the worker alive."""
        if self._state is not PipelineState.RUNNING:
            return
        if self._source is not None:
            try:
                # Join the capture thread before touching the device; see stop().
                if isinstance(self._source, CameraManager):
                    self._source.stop()
                else:
                    self._source.release()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("error pausing source: %s", exc)
        self._set_state(PipelineState.PAUSED, "Paused")

    def resume(self) -> None:
        if self._state is PipelineState.PAUSED:
            try:
                if self._source is not None:
                    self._source.open()
                    if isinstance(self._source, CameraManager):
                        self._source.start()
                self._set_state(PipelineState.RUNNING, "Running")
            except VisionAIError as exc:
                self._fail(exc)

    def toggle_pause(self) -> None:
        if self._state is PipelineState.PAUSED:
            self.resume()
        else:
            self.pause()

    def stop(self) -> None:
        """Stop the worker and release the camera. Safe to call repeatedly."""
        if self._state is PipelineState.IDLE and self._worker.thread is None:
            return
        self._set_state(PipelineState.STOPPING, "Stopping")
        self._worker.stop.set()
        thread, self._worker.thread = self._worker.thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

        # Order matters: a camera's capture thread must be joined before the
        # device is released. cv2.VideoCapture is not thread-safe, so closing it
        # while the capture thread is inside read() is a use-after-free that
        # segfaults the process.
        if self._source is not None:
            try:
                if isinstance(self._source, CameraManager):
                    self._source.stop()
                else:
                    self._source.release()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("error releasing source: %s", exc)
        self._set_state(PipelineState.IDLE, "Stopped")

    def latest(self) -> PipelineResult | None:
        """The most recent completed result, or ``None`` before the first frame."""
        with self._lock:
            return self._latest

    def wait_for_result(self, timeout: float = 5.0) -> PipelineResult | None:
        """Block until a result exists or the timeout expires (used by the CLI)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.latest()
            if result is not None:
                return result
            if self._state is PipelineState.ERROR:
                return self.latest()
            time.sleep(0.01)
        return self.latest()

    def add_listener(self, listener: Callable[[PipelineResult], None]) -> None:
        self._listeners.append(listener)

    def refresh_stats(self) -> None:
        if self.pipeline is None or self._source is None:
            return
        self._status.inference_fps = self.pipeline.stats.observed_fps()
        if isinstance(self._source, CameraManager):
            self._status.capture_fps = self._source.actual_fps
            self._status.frames_dropped = self._source.stats.frames_dropped
        self._status.last_error = self._status.last_error or (
            self.pipeline.detector_stage.last_error or self.pipeline.classifier_stage.last_error
        )

    def _set_state(self, state: PipelineState, message: str) -> None:
        self._state = state
        self._status.state = state
        self._status.message = message
        LOG.debug("engine state -> %s (%s)", state.value, message)

    def _fail(self, exc: BaseException) -> None:
        self._status.last_error = f"{type(exc).__name__}: {exc}"
        self._set_state(PipelineState.ERROR, self._status.last_error)
        LOG.error("pipeline error: %s", exc)

    def describe(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "status": self._status.describe(),
            "source": self._source.describe() if self._source is not None else None,
            "detector_report": self._detector_report.describe() if self._detector_report else None,
            "classifier_report": (
                self._classifier_report.describe() if self._classifier_report else None
            ),
            "pipeline": self.pipeline.describe() if self.pipeline else None,
            "listeners": len(self._listeners),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        if self._owns_pipeline and self.pipeline is not None:
            self.pipeline.close()

    def __enter__(self) -> PipelineEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
