"""Webcam capture with a dedicated thread.

Capture runs on its own thread and publishes only the newest frame, so a slow
consumer drops stale frames instead of building a backlog. That is what keeps
the UI responsive when inference is slower than the camera.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visionai.ai.errors import (
    CameraError,
    CameraPermissionError,
    CameraUnavailableError,
    FrameError,
    InvalidCameraError,
)
from visionai.camera.frame_source import FrameSource, VideoFileSource
from visionai.config.settings import CameraSettings

LOG = logging.getLogger(__name__)

PERMISSION_HINTS = (
    "permission denied",
    "operation not permitted",
    "busy",
    "resource busy",
)


@dataclass
class CameraDevice:
    """A capture device discovered on this machine."""

    index: int
    name: str
    path: str | None = None
    is_default: bool = False

    def describe(self) -> str:
        location = f" ({self.path})" if self.path else ""
        return f"[{self.index}] {self.name}{location}"


@dataclass
class CaptureStats:
    """Counters describing the capture thread."""

    frames_captured: int = 0
    frames_dropped: int = 0
    read_errors: int = 0
    capture_fps: float = 0.0
    last_frame_time: float = 0.0
    last_error: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "read_errors": self.read_errors,
            "capture_fps": round(self.capture_fps, 2),
            "last_error": self.last_error or None,
        }


def enumerate_cameras(max_index: int = 10) -> list[CameraDevice]:
    """List V4L2 devices under ``/dev/video*`` without opening them."""
    devices: list[CameraDevice] = []
    video_dir = Path("/dev")
    if not video_dir.is_dir():
        return devices
    entries = sorted(
        (p for p in video_dir.glob("video*")),
        key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 1 << 30,
    )
    for position, entry in enumerate(entries):
        if not entry.exists():
            continue
        index = int(entry.name[5:]) if entry.name[5:].isdigit() else position
        devices.append(
            CameraDevice(
                index=index,
                name=_device_name(entry),
                path=str(entry),
                is_default=position == 0,
            )
        )
        if len(devices) >= max_index:
            break
    return devices


def _device_name(path: Path) -> str:
    name_file = Path("/sys/class/video4linux") / path.name / "name"
    try:
        text = name_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    return "video capture device"


def _looks_like_permission_problem(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in PERMISSION_HINTS)


class CameraManager(FrameSource):
    """Opens a capture device and reads frames on a background thread."""

    name = "camera"
    display_name = "Webcam"

    def __init__(self, settings: CameraSettings, backend: int | None | str = None) -> None:
        self.settings = settings.validate()
        self._backend = backend
        self._capture: Any = None
        self._source: FrameSource | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._sequence = 0
        self.stats = CaptureStats()
        self._actual_size = (0, 0)
        self._is_open = False
        self._capture_start = 0.0
        self._fps_ema = 0.0

    def open(self) -> None:
        if self._is_open:
            return
        import cv2

        target = self.settings.capture_target
        if isinstance(target, str) and not Path(target).exists():
            raise InvalidCameraError(f"camera device path does not exist: {target}")
        if isinstance(target, int) and not _camera_index_may_exist(target):
            raise InvalidCameraError(
                f"camera index {target} does not exist; "
                f"available devices: {[d.describe() for d in enumerate_cameras()] or 'none'}"
            )

        capture = self._open_capture(cv2, target)
        if capture is None:
            available = [d.describe() for d in enumerate_cameras()]
            raise CameraUnavailableError(
                f"could not open camera {target!r}. Detected devices: "
                f"{available or 'none'}. Check that the device is not in use by "
                "another application and that you are in the 'video' group."
            )

        self._capture = capture
        self._is_open = True
        self._apply_format(cv2)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.settings.width
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.settings.height
        self._actual_size = (width, height)
        self._capture_start = time.perf_counter()
        self._stop.clear()
        LOG.info(
            "camera %s open at %dx%d requested_fps=%d backend=%s",
            target,
            width,
            height,
            self.settings.fps,
            self.settings.backend,
        )

    def _open_capture(self, cv2_module: Any, target: Any) -> Any:
        candidates: list[Any] = []
        if isinstance(self._backend, int):
            candidates.append(self._backend)
        if self._backend is None:
            configured = self.settings.backend
            if configured != "auto":
                candidates.append(cv2_module.CAP_V4L2 if configured == "v4l2" else cv2_module.CAP_GSTREAMER)
            else:
                candidates.extend([cv2_module.CAP_V4L2, cv2_module.CAP_ANY])
        else:
            candidates.append(self._backend)

        last_error = ""
        for backend in candidates:
            capture = cv2_module.VideoCapture(target, backend)
            if capture.isOpened():
                return capture
            capture.release()
            last_error = f"backend {backend}"
            if _looks_like_permission_problem(last_error):
                break

        if last_error and _looks_like_permission_problem(last_error):
            raise CameraPermissionError(
                f"permission denied opening camera {target!r}; add your user to the "
                "'video' group or install a udev rule for the device"
            )
        return None

    def _apply_format(self, cv2_module: Any) -> None:
        assert self._capture is not None
        settings = self.settings
        for prop, value in (
            (cv2_module.CAP_PROP_FRAME_WIDTH, settings.width),
            (cv2_module.CAP_PROP_FRAME_HEIGHT, settings.height),
            (cv2_module.CAP_PROP_FPS, settings.fps),
        ):
            try:
                self._capture.set(prop, value)
            except Exception as exc:  # noqa: BLE001
                LOG.debug("could not set capture property %s: %s", prop, exc)
        if settings.fourcc:
            try:
                code = cv2_module.VideoWriter_fourcc(*settings.fourcc)
                if self._capture.set(cv2_module.CAP_PROP_FOURCC, code):
                    LOG.debug("requested FOURCC %s", settings.fourcc)
            except Exception as exc:  # noqa: BLE001
                LOG.debug("could not set FOURCC %s: %s", settings.fourcc, exc)

    def read(self) -> np.ndarray | None:
        """Blocking single read. Prefer :meth:`start` for live use."""
        if not self._is_open:
            self.open()
        return self._read_once()

    def _read_once(self) -> np.ndarray | None:
        import cv2

        if self._capture is None:
            return None
        try:
            ok, frame = self._capture.read()
        except Exception as exc:  # noqa: BLE001
            self.stats.read_errors += 1
            self.stats.last_error = f"read failed: {exc}"
            raise CameraError(f"camera read failed: {exc}") from exc
        if not ok or frame is None:
            self.stats.read_errors += 1
            self.stats.last_error = "camera returned no frame"
            raise FrameError("camera returned no frame; the device may have been unplugged")
        if self.settings.mirror:
            frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        self.stats.frames_captured += 1
        self.stats.last_frame_time = now
        if self._capture_start:
            self._fps_ema = 0.8 * self._fps_ema + 0.2 * (
                1.0 / max(1e-6, now - self._capture_start)
            )
            self._capture_start = now
            self.stats.capture_fps = self._fps_ema
        return frame

    def start(self) -> None:
        """Run capture on a daemon thread publishing the newest frame only."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        warmup = max(0, self.settings.warmup_frames)
        for _ in range(warmup):
            try:
                self._read_once()
            except CameraError as exc:
                LOG.debug("warmup frame failed: %s", exc)
                break
        while not self._stop.is_set():
            try:
                frame = self._read_once()
            except CameraError as exc:
                self.stats.last_error = str(exc)
                LOG.warning("capture loop error: %s", exc)
                time.sleep(0.05)
                continue
            with self._frame_lock:
                if self._frame is not None:
                    self.stats.frames_dropped += 1
                self._frame = frame
                self._frame_seq = self._sequence = self._sequence + 1

    def latest(self) -> tuple[np.ndarray | None, int]:
        """Return the newest frame and its sequence number without blocking."""
        with self._frame_lock:
            return self._frame, self._frame_seq

    def has_new_frame(self, last_seen: int) -> bool:
        with self._frame_lock:
            return self._frame_seq != last_seen

    def release(self) -> None:
        """Close the device.

        Safe to call at any time: if the capture thread is still running it is
        stopped first, because closing a cv2.VideoCapture while another thread
        is inside read() is a use-after-free.
        """
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._stop.set()
            thread.join(timeout=2.0)
            self._thread = None
        self._is_open = False
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("error releasing capture: %s", exc)
            self._capture = None
        if self._source is not None:
            self._source.release()
            self._source = None
        with self._frame_lock:
            self._frame = None

    def stop(self) -> None:
        """Stop the capture thread, then release the device."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self.release()

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._actual_size

    @property
    def actual_fps(self) -> float:
        return self.stats.capture_fps

    def describe(self) -> dict[str, Any]:
        width, height = self._actual_size
        return {
            "target": str(self.settings.capture_target),
            "open": self._is_open,
            "running": self.is_running,
            "actual_resolution": [width, height],
            "requested_resolution": list(self.settings.resolution),
            "requested_fps": self.settings.fps,
            "backend": self.settings.backend,
            "mirror": self.settings.mirror,
            **self.stats.describe(),
        }


def _camera_index_may_exist(index: int) -> bool:
    if index < 0:
        return False
    return Path(f"/dev/video{index}").exists()


def open_source(spec: str, camera_settings: CameraSettings) -> FrameSource:
    """Open the source described by a CLI ``--source`` string.

    ``0`` / ``/dev/video0`` select a camera, ``synthetic`` the generator,
    ``none`` a placeholder and anything else a video file.
    """
    from dataclasses import replace

    text = str(spec).strip()
    lowered = text.lower()

    if text.isdigit():
        return CameraManager(replace(camera_settings, index=int(text)))
    if text.startswith("/dev/"):
        return CameraManager(replace(camera_settings, device_path=text))
    if lowered in ("synthetic", "test", "demo"):
        from visionai.camera.frame_source import SyntheticFrameSource

        return SyntheticFrameSource(
            camera_settings.width, camera_settings.height, faces=2, fps=camera_settings.fps
        )
    if lowered in ("none", "null", "black"):
        from visionai.camera.frame_source import _BlackSource

        return _BlackSource(camera_settings.width, camera_settings.height)
    if Path(text).suffix:
        return VideoFileSource(text)
    return CameraManager(camera_settings)
