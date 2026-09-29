"""Frame sources.

A source is anything that yields BGR frames. The webcam is the real target, but
video files and a deterministic synthetic generator keep the app usable in CI,
on a headless box, and for tests that must never touch hardware.
"""

from __future__ import annotations

import abc
import logging
import math
import time
from pathlib import Path

import numpy as np

from visionai.ai.errors import CameraError, FrameError, InvalidCameraError

LOG = logging.getLogger(__name__)


class FrameSource(abc.ABC):
    """Produces frames on demand with no internal buffering."""

    name: str = "source"
    display_name: str = "Frame source"

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the underlying device. Must be idempotent."""

    @abc.abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next frame, or ``None`` when the source is exhausted."""

    @abc.abstractmethod
    def release(self) -> None:
        """Release resources. Must be idempotent."""

    @property
    def is_open(self) -> bool:
        return False

    @property
    def frame_size(self) -> tuple[int, int]:
        return (0, 0)

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class SyntheticFrameSource(FrameSource):
    """Generates moving synthetic faces so the pipeline runs without a camera.

    Not a model and not a benchmark substitute: the faces are crude geometric
    shapes. It exists to exercise capture, inference and UI code paths.
    """

    name = "synthetic"
    display_name = "Synthetic test pattern"

    def __init__(self, width: int = 960, height: int = 540, faces: int = 2, fps: float = 30.0):
        self.width = int(width)
        self.height = int(height)
        self.faces = int(max(0, faces))
        self.fps = float(max(1.0, fps))
        self._open = False
        self._started = 0.0
        self._frame_index = 0

    def open(self) -> None:
        if self._open:
            return
        self._started = time.perf_counter()
        self._frame_index = 0
        self._open = True

    def read(self) -> np.ndarray:
        if not self._open:
            self.open()
        elapsed = time.perf_counter() - self._started
        self._frame_index += 1
        return self.render(elapsed)

    def render(self, elapsed: float) -> np.ndarray:
        import cv2

        frame = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        gradient = np.linspace(35, 75, self.width, dtype=np.uint8)
        frame[:, :, 0] = gradient
        for index in range(self.faces):
            phase = elapsed * 1.1 + index * (math.tau / max(1, self.faces))
            size = int(min(self.width, self.height) * 0.18)
            cx = int(self.width * (0.5 + 0.32 * math.cos(phase)))
            cy = int(self.height * (0.5 + 0.20 * math.sin(phase * 0.8)))
            smile = 0.5 + 0.5 * math.sin(phase * 1.7)
            box = (cx - size // 2, cy - size // 2, size, size)
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + size, box[1] + size), (120, 90, 70), -1)
            eye_y = cy - size // 5
            eye_dx = size // 6
            for sign in (-1, 1):
                cv2.ellipse(
                    frame,
                    (cx + sign * eye_dx, eye_y),
                    (max(3, size // 16), max(2, int(size // 24) + int(3 * smile))),
                    0,
                    0,
                    360,
                    (250, 250, 250),
                    -1,
                )
            mouth_y = cy + size // 4
            mouth_w = int(size * (0.25 + 0.15 * smile))
            cv2.ellipse(
                frame,
                (cx, mouth_y - int(size * 0.06 * smile)),
                (mouth_w, max(3, int(size * (0.06 + 0.10 * smile)))),
                0,
                0,
                180,
                (60, 60, 160),
                3,
            )
        cv2.putText(
            frame,
            f"SYNTHETIC SOURCE - frame {self._frame_index}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.width, self.height


class VideoFileSource(FrameSource):
    """Reads frames from a video file, looping when configured to."""

    name = "file"
    display_name = "Video file"

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.loop = bool(loop)
        self._capture = None
        self._size = (0, 0)

    def open(self) -> None:
        import cv2

        if self._capture is not None:
            return
        if not self.path.is_file():
            raise InvalidCameraError(f"video file not found: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise CameraError(f"could not open video file {self.path}")
        self._capture = capture
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._size = (width, height)

    def read(self) -> np.ndarray | None:
        import cv2

        if self._capture is None:
            self.open()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            if not self.loop:
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise FrameError(f"video file {self.path} produced no frames")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size


class _BlackSource(FrameSource):
    """Placeholder used when capture is disabled but the pipeline must keep running."""

    name = "none"
    display_name = "Camera disabled"

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width, self.height = width, height
        self._open = False

    def open(self) -> None:
        self._open = True

    def read(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        import cv2

        cv2.putText(
            frame,
            "Camera disabled - press Enable Camera in Settings",
            (24, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.width, self.height
