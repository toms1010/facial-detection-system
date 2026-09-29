"""YuNet face detector (OpenCV DNN).

YuNet is the default detector because it is a small, accurate CNN that ships as
a 230 KB ONNX file and runs on CPU in a couple of milliseconds, which keeps the
whole application usable offline with no PyTorch install.

It returns face boxes only, including five extra landmark points per face.
The landmarks are kept on the detection because they make the crop alignment in
the classifier stage noticeably better on tilted faces.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from visionai.ai.errors import InferenceError, ModelMissingError
from visionai.ai.models.base import Box, FaceDetection, FaceDetector, validate_frame

LOG = logging.getLogger(__name__)

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
INT8_FILENAME = "face_detection_yunet_2023mar_int8.onnx"

#: YuNet was trained at 320x320; feeding larger frames costs time for no gain.
INPUT_SIZE = 320


def resolve_yunet_path(name: str | None = None) -> Path:
    """Locate the YuNet weights in the repository models directory."""
    from visionai.ai.models.haar_detector import _search_dirs

    if name:
        candidate = Path(name).expanduser()
        if candidate.is_file():
            return candidate
        for directory in _search_dirs():
            candidate = directory / Path(name).name
            if candidate.is_file():
                return candidate
        raise ModelMissingError(f"YuNet weights not found: {name}")

    for directory in _search_dirs():
        for filename in (MODEL_FILENAME, INT8_FILENAME):
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    raise ModelMissingError(
        f"YuNet weights not found. Place {MODEL_FILENAME} in models/face/ or run "
        "`python scripts/fetch_models.py`."
    )


def _quiet_opencv() -> None:
    """Silence OpenCV's C++ log, which warns on every DNN target request."""
    with contextlib.suppress(Exception):
        cv2.setLogLevel(2)
    with contextlib.suppress(Exception):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


def yunet_available() -> bool:
    """True when this OpenCV build exposes the YuNet API."""
    return hasattr(cv2, "FaceDetectorYN")


class YuNetFaceDetector(FaceDetector):
    """CNN face detector from OpenCV's DNN module."""

    name = "yunet"
    display_name = "YuNet (OpenCV DNN)"
    requires_weights = True

    def __init__(
        self,
        confidence: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        weights: str | None = None,
        input_size: int = INPUT_SIZE,
    ) -> None:
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.nms_threshold = float(max(0.0, min(1.0, nms_threshold)))
        self.top_k = int(max(1, top_k))
        self.weights = weights
        self.input_size = int(input_size)
        self._detector = None
        self._path: Path | None = None
        self._last_latency_ms = 0.0
        self._last_size: tuple[int, int] | None = None

    def load(self) -> None:
        if self._detector is not None:
            return
        if not yunet_available():
            raise ModelMissingError(
                "this OpenCV build does not expose FaceDetectorYN; install "
                "opencv-python>=4.8 or switch the detector to haar or yolo"
            )
        self._path = resolve_yunet_path(self.weights)
        _quiet_opencv()
        try:
            detector = cv2.FaceDetectorYN.create(
                str(self._path),
                "",
                (self.input_size, self.input_size),
                self.confidence,
                self.nms_threshold,
                self.top_k,
            )
        except cv2.error as exc:
            raise ModelMissingError(f"could not load YuNet weights {self._path}: {exc}") from exc
        self._detector = detector
        LOG.info("YuNet face detector ready using %s", self._path.name)

    @property
    def is_ready(self) -> bool:
        return self._detector is not None

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if self._detector is None:
            self.load()
        image = validate_frame(frame, "frame")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        start = time.perf_counter()

        height, width = image.shape[:2]
        self._ensure_input_size(width, height)

        try:
            _, faces = self._detector.detect(image)
        except cv2.error as exc:
            raise InferenceError(f"YuNet inference failed: {exc}") from exc

        detections: list[FaceDetection] = []
        if faces is not None:
            for face in np.asarray(faces).reshape(-1, 15):
                x, y, w, h = (float(v) for v in face[:4])
                score = float(face[4])
                box = Box(x, y, x + w, y + h).clamp(width, height)
                if box.width < 8 or box.height < 8:
                    continue
                landmarks = np.asarray(face[5:15], dtype=np.float32).reshape(5, 2)
                detection = FaceDetection(box=box, score=score, source_class="face")
                detection.landmarks = landmarks
                detections.append(detection)

        detections.sort(key=lambda d: d.box.area, reverse=True)
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        return detections

    def _ensure_input_size(self, width: int, height: int) -> None:
        """YuNet rejects frames whose size differs from the configured input."""
        size = (width, height)
        if size == self._last_size:
            return
        self._detector.setInputSize(size)
        self._last_size = size

    def close(self) -> None:
        self._detector = None
        self._last_size = None

    def describe(self) -> dict:
        info = super().describe()
        info["weights"] = self._path.name if self._path else None
        info["input_size"] = [self.input_size]
        return info
