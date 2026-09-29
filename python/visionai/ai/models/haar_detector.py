"""OpenCV Haar-cascade face detector.

This is the default detector because it needs no downloads and no GPU: the
cascade XML ships inside the OpenCV wheel. It is a reasonable baseline, though
far less accurate on profile views and crowded scenes than a YOLO face model.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from visionai.ai.errors import FrameError, ModelLoadError, ModelMissingError
from visionai.ai.models.base import Box, FaceDetection, FaceDetector, validate_frame

LOG = logging.getLogger(__name__)

CASCADE_NAMES: tuple[str, ...] = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
)


def _cascade_dir() -> Path:
    getter = getattr(cv2.data, "haarcascades", None)
    if getter:
        return Path(getter)
    return Path(cv2.__file__).resolve().parent / "data"


def _search_dirs() -> list[Path]:
    """Directories searched for cascade XMLs, most specific first.

    OpenCV 5.0 stopped shipping the cascade data files inside the wheel, so the
    repository's ``models/face`` directory is checked first: it is the only
    location that works on a fresh checkout with no downloads.
    """
    from visionai.config.paths import PackagePaths

    dirs = [PackagePaths.discover().face_models, _cascade_dir()]
    override = os.environ.get("VISIONAI_MODEL_DIR", "").strip()
    if override:
        dirs.insert(0, Path(override).expanduser())
    return [d for d in dirs if d.is_dir()]


def resolve_cascade_path(name: str | None = None) -> Path:
    """Locate a bundled cascade, preferring an explicit name or path."""
    if name:
        candidate = Path(name).expanduser()
        if candidate.is_file():
            return candidate
        for directory in _search_dirs():
            candidate = directory / Path(name).name
            if candidate.is_file():
                return candidate
        raise ModelMissingError(f"Haar cascade not found: {name}")

    for directory in _search_dirs():
        for filename in CASCADE_NAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    searched = ", ".join(str(d) for d in _search_dirs())
    raise ModelMissingError(
        f"no Haar cascade XML found. Searched: {searched or 'no directories exist'}. "
        "Install opencv-python (which ships the data files) or place a cascade XML "
        "in models/face/."
    )


class HaarFaceDetector(FaceDetector):
    """Viola-Jones style cascade detector from the OpenCV data files."""

    name = "haar"
    display_name = "OpenCV Haar cascade"

    def __init__(
        self,
        confidence: float = 0.5,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 40,
        cascade: str | None = None,
        max_side: int = 640,
    ) -> None:
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.scale_factor = float(max(1.01, scale_factor))
        self.min_neighbors = int(max(0, min_neighbors))
        self.min_size = int(max(16, min_size))
        self._cascade_name = cascade
        self._cascade: cv2.CascadeClassifier | None = None
        self._path: Path | None = None
        self._last_latency_ms = 0.0
        self._max_side = int(max_side)

    def load(self) -> None:
        if self._cascade is not None:
            return
        self._path = resolve_cascade_path(self._cascade_name)
        cascade = cv2.CascadeClassifier(str(self._path))
        if cascade.empty():
            raise ModelLoadError(f"Haar cascade at {self._path} failed to parse")
        self._cascade = cascade
        LOG.info("loaded Haar cascade %s", self._path.name)

    @property
    def is_ready(self) -> bool:
        return self._cascade is not None

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if self._cascade is None:
            self.load()
        image = validate_frame(frame, "frame")
        start = time.perf_counter()

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = 1.0
        if self._max_side and max(gray.shape[:2]) > self._max_side:
            scale = self._max_side / max(gray.shape[:2])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)

        try:
            raw = self._cascade.detectMultiScale(  # type: ignore[union-attr]
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(self.min_size, self.min_size),
            )
        except cv2.error as exc:
            raise FrameError(f"Haar detection failed: {exc}") from exc

        detections: list[FaceDetection] = []
        for x, y, w, h in np.asarray(raw).reshape(-1, 4):
            if w < 8 or h < 8:
                continue
            box = Box(float(x), float(y), float(x + w), float(y + h))
            if scale != 1.0:
                box = box.scaled(1.0 / scale, 1.0 / scale)
            height, width = image.shape[:2]
            box = box.clamp(width, height)
            if box.width < 8 or box.height < 8:
                continue
            detections.append(FaceDetection(box=box, score=self.confidence, source_class="face"))

        detections.sort(key=lambda d: d.box.area, reverse=True)
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        return detections

    def close(self) -> None:
        self._cascade = None

    def describe(self) -> dict:
        info = super().describe()
        info["cascade"] = self._path.name if self._path else None
        return info
