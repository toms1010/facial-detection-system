"""YOLO face detection via the Ultralytics runtime.

YOLO is used strictly as a *detector* here. It localises faces; it does not
classify expressions.

Two weight families are supported:

``face`` weights
    A dedicated face detector (for example ``yolov8n-face.pt``) whose boxes are
    used directly.

``person`` weights
    A generic COCO model such as ``yolo11n.pt`` that only knows the ``person``
    class. Because no face model is bundled with this repository, the detector
    can derive an approximate face region from the upper part of the person box.
    That approximation is off by default in strict mode and is always reported
    in the UI as an estimate.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from visionai.ai.errors import DeviceError, ModelMissingError
from visionai.ai.models.base import Box, FaceDetection, FaceDetector, validate_frame
from visionai.ai.models.devices import resolve_torch_device

LOG = logging.getLogger(__name__)

FACE_CLASS_NAMES = frozenset({"face", "faces", "head", "human face"})
PERSON_CLASS_NAMES = frozenset({"person", "people", "human", "man", "woman"})

#: Fraction of the person box height, measured from the top, where a face sits.
DEFAULT_FACE_REGION = (0.05, 0.52)


class YoloFaceDetector(FaceDetector):
    """Ultralytics YOLO wrapper that returns face boxes only."""

    name = "yolo"
    display_name = "YOLO (Ultralytics)"
    requires_weights = True
    supports_batching = True

    def __init__(
        self,
        weights: str | os.PathLike[str] | None = None,
        confidence: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        half: bool = False,
        allowed_classes: Sequence[str] | None = None,
        derive_face_from_person: bool | None = None,
        face_region: tuple[float, float] = DEFAULT_FACE_REGION,
    ) -> None:
        self.weights = str(weights) if weights else ""
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.iou = float(max(0.0, min(1.0, iou)))
        self.imgsz = int(imgsz)
        self.device_request = device
        self.half = bool(half)
        self.allowed_classes = tuple(allowed_classes) if allowed_classes else None
        self.derive_face_from_person = derive_face_from_person
        self.face_region = face_region
        self._model: Any = None
        self._class_names: dict[int, str] = {}
        self._device = "cpu"
        self._last_latency_ms = 0.0
        self._using_person_box = False

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.weights:
            raise ModelMissingError(
                "YOLO detector needs weights; set models.detector_weights to a "
                ".pt file or choose the Haar detector"
            )
        path = Path(self.weights).expanduser()
        if not path.is_file() and not _looks_like_registry_name(self.weights):
            raise ModelMissingError(f"YOLO weights not found: {path}")
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except Exception as exc:  # noqa: BLE001
            raise ModelMissingError(
                "ultralytics is not installed; run `pip install linux-ai-vision[ai]`"
            ) from exc

        choice = resolve_torch_device(self.device_request)
        if choice.is_gpu:
            self.half = self.half and choice.kind == "cuda"
        else:
            self.half = False
        try:
            self._model = YOLO(str(path) if path.is_file() else self.weights)
            self._model.to(choice.device)
        except DeviceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelMissingError(f"could not load YOLO weights {self.weights}: {exc}") from exc

        self._device = choice.device
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            self._class_names = {int(k): str(v).lower() for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self._class_names = {i: str(n).lower() for i, n in enumerate(names)}
        self._using_person_box = not self._has_face_class()
        if self._using_person_box and self.derive_face_from_person is None:
            self.derive_face_from_person = True
            LOG.info(
                "YOLO weights expose no 'face' class; estimating the face region from "
                "the upper part of each 'person' box"
            )
        LOG.info("YOLO detector ready on %s (classes=%s)", choice.device, self._class_names)

    def _has_face_class(self) -> bool:
        if self.allowed_classes:
            return any(c.lower() in FACE_CLASS_NAMES for c in self.allowed_classes)
        return any(name in FACE_CLASS_NAMES for name in self._class_names.values())

    def _class_allowed(self, class_id: int, class_name: str) -> bool:
        if self.allowed_classes:
            return class_name in {c.lower() for c in self.allowed_classes}
        if class_name in FACE_CLASS_NAMES:
            return True
        return class_name in PERSON_CLASS_NAMES and bool(self.derive_face_from_person)

    def _score_for(self, class_name: str) -> float:
        if class_name in FACE_CLASS_NAMES:
            return 1.0
        return 0.85

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def uses_person_box_approximation(self) -> bool:
        return self._using_person_box

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if self._model is None:
            self.load()
        image = validate_frame(frame, "frame")
        start = time.perf_counter()
        try:
            results = self._model.predict(
                image,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self._device,
                half=self.half,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"YOLO inference failed on {self._device}: {exc}") from exc

        detections = self._decode(results, image.shape[1], image.shape[0])
        detections.sort(key=lambda d: d.box.area, reverse=True)
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        return detections

    def _decode(self, results: Any, frame_width: int, frame_height: int) -> list[FaceDetection]:
        detections: list[FaceDetection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = np.asarray(getattr(boxes, "xyxy", []), dtype=np.float32).reshape(-1, 4)
            confs = np.asarray(getattr(boxes, "conf", []), dtype=np.float32).reshape(-1)
            clss = np.asarray(getattr(boxes, "cls", []), dtype=np.int32).reshape(-1)
            for index in range(xyxy.shape[0]):
                class_id = int(clss[index]) if index < len(clss) else 0
                class_name = self._class_names.get(class_id, "")
                if not self._class_allowed(class_id, class_name):
                    continue
                raw_score = float(confs[index]) if index < len(confs) else 0.0
                x1, y1, x2, y2 = (float(v) for v in xyxy[index])
                box = Box(x1, y1, x2, y2).clamp(frame_width, frame_height)
                if box.width < 8 or box.height < 8:
                    continue
                if class_name in PERSON_CLASS_NAMES and self.derive_face_from_person:
                    box = self._person_to_face(box, frame_width, frame_height)
                detections.append(
                    FaceDetection(
                        box=box,
                        score=raw_score * self._score_for(class_name),
                        source_class=class_name or "face",
                    )
                )
        return detections

    def _person_to_face(self, box: Box, frame_width: int, frame_height: int) -> Box:
        top, bottom = self.face_region
        expected = max(8.0, box.height * (bottom - top))
        centre_x, _ = box.center
        half = expected * 0.5
        face = Box(centre_x - half, box.y1, centre_x + half, box.y1 + expected)
        return face.clamp(frame_width, frame_height)

    def close(self) -> None:
        self._model = None

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "weights": self.weights or None,
                "device": self._device,
                "imgsz": self.imgsz,
                "classes": self._class_names,
                "person_box_approximation": self._using_person_box,
            }
        )
        return info


def _looks_like_registry_name(name: str) -> bool:
    bare = name.strip().lower().removesuffix(".pt")
    if not bare or "/" in bare or "\\" in bare:
        return False
    return bare.startswith("yolo") or bare.endswith(("n", "s", "m", "l", "x"))
