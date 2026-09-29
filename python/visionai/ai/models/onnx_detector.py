"""Generic ONNX YOLO-style face detector.

Works with exported Ultralytics models whose head already carries NMS, as well
as raw ``(1, 4+nc, 8400)`` style outputs. Keeping this backend dependency-light
means a machine without PyTorch can still run a real YOLO face model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionai.ai.errors import InferenceError, ModelLoadError, ModelMissingError
from visionai.ai.models.base import Box, FaceDetection, FaceDetector, validate_frame
from visionai.ai.models.devices import build_onnx_session, resolve_onnx_provider

LOG = logging.getLogger(__name__)


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_h, pad_w = (size - new_h) // 2, (size - new_w) // 2
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas, scale, float(pad_w)


class OnnxYoloFaceDetector(FaceDetector):
    """ONNX Runtime wrapper around an exported YOLO face detector."""

    name = "onnx"
    display_name = "YOLO (ONNX Runtime)"
    requires_weights = True

    def __init__(
        self,
        weights: str | None = None,
        confidence: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        allowed_classes: tuple[int, ...] | None = None,
        derive_face_from_person: bool | None = None,
    ) -> None:
        self.weights = str(weights) if weights else ""
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.iou = float(max(0.0, min(1.0, iou)))
        self.imgsz = int(imgsz)
        self.device_request = device
        self.allowed_classes = allowed_classes
        self.derive_face_from_person = derive_face_from_person
        self._session: Any = None
        self._input_name = ""
        self._output_names: list[str] = []
        self._provider = "CPUExecutionProvider"
        self._last_latency_ms = 0.0
        self._class_count: int | None = None

    def load(self) -> None:
        if self._session is not None:
            return
        if not self.weights:
            raise ModelMissingError(
                "ONNX detector needs weights; set models.detector_weights to a .onnx file"
            )
        path = Path(self.weights).expanduser()
        if not path.is_file():
            raise ModelMissingError(f"ONNX weights not found: {path}")
        try:
            choice = resolve_onnx_provider(self.device_request)
            self._session = build_onnx_session(str(path), providers=[choice.device])
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"could not load ONNX model {path}: {exc}") from exc

        inputs = self._session.get_inputs()
        if not inputs:
            raise ModelLoadError(f"ONNX model {path.name} declares no inputs")
        self._input_name = inputs[0].name
        shape = inputs[0].shape
        if isinstance(shape[-1], int) and shape[-1] > 0:
            self.imgsz = int(shape[-1])
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._provider = self._session.get_providers()[0]
        LOG.info("ONNX detector ready via %s (input=%s)", self._provider, self._input_name)

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def _require_session(self) -> Any:
        """Return the inference session, loading it on first use.

        Raises rather than returning ``None`` so a use after :meth:`close` is
        reported as a model problem instead of an ``AttributeError``.
        """
        if self._session is None:
            self.load()
        if self._session is None:
            raise ModelLoadError("the ONNX detector session was closed; load() it again")
        return self._session

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if self._session is None:
            self.load()
        session = self._require_session()
        image = validate_frame(frame, "frame")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        start = time.perf_counter()

        canvas, scale, pad_w = _letterbox(image, self.imgsz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        try:
            outputs = session.run(self._output_names, {self._input_name: blob})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"ONNX inference failed via {self._provider}: {exc}") from exc

        raw = np.asarray(outputs[0])
        detections = self._decode(raw, image.shape[1], image.shape[0], scale, pad_w)
        detections.sort(key=lambda d: d.box.area, reverse=True)
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        return detections

    def _decode(
        self, raw: np.ndarray, frame_w: int, frame_h: int, scale: float, pad_w: float
    ) -> list[FaceDetection]:
        array = raw
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise InferenceError(f"unsupported ONNX output shape {raw.shape}")

        if array.shape[0] < array.shape[1] and array.shape[0] <= 256:
            array = array.T

        is_nms_output = array.shape[1] in (5, 6)
        detections: list[FaceDetection] = []
        if is_nms_output:
            for row in array[array[:, 4] >= self.confidence]:
                class_id = int(row[5]) if array.shape[1] > 5 else 0
                if not self._keep(class_id):
                    continue
                x1, y1, x2, y2 = (float(v) for v in row[:4])
                box = self._to_frame(x1, y1, x2, y2, scale, pad_w)
                box = box.clamp(frame_w, frame_h)
                if box.width >= 8 and box.height >= 8:
                    detections.append(
                        FaceDetection(
                            box=box,
                            score=float(row[4]),
                            source_class=self._class_name(class_id),
                        )
                    )
        else:
            detections = self._decode_raw_head(array, frame_w, frame_h, scale, pad_w)
        return detections

    def _decode_raw_head(
        self, array: np.ndarray, frame_w: int, frame_h: int, scale: float, pad_w: float
    ) -> list[FaceDetection]:
        if self._class_count is None:
            self._class_count = max(0, array.shape[1] - 4)
        detections: list[FaceDetection] = []
        for index in range(array.shape[1] - 4):
            channel = array[:, index + 4]
            class_id = int(np.argmax(channel))
            score = float(channel[class_id])
            if score < self.confidence or not self._keep(class_id):
                continue
            cx, cy, w, h = (float(v) for v in array[index, :4])
            box = self._to_frame(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, scale, pad_w)
            box = box.clamp(frame_w, frame_h)
            if box.width < 8 or box.height < 8:
                continue
            detections.append(
                FaceDetection(box=box, score=score, source_class=self._class_name(class_id))
            )
        return detections

    def _to_frame(self, x1: float, y1: float, x2: float, y2: float, scale: float, pad: float) -> Box:
        def undo_x(value: float) -> float:
            return (value - pad) / scale

        return Box(undo_x(x1), y1 / scale, undo_x(x2), y2 / scale)

    def _class_name(self, class_id: int) -> str:
        if self._class_count in (None, 1) or class_id == 0:
            return "face"
        return "person"

    def _keep(self, class_id: int) -> bool:
        if self.allowed_classes is not None:
            return class_id in self.allowed_classes
        return True

    def close(self) -> None:
        self._session = None

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "weights": self.weights or None,
                "provider": self._provider,
                "imgsz": self.imgsz,
            }
        )
        return info
