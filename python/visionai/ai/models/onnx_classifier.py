"""Expression classification with a real trained model through ONNX Runtime."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from visionai.ai.errors import InferenceError, ModelMissingError
from visionai.ai.models.base import (
    ExpressionClassifier,
    ExpressionResult,
    validate_frame,
)
from visionai.ai.models.descriptor import ModelDescriptor, load_descriptor, preprocess
from visionai.ai.models.devices import build_onnx_session, resolve_onnx_provider

LOG = logging.getLogger(__name__)


class OnnxExpressionClassifier(ExpressionClassifier):
    """Runs any ONNX expression model described by a :class:`ModelDescriptor`."""

    name = "onnx"
    display_name = "ONNX expression model"
    is_heuristic = False
    is_trained_model = True
    requires_weights = True

    def __init__(
        self,
        model_path: str | Path | None = None,
        descriptor_path: str | Path | None = None,
        device: str = "auto",
        confidence_threshold: float = 0.0,
    ) -> None:
        self.model_path = str(model_path) if model_path else ""
        self.descriptor_path = str(descriptor_path) if descriptor_path else None
        self.device_request = device
        self.confidence_threshold = float(confidence_threshold)
        self._session = None
        self._descriptor: ModelDescriptor | None = None
        self._input_name = ""
        self._output_name = ""
        self._provider = "CPUExecutionProvider"
        self._last_latency_ms = 0.0

    def load(self) -> None:
        if self._session is not None:
            return
        if not self.model_path:
            raise ModelMissingError(
                "no expression model configured; set models.classifier_weights to an "
                "ONNX file or keep the heuristic classifier"
            )
        descriptor = load_descriptor(self.model_path, self.descriptor_path)
        if descriptor.framework != "onnx":
            raise ModelMissingError(
                f"descriptor for {descriptor.name} declares framework "
                f"{descriptor.framework!r}, not 'onnx'"
            )
        choice = resolve_onnx_provider(self.device_request)
        try:
            self._session = build_onnx_session(str(descriptor.weights), providers=[choice.device])
        except Exception as exc:  # noqa: BLE001
            raise ModelMissingError(
                f"could not create an ONNX session for {descriptor.weights}: {exc}"
            ) from exc

        inputs = self._session.get_inputs()
        if not inputs:
            raise ModelMissingError(f"ONNX model {descriptor.name} has no inputs")
        self._input_name = inputs[0].name
        outputs = self._session.get_outputs()
        if not outputs:
            raise ModelMissingError(f"ONNX model {descriptor.name} has no outputs")
        self._output_name = outputs[0].name
        self._provider = self._session.get_providers()[0]
        self._descriptor = descriptor
        self.input_size = descriptor.input_size
        self.class_names = descriptor.canonical_classes
        LOG.info(
            "ONNX expression model %s ready via %s with %d classes",
            descriptor.name,
            self._provider,
            len(self.class_names),
        )

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def descriptor(self) -> ModelDescriptor | None:
        return self._descriptor

    def predict(self, crop: np.ndarray) -> ExpressionResult:
        if self._session is None:
            self.load()
        assert self._descriptor is not None
        image = validate_frame(crop, "face crop")
        start = time.perf_counter()
        tensor = preprocess(image, self._descriptor)
        try:
            outputs = self._session.run([self._output_name], {self._input_name: tensor})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"ONNX expression inference failed: {exc}") from exc

        result = self._to_result(np.asarray(outputs[0]))
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = result.latency_ms
        return result

    def _to_result(self, output: np.ndarray) -> ExpressionResult:
        assert self._descriptor is not None
        vector = output.reshape(-1)
        labels = self._descriptor.canonical_classes
        if vector.size == len(self._descriptor.classes) and len(labels) < vector.size:
            extra = vector.size - len(labels)
            labels = tuple(labels) + tuple(f"class_{i}" for i in range(extra))
        if vector.size != len(labels):
            raise InferenceError(
                f"model {self._descriptor.name} produced {vector.size} scores for "
                f"{len(labels)} declared classes; check the descriptor label order"
            )
        finite = np.where(np.isfinite(vector), vector, 0.0)
        result = ExpressionResult.from_scores(
            dict(zip(labels, (float(v) for v in finite), strict=True)),
            model=self._descriptor.name,
            is_heuristic=False,
            threshold=self.confidence_threshold,
        )
        result.alternatives = tuple(k for k, _ in result.top_k(3)[1:])
        return result

    def close(self) -> None:
        self._session = None

    def describe(self) -> dict:
        info = super().describe()
        info["provider"] = self._provider
        info["descriptor"] = self._descriptor.describe() if self._descriptor else None
        return info
