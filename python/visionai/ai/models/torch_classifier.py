"""Expression classification with a TorchScript model."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from visionai.ai.errors import DeviceError, InferenceError, ModelMissingError
from visionai.ai.models.base import (
    ExpressionClassifier,
    ExpressionResult,
    to_probabilities,
    validate_frame,
)
from visionai.ai.models.descriptor import ModelDescriptor, load_descriptor, preprocess
from visionai.ai.models.devices import resolve_torch_device, thread_limits

LOG = logging.getLogger(__name__)


class TorchScriptExpressionClassifier(ExpressionClassifier):
    """Runs a TorchScript ``.pt`` expression model with CUDA or CPU fallback."""

    name = "torchscript"
    display_name = "TorchScript expression model"
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
        self._module = None
        self._torch = None
        self._descriptor: ModelDescriptor | None = None
        self._device = "cpu"
        self._last_latency_ms = 0.0

    def load(self) -> None:
        if self._module is not None:
            return
        if not self.model_path:
            raise ModelMissingError(
                "no expression model configured; set models.classifier_weights to a "
                "TorchScript file or keep the heuristic classifier"
            )
        descriptor = load_descriptor(self.model_path, self.descriptor_path)
        if descriptor.framework != "torchscript":
            raise ModelMissingError(
                f"descriptor for {descriptor.name} declares framework "
                f"{descriptor.framework!r}, not 'torchscript'"
            )
        try:
            import torch  # type: ignore[import-untyped]
        except Exception as exc:  # noqa: BLE001
            raise ModelMissingError("PyTorch is not installed") from exc

        choice = resolve_torch_device(self.device_request)
        try:
            torch.set_num_threads(thread_limits(4))
            module = torch.jit.load(str(descriptor.weights), map_location=choice.device)
            module.eval()
        except Exception as exc:  # noqa: BLE001
            raise ModelMissingError(f"could not load TorchScript {descriptor.weights}: {exc}") from exc

        self._torch = torch
        self._module = module
        self._device = choice.device
        self._descriptor = descriptor
        self.input_size = descriptor.input_size
        self.class_names = descriptor.canonical_classes
        LOG.info(
            "TorchScript expression model %s ready on %s", descriptor.name, choice.device
        )

    @property
    def is_ready(self) -> bool:
        return self._module is not None

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def descriptor(self) -> ModelDescriptor | None:
        return self._descriptor

    def predict(self, crop: np.ndarray) -> ExpressionResult:
        module = self._require_module()
        torch = self._torch
        assert torch is not None and self._descriptor is not None
        image = validate_frame(crop, "face crop")
        start = time.perf_counter()
        tensor = torch.from_numpy(preprocess(image, self._descriptor)).to(self._device)
        try:
            with torch.inference_mode():
                output = module(tensor)
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"TorchScript inference failed on {self._device}: {exc}") from exc

        result = self._to_result(self._as_numpy(output))
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = result.latency_ms
        return result

    def _require_module(self) -> Any:
        """Return the loaded TorchScript module, loading it on first use.

        Raises rather than returning ``None`` so a use after :meth:`close` is
        reported as a model problem instead of an ``AttributeError``.
        """
        if self._module is None:
            self.load()
        if self._module is None:
            raise ModelMissingError("the TorchScript classifier was closed; load() it again")
        return self._module

    def _as_numpy(self, output: object) -> np.ndarray:
        torch = self._torch
        assert torch is not None
        if isinstance(output, (list, tuple)):
            output = output[0]
        for method in ("detach", "cpu", "numpy"):
            step = getattr(output, method, None)
            if callable(step):
                output = step()
        return np.asarray(output)

    def _to_result(self, output: np.ndarray) -> ExpressionResult:
        assert self._descriptor is not None
        vector = output.reshape(-1).astype(np.float64)
        labels = self._descriptor.canonical_classes
        if vector.size != len(labels):
            raise InferenceError(
                f"model {self._descriptor.name} produced {vector.size} scores for "
                f"{len(labels)} declared classes; check the descriptor label order"
            )
        result = ExpressionResult.from_scores(
            dict(zip(labels, (float(v) for v in to_probabilities(vector)), strict=True)),
            model=self._descriptor.name,
            is_heuristic=False,
            threshold=self.confidence_threshold,
        )
        result.alternatives = tuple(k for k, _ in result.top_k(3)[1:])
        return result

    def predict_batch(self, crops: Sequence[np.ndarray]) -> list[ExpressionResult]:
        if not crops or self._module is None:
            return super().predict_batch(crops)
        torch = self._torch
        assert torch is not None and self._descriptor is not None
        start = time.perf_counter()
        batch = np.concatenate([preprocess(c, self._descriptor) for c in crops], axis=0)
        tensor = torch.from_numpy(batch).to(self._device)
        try:
            with torch.inference_mode():
                output = self._module(tensor)
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"batch inference failed on {self._device}: {exc}") from exc
        array = self._as_numpy(output)
        results = [
            self._to_result(array[i] if array.ndim > 1 else array)
            for i in range(len(crops))
        ]
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        return results

    def close(self) -> None:
        self._module = None

    def describe(self) -> dict:
        info = super().describe()
        info["device"] = self._device
        info["descriptor"] = self._descriptor.describe() if self._descriptor else None
        return info
