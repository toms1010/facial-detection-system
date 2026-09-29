"""Compute-device selection shared by the torch and onnxruntime backends.

The rule everywhere is the same: honour an explicit request, otherwise prefer
CUDA when a runtime actually reports a usable device, and never fail hard when
accelerated compute is missing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from visionai.ai.errors import DeviceError

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceChoice:
    """The resolved device plus why it was chosen."""

    device: str
    kind: str
    reason: str
    available: tuple[str, ...] = ()

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "gpu")

    def describe(self) -> str:
        return f"{self.device} ({self.kind}) - {self.reason}"


def torch_devices() -> tuple[str, ...]:
    try:
        import torch  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001 - absence must not break anything
        LOG.debug("torch unavailable while probing devices: %s", exc)
        return ()
    found = ["cpu"]
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                found.append(f"cuda:{index}")
    except Exception as exc:  # noqa: BLE001
        LOG.debug("torch CUDA probe failed: %s", exc)
    return tuple(found)


def onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        LOG.debug("onnxruntime unavailable while probing providers: %s", exc)
        return ()
    return tuple(ort.get_available_providers())


def resolve_torch_device(requested: str = "auto") -> DeviceChoice:
    available = torch_devices()
    if not available:
        raise DeviceError("PyTorch is not installed; GPU detection is unavailable")
    requested = (requested or "auto").strip().lower()
    if requested.startswith("cuda"):
        if requested in available or requested.split(":", 1)[0] in available:
            return DeviceChoice(requested, "cuda", "explicitly requested", available)
        LOG.warning("device %r requested but unavailable; falling back to cpu", requested)
        return DeviceChoice("cpu", "cpu", "CUDA unavailable", available)
    if requested == "cpu":
        return DeviceChoice("cpu", "cpu", "explicitly requested", available)
    for candidate in available:
        if candidate.startswith("cuda"):
            return DeviceChoice(candidate, "cuda", "CUDA detected", available)
    return DeviceChoice("cpu", "cpu", "no CUDA device present", available)


def resolve_onnx_provider(requested: str = "auto") -> DeviceChoice:
    available = onnx_providers()
    if not available:
        raise DeviceError("onnxruntime is not installed; ONNX inference is unavailable")
    requested = (requested or "auto").strip().lower()
    if requested == "cpu":
        return DeviceChoice("CPUExecutionProvider", "cpu", "explicitly requested", available)
    if requested.startswith("cuda"):
        for provider in available:
            if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                return DeviceChoice(provider, "cuda", "explicitly requested", available)
        LOG.warning("CUDA provider requested but unavailable; falling back to CPU")
        return DeviceChoice("CPUExecutionProvider", "cpu", "CUDA provider unavailable", available)
    for provider in available:
        if provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
            return DeviceChoice(provider, "cuda", "accelerated provider available", available)
    return DeviceChoice("CPUExecutionProvider", "cpu", "no accelerated provider", available)


def build_onnx_session(path: str, providers: list[str] | None = None, **kwargs: Any):
    """Create an onnxruntime session with sensible inference-only defaults."""
    import onnxruntime as ort  # type: ignore[import-untyped]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    for key, value in kwargs.items():
        if hasattr(options, key) and value is not None:
            setattr(options, key, value)
    return ort.InferenceSession(path, sess_options=options, providers=providers or None)


def thread_limits(default: int = 2) -> int:
    """Intra-op thread count that avoids oversubscribing a laptop CPU."""
    env = os.environ.get("VISIONAI_NUM_THREADS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    return max(1, default)


__all__ = [
    "DeviceChoice",
    "build_onnx_session",
    "onnx_providers",
    "resolve_onnx_provider",
    "resolve_torch_device",
    "thread_limits",
    "torch_devices",
]
