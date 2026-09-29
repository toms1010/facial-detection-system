"""The realtime pipeline: threaded engine plus shared statistics.

``PipelineEngine`` imports the AI layer, which in turn needs ``PipelineStats``
from this package. The re-exports here are therefore lazy so that importing
``visionai.pipeline.stats`` does not drag the engine (and the AI layer) in.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PerformanceReport", "PipelineEngine", "PipelineState", "PipelineStats"]

_LAZY = {
    "PipelineEngine": "visionai.pipeline.engine",
    "PipelineState": "visionai.pipeline.engine",
    "PipelineStats": "visionai.pipeline.stats",
    "PerformanceReport": "visionai.pipeline.stats",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
