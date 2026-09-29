"""Desktop and headless user interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - Qt is an optional dependency
    from visionai.ui.app import create_window, has_display, qt_available, run
    from visionai.ui.headless import render_text_frame, run_headless

__all__ = [
    "create_window",
    "has_display",
    "qt_available",
    "render_text_frame",
    "run",
    "run_headless",
]

_LAZY = {
    "create_window": "visionai.ui.app",
    "run": "visionai.ui.app",
    "qt_available": "visionai.ui.app",
    "has_display": "visionai.ui.app",
    "render_text_frame": "visionai.ui.headless",
    "run_headless": "visionai.ui.headless",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
