"""Shared pytest fixtures.

Every test runs without a display server, without a camera and without network
access. Tests that genuinely need one of those are marked and skipped
automatically.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionai.ai.inference import InferencePipeline  # noqa: E402
from visionai.ai.models.registry import ModelRegistry  # noqa: E402
from visionai.camera.frame_source import SyntheticFrameSource  # noqa: E402
from visionai.config.paths import PackagePaths  # noqa: E402
from visionai.config.settings import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temporary config and log directory."""
    config = Settings()
    config.logging.file = False
    config.logging.console = False
    config.hardware.enabled = False
    return config.validate()


@pytest.fixture
def paths() -> PackagePaths:
    return PackagePaths.discover()


@pytest.fixture
def registry(paths: PackagePaths) -> ModelRegistry:
    return ModelRegistry(paths.face_models, paths.emotion_models)


@pytest.fixture(scope="session")
def app():
    """The session-wide QApplication, held so it outlives every widget."""
    global _QT_APP

    pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is not installed")
    from PySide6.QtWidgets import QApplication

    _QT_APP = QApplication.instance() or QApplication([])
    return _QT_APP


@pytest.fixture
def frame() -> np.ndarray:
    """A deterministic BGR test frame."""
    rng = np.random.default_rng(1234)
    base = rng.integers(40, 200, size=(360, 640, 3), dtype=np.uint8)
    base[100:260, 200:440] = (150, 180, 210)
    return base


@pytest.fixture
def gray_frame() -> np.ndarray:
    return np.full((120, 160), 128, dtype=np.uint8)


@pytest.fixture
def face_crop() -> np.ndarray:
    """A small face-like crop with dark eyes and a mouth line."""
    crop = np.full((96, 96, 3), 170, dtype=np.uint8)
    crop[28:40, 26:40] = (40, 40, 40)
    crop[28:40, 56:70] = (40, 40, 40)
    crop[62:70, 34:62] = (60, 60, 120)
    return crop


@pytest.fixture
def synthetic_source() -> Iterator[SyntheticFrameSource]:
    source = SyntheticFrameSource(width=320, height=240, faces=2)
    source.open()
    yield source
    source.release()


@pytest.fixture
def pipeline(
    settings: Settings, registry: ModelRegistry
) -> Iterator[InferencePipeline]:
    """A real pipeline built from whatever backends are installed."""
    detector, detector_report = registry.build_detector(settings)
    classifier, classifier_report = registry.build_classifier(settings)
    instance = InferencePipeline(
        settings, registry, detector, classifier, detector_report, classifier_report
    )
    yield instance
    instance.close()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long running tests")
    config.addinivalue_line("markers", "requires_ai: needs torch or ultralytics")
    config.addinivalue_line("markers", "requires_onnx: needs onnxruntime")
    config.addinivalue_line("markers", "requires_qt: needs PySide6")
    config.addinivalue_line("markers", "requires_native: needs the compiled C++ layer")
    config.addinivalue_line("markers", "camera: needs a real camera device")


_MARKER_REQUIREMENTS = {
    "requires_qt": lambda: has_qt(),
    "requires_native": lambda: has_native(),
    "requires_onnx": lambda: has_module("onnxruntime"),
    "requires_ai": lambda: has_module("torch"),
    "camera": lambda: has_cameras(),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip a marked test when its optional dependency is unavailable."""
    skip = pytest.mark.skip(reason="an optional dependency is not installed")
    for item in items:
        for marker in item.iter_markers():
            check = _MARKER_REQUIREMENTS.get(marker.name)
            if check is not None and not check():
                item.add_marker(skip)
                break


#: Threads this project starts. They are daemons, so they must not still be
#: running when the interpreter tears down: the C++ side of PySide6 and OpenCV
#: finalises underneath them and the process dies with SIGSEGV.
OWNED_THREADS = ("camera-capture", "visionai-inference", "hardware-monitor")

#: The QApplication must outlive every widget created from it. A widget whose
#: __del__ runs after the QApplication singleton has been destroyed is the other
#: classic way this suite used to die with SIGSEGV, so the reference is held
#: here and widgets are explicitly destroyed in the teardown below.
_QT_APP: Any = None


def _drain_qt_objects(app: Any) -> None:
    """Destroy surviving widgets while the QApplication is still alive."""
    if app is None:
        return
    from PySide6.QtWidgets import QApplication

    for widget in QApplication.topLevelWidgets():
        try:
            widget.close()
            widget.setParent(None)
            widget.deleteLater()
        except RuntimeError:
            # Already deleted by C++ side; nothing left to do.
            pass
    for _ in range(3):
        app.processEvents()
        try:
            app.sendPostedEvents(None, 0)
        except RuntimeError:
            break


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Tear down Qt objects and background workers before finalisation starts."""
    import gc
    import threading

    current = threading.current_thread().name
    for thread in list(threading.enumerate()):
        if thread.name in OWNED_THREADS and thread.name != current and thread.is_alive():
            # The workers poll on an Event, so they exit promptly once the
            # owning object has already been released.
            thread.join(timeout=1.0)

    if _QT_APP is not None:
        _drain_qt_objects(_QT_APP)
    gc.collect()


def has_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # noqa: BLE001
        return False
    return True


def has_qt() -> bool:
    return has_module("PySide6.QtWidgets")


def has_native() -> bool:
    from visionai.config.paths import find_native_candidates

    pybind_path, cabi_path = find_native_candidates()
    return pybind_path is not None or cabi_path is not None


def has_cameras() -> bool:
    return bool(Path("/dev/video0").exists())


requires_qt = pytest.mark.skipif(not has_qt(), reason="PySide6 is not installed")
requires_native = pytest.mark.skipif(not has_native(), reason="native layer is not built")
requires_camera = pytest.mark.skipif(not has_cameras(), reason="no camera device")
requires_onnx = pytest.mark.skipif(
    not has_module("onnxruntime"), reason="onnxruntime is not installed"
)
