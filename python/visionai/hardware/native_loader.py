"""Discovery and loading of the compiled native hardware layer.

Three backends are tried in order of preference:

``pybind11``
    ``_visionai_native``, the richest interface (per-subsystem sampling plus the
    procfs parsers, which the test suite uses directly).
``cabi``
    ``libvisionai_hw.so`` loaded with :mod:`ctypes` and its JSON C ABI. Needs no
    Python headers at build time, so it works where the extension was not built.
``python``
    A pure-Python reader of ``/proc`` and ``/sys`` in this package. Slower and
    less complete, but it means the Hardware panel is never dead.

The bridge never raises on a missing library: it reports which backend is
active and why the others were skipped.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from visionai.ai.errors import NativeLibraryError
from visionai.config.paths import PackagePaths, find_native_candidates, find_native_library

LOG = logging.getLogger(__name__)

PYBIND11_MODULE = "_visionai_native"
CABI_LIBRARY = "libvisionai_hw.so"


class NativeBackend(Protocol):
    """The surface the bridge needs from any native implementation."""

    name: str

    def sample_all(self, disk_path: str, interface: str) -> dict[str, Any]: ...

    def sample(self, subsystem: str, argument: str) -> dict[str, Any]: ...

    def capabilities(self) -> list[str]: ...

    def library_version(self) -> str: ...


@dataclass
class BackendInfo:
    """Describes one candidate backend and whether it loaded."""

    name: str
    available: bool
    detail: str = ""
    path: str | None = None

    def describe(self) -> str:
        state = "available" if self.available else "unavailable"
        suffix = f" - {self.detail}" if self.detail else ""
        return f"{self.name}: {state}{suffix}"


@dataclass
class NativeLoadReport:
    """Aggregate outcome of the discovery process."""

    active: str = "python"
    version: str = "0.0.0"
    candidates: list[BackendInfo] = field(default_factory=list)
    is_native: bool = False

    @property
    def reason(self) -> str:
        for candidate in self.candidates:
            if candidate.available:
                continue
            if candidate.detail:
                return candidate.detail
        return ""

    def describe(self) -> str:
        lines = [f"active backend: {self.active} (v{self.version})"]
        lines.extend(f"  {c.describe()}" for c in self.candidates)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "version": self.version,
            "is_native": self.is_native,
            "candidates": [
                {"name": c.name, "available": c.available, "detail": c.detail, "path": c.path}
                for c in self.candidates
            ],
        }


class _Pybind11Backend:
    """Adapter over the compiled pybind11 extension."""

    name = "pybind11"

    def __init__(self, module: Any, path: str | None = None) -> None:
        self._module = module
        self._monitor = module.HardwareMonitor()
        self._path = path

    def sample_all(self, disk_path: str, interface: str) -> dict[str, Any]:
        return self._monitor.snapshot()

    def sample(self, subsystem: str, argument: str) -> dict[str, Any]:
        if subsystem == "disk":
            return self._monitor.sample_disk(argument or "/")
        if subsystem == "network":
            return self._monitor.sample_network(argument or "")
        if subsystem == "cpu":
            return self._monitor.sample_cpu()
        if subsystem == "memory":
            return self._monitor.sample_memory()
        if subsystem == "temperature":
            return self._monitor.sample_temperature()
        if subsystem == "gpus":
            return {"gpus": self._monitor.sample_gpus()}
        if subsystem == "system":
            return self._monitor.sample_system()
        raise ValueError(f"unknown subsystem {subsystem!r}")

    def capabilities(self) -> list[str]:
        return [c for c in str(self._module.capabilities()).split(",") if c]

    def library_version(self) -> str:
        return str(self._module.library_version())


class _CtypesBackend:
    """Adapter over the JSON C ABI using ctypes."""

    name = "cabi"

    def __init__(self, library_path: Path) -> None:
        self._library = ctypes.CDLL(str(library_path))
        self._path = str(library_path)
        self._configure()

    def _configure(self) -> None:
        lib = self._library
        lib.visionai_snapshot_json.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.visionai_snapshot_json.restype = ctypes.c_void_p
        lib.visionai_subsystem_json.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.visionai_subsystem_json.restype = ctypes.c_void_p
        lib.visionai_version_json.argtypes = []
        lib.visionai_version_json.restype = ctypes.c_void_p
        lib.visionai_capabilities_json.argtypes = []
        lib.visionai_capabilities_json.restype = ctypes.c_void_p
        lib.visionai_free_string.argtypes = [ctypes.c_void_p]
        lib.visionai_free_string.restype = None

    def _take(self, pointer: int) -> str:
        if not pointer:
            return ""
        try:
            text = ctypes.cast(pointer, ctypes.c_char_p).value
            return text.decode("utf-8", errors="replace") if text else ""
        finally:
            self._library.visionai_free_string(pointer)

    def sample_all(self, disk_path: str, interface: str) -> dict[str, Any]:
        raw = self._take(
            self._library.visionai_snapshot_json(
                disk_path.encode("utf-8"), interface.encode("utf-8")
            )
        )
        return json.loads(raw) if raw else {}

    def sample(self, subsystem: str, argument: str) -> dict[str, Any]:
        raw = self._take(
            self._library.visionai_subsystem_json(
                subsystem.encode("utf-8"), (argument or "").encode("utf-8")
            )
        )
        return json.loads(raw) if raw else {}

    def capabilities(self) -> list[str]:
        raw = self._take(self._library.visionai_capabilities_json())
        if not raw:
            return []
        return [c for c in json.loads(raw).get("capabilities", "").split(",") if c]

    def library_version(self) -> str:
        raw = self._take(self._library.visionai_version_json())
        return json.loads(raw).get("version", "0.0.0") if raw else "0.0.0"


def _import_pybind11(path: Path | None) -> Any:
    """Import the extension, preferring an explicit build directory."""
    if path is not None and path.name.startswith(PYBIND11_MODULE):
        directory = str(path.parent)
        if directory not in sys.path:
            sys.path.insert(0, directory)
    return importlib.import_module(PYBIND11_MODULE)


class NativeLoader:
    """Finds and loads the best available native backend."""

    def __init__(self, prefer_native: bool = True, require_native: bool = False) -> None:
        self.prefer_native = prefer_native
        self.require_native = require_native
        self._backend: NativeBackend | None = None
        self._report: NativeLoadReport | None = None

    @property
    def backend(self) -> NativeBackend | None:
        return self._backend

    @property
    def report(self) -> NativeLoadReport:
        if self._report is None:
            self.load()
        assert self._report is not None
        return self._report

    @property
    def is_native(self) -> bool:
        return self._backend is not None and self._backend.name != "python"

    def load(self, force: bool = False) -> NativeLoadReport:
        if self._report is not None and not force:
            return self._report

        candidates: list[BackendInfo] = []
        if self.prefer_native:
            pybind_path, cabi_path = find_native_candidates()
            if pybind_path is None and cabi_path is None:
                candidates.append(
                    BackendInfo(
                        "native",
                        False,
                        "no compiled artefact found; build with "
                        "`cmake -S . -B cpp/build && cmake --build cpp/build`",
                    )
                )
            else:
                candidates.extend(self._try_pybind11(pybind_path))
                candidates.append(
                    BackendInfo("cabi", False, "not needed: pybind11 is active")
                    if self._backend is not None
                    else self._try_cabi(cabi_path)[0]
                )
        else:
            candidates.append(BackendInfo("native", False, "disabled by settings"))

        if self._backend is None:
            if self.require_native:
                raise NativeLibraryError(
                    "no native hardware layer available:\n"
                    + "\n".join(c.describe() for c in candidates)
                )
            from visionai.hardware.python_backend import PythonBackend

            self._backend = PythonBackend()
            LOG.info(
                "using the pure-Python hardware backend; native layer unavailable"
            )
        else:
            LOG.info("native hardware backend active: %s", self._backend.name)

        self._report = NativeLoadReport(
            active=self._backend.name,
            version=self._safe_version(),
            candidates=candidates,
            is_native=self.is_native,
        )
        return self._report

    def _try_pybind11(self, path: Path | None) -> list[BackendInfo]:
        if path is None:
            return [BackendInfo("pybind11", False, "extension module not built")]
        try:
            module = _import_pybind11(path)
        except Exception as exc:  # noqa: BLE001
            return [BackendInfo("pybind11", False, f"{type(exc).__name__}: {exc}", str(path))]
        try:
            self._backend = _Pybind11Backend(module, str(path))
        except Exception as exc:  # noqa: BLE001
            return [BackendInfo("pybind11", False, f"{type(exc).__name__}: {exc}", str(path))]
        return [BackendInfo("pybind11", True, "extension module loaded", str(path))]

    def _try_cabi(self, path: Path | None) -> list[BackendInfo]:
        if path is None:
            if self._backend is None:
                return [BackendInfo("cabi", False, "libvisionai_hw.so not built")]
            return []
        try:
            self._backend = _CtypesBackend(path)
        except Exception as exc:  # noqa: BLE001
            return [BackendInfo("cabi", False, f"{type(exc).__name__}: {exc}", str(path))]
        return [BackendInfo("cabi", True, "C ABI loaded via ctypes", str(path))]

    def _safe_version(self) -> str:
        if self._backend is None:
            return "0.0.0"
        try:
            return self._backend.library_version()
        except Exception:  # noqa: BLE001
            return "unknown"


def module_available(name: str) -> bool:
    """True when an importable module is present, without importing it fully."""
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def describe_environment() -> dict[str, Any]:
    """Machine facts useful in bug reports and the About panel."""
    paths = PackagePaths.discover()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "is_64bit": platform.machine().endswith("64"),
        "opencv": _optional_version("cv2"),
        "numpy": _optional_version("numpy"),
        "torch": _optional_version("torch"),
        "onnxruntime": _optional_version("onnxruntime"),
        "ultralytics": _optional_version("ultralytics"),
        "pyside6": _optional_version("PySide6"),
        "project_root": str(paths.root),
        "build_dir": str(paths.native),
        "native_library": str(find_native_library() or "not built"),
        "threads": os.cpu_count(),
    }


def _optional_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))
