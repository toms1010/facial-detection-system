"""Filesystem locations used by the application.

Every path is resolved lazily so tests can redirect the XDG roots without
touching global state at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_DIRNAME = "linux-ai-vision"
CONFIG_FILENAME = "settings.json"


def _xdg(env_var: str, default: str) -> Path:
    raw = os.environ.get(env_var, "").strip()
    base = Path(raw) if raw else Path.home() / default
    return base.expanduser()


def project_root() -> Path:
    """Repository root when running from a source checkout.

    ``python/visionai/config/paths.py`` -> repository root is three parents up.
    When installed as a wheel there is no repository, so the package directory
    is used instead.
    """
    here = Path(__file__).resolve()
    candidate = here.parents[3]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return here.parents[1]


def package_dir() -> Path:
    """Root of the installed ``visionai`` package."""
    return Path(__file__).resolve().parents[1]


def user_config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / APP_DIRNAME


def user_data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / APP_DIRNAME


def default_config_path() -> Path:
    """User configuration file, overridable with ``VISIONAI_CONFIG``."""
    override = os.environ.get("VISIONAI_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return user_config_dir() / CONFIG_FILENAME


def default_log_dir() -> Path:
    override = os.environ.get("VISIONAI_LOG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return user_data_dir() / "logs"


def default_cache_dir() -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache") / APP_DIRNAME


@dataclass(frozen=True)
class PackagePaths:
    """Bundle of the model directories the application reads from."""

    root: Path = None  # type: ignore[assignment]
    models: Path = None  # type: ignore[assignment]
    face_models: Path = None  # type: ignore[assignment]
    emotion_models: Path = None  # type: ignore[assignment]
    native: Path = None  # type: ignore[assignment]

    @classmethod
    def discover(cls, root: Path | None = None) -> PackagePaths:
        base = Path(root) if root is not None else project_root()
        return cls(
            root=base,
            models=base / "models",
            face_models=base / "models" / "face",
            emotion_models=base / "models" / "emotion",
            native=base / "cpp" / "build",
        )

    def ensure_model_dirs(self) -> None:
        self.face_models.mkdir(parents=True, exist_ok=True)
        self.emotion_models.mkdir(parents=True, exist_ok=True)


def _native_search_roots() -> list[Path]:
    roots: list[Path] = [PackagePaths.discover().native]
    env_build = os.environ.get("VISIONAI_BUILD_DIR", "").strip()
    if env_build:
        roots.append(Path(env_build).expanduser())
    roots.append(package_dir())
    roots.append(package_dir() / "lib")
    return roots


def find_native_candidates() -> tuple[Path | None, Path | None]:
    """Locate the pybind11 extension and the C ABI shared library.

    Returns ``(pybind11_path, cabi_path)``, each ``None`` when not built. The
    build tree nests artefacts one level down (``cpp/build/cpp/...``) depending
    on the generator, so the search is recursive but skips heavy directories.
    """
    found: dict[str, Path | None] = {"pybind": None, "cabi": None}

    override = os.environ.get("VISIONAI_NATIVE_LIB", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            return None, None
        return (
            (candidate, None)
            if candidate.name.startswith("_visionai_native")
            else (None, candidate)
        )

    skip = {"CMakeFiles", ".git", "__pycache__", "node_modules"}
    for root in _native_search_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part in skip for part in path.parts):
                continue
            name = path.name
            if name.startswith("_visionai_native") and name.endswith((".so", ".pyd", ".dylib")):
                found["pybind"] = found["pybind"] or path
            elif name.startswith("libvisionai_hw.so"):
                found["cabi"] = found["cabi"] or path
        if found["pybind"] and found["cabi"]:
            break
    return found["pybind"], found["cabi"]


def find_native_library() -> Path | None:
    """The best available native artefact, preferring the pybind11 extension."""
    pybind_path, cabi_path = find_native_candidates()
    return pybind_path or cabi_path
