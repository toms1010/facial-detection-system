"""Typed, validated application settings with JSON persistence.

Design goals:

* every value is clamped into a valid range instead of raising, so a hand-edited
  config file can never wedge the application;
* unknown keys are preserved on load so a newer config survives a downgrade;
* groups map one-to-one onto the UI settings tabs.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from visionai.ai.errors import ConfigError
from visionai.config.paths import default_config_path

LOG = logging.getLogger(__name__)

T = TypeVar("T")

CAMERA_BACKENDS: dict[str, int | None] = {
    "auto": None,
    "v4l2": 2000,
    "gstreamer": 1800,
}


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    """Clamp to a range, falling back to ``default`` for anything unparseable.

    Settings come from a hand-editable JSON file and from the command line, so
    this must never raise: a bad value degrades to the default instead of
    wedging the application.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        LOG.warning("setting value %r is not a number; using %s", value, default)
        return default
    if not math.isfinite(parsed):
        LOG.warning("setting value %r is not finite; using %s", value, default)
        return default
    return low if parsed < low else high if parsed > high else parsed


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        LOG.warning("setting value %r is not a whole number; using %s", value, default)
        return default
    return max(low, min(high, parsed))


def _as_choice(value: Any, choices: tuple[str, ...], default: str) -> str:
    text = str(value).strip().lower()
    return text if text in choices else default


def _is_setting(target: Any, name: str) -> bool:
    """True only for a declared field, so a method name cannot be overwritten."""
    return is_dataclass(target) and name in {f.name for f in fields(target)}


def _valid_keys_hint(target: Any, parts: list[str]) -> str:
    if target is None:
        return ""
    names = sorted(f.name for f in fields(target))
    prefix = ".".join(parts[:-1])
    where = f" in {prefix!r}" if prefix else ""
    return f"; valid keys{where}: {', '.join(names)}"


@dataclass
class CameraSettings:
    """Camera device selection and capture format."""

    index: int = 0
    device_path: str | None = None

    def __post_init__(self) -> None:
        self.validate()  # CameraSettings is always valid from construction
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"
    fourcc: str = "MJPG"
    mirror: bool = False
    warmup_frames: int = 3

    def validate(self) -> CameraSettings:
        self.index = _clamp_int(self.index, 0, 32, 0)
        self.width = _clamp_int(self.width, 160, 7680, 1280)
        self.height = _clamp_int(self.height, 120, 4320, 720)
        self.fps = _clamp_int(self.fps, 1, 240, 30)
        self.backend = _as_choice(self.backend, tuple(CAMERA_BACKENDS), "auto")
        self.fourcc = str(self.fourcc or "MJPG").strip().upper()[:4] or "MJPG"
        self.mirror = bool(self.mirror)
        self.warmup_frames = _clamp_int(self.warmup_frames, 0, 60, 3)
        if self.device_path is not None:
            self.device_path = str(self.device_path).strip() or None
        return self

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def capture_target(self) -> str | int:
        """The value handed to ``cv2.VideoCapture``."""
        return self.device_path if self.device_path else self.index


@dataclass
class ModelSettings:
    """Detector/classifier selection, weights and thresholds."""

    detector: str = "yunet"
    detector_weights: str | None = None
    classifier: str = "heuristic"
    classifier_weights: str | None = None
    device: str = "auto"
    half_precision: bool = False
    imgsz: int = 640
    detection_confidence: float = 0.5
    detection_iou: float = 0.45
    emotion_confidence: float = 0.35
    max_faces: int = 16
    pad_ratio: float = 0.12
    grayscale_input: bool = False

    def __post_init__(self) -> None:
        self.validate()  # ModelSettings is always valid from construction

    def validate(self) -> ModelSettings:
        self.detector = _as_choice(
            self.detector, ("yunet", "haar", "yolo", "onnx", "auto"), "yunet"
        )
        self.classifier = _as_choice(
            self.classifier, ("heuristic", "onnx", "torchscript", "auto"), "heuristic"
        )
        self.device = _as_choice(self.device, ("auto", "cpu", "cuda", "cuda:0"), "auto")
        self.half_precision = bool(self.half_precision)
        self.imgsz = _clamp_int(self.imgsz, 64, 4096, 640)
        self.detection_confidence = float(_clamp(self.detection_confidence, 0.01, 1.0, 0.5))
        self.detection_iou = float(_clamp(self.detection_iou, 0.0, 1.0, 0.45))
        self.emotion_confidence = float(_clamp(self.emotion_confidence, 0.0, 1.0, 0.35))
        self.max_faces = _clamp_int(self.max_faces, 1, 64, 16)
        self.pad_ratio = float(_clamp(self.pad_ratio, 0.0, 0.5, 0.12))
        self.grayscale_input = bool(self.grayscale_input)
        for attr in ("detector_weights", "classifier_weights"):
            value = getattr(self, attr)
            setattr(self, attr, str(value).strip() or None if value is not None else None)
        return self

    @property
    def wants_gpu(self) -> bool:
        return self.device.startswith("cuda")


@dataclass
class PipelineSettings:
    """Throughput and tracking behaviour of the inference pipeline."""

    inference_fps: float = 15.0
    track_max_age: int = 15
    track_iou_threshold: float = 0.3
    track_smooth: bool = True
    smooth_alpha: float = 0.45
    label_hold_frames: int = 12
    processing_width: int = 960

    def __post_init__(self) -> None:
        self.validate()  # PipelineSettings is always valid from construction

    def validate(self) -> PipelineSettings:
        self.inference_fps = float(_clamp(self.inference_fps, 0.5, 120.0, 15.0))
        self.track_max_age = _clamp_int(self.track_max_age, 0, 600, 15)
        self.track_iou_threshold = float(_clamp(self.track_iou_threshold, 0.0, 1.0, 0.3))
        self.track_smooth = bool(self.track_smooth)
        self.smooth_alpha = float(_clamp(self.smooth_alpha, 0.0, 1.0, 0.45))
        self.label_hold_frames = _clamp_int(self.label_hold_frames, 0, 600, 12)
        self.processing_width = _clamp_int(self.processing_width, 160, 7680, 960)
        return self


@dataclass
class HardwareSettings:
    """Native hardware monitoring behaviour."""

    enabled: bool = True
    interval: float = 1.0
    disk_path: str = "/"
    network_interface: str | None = None
    prefer_native: bool = True
    history_size: int = 240

    def __post_init__(self) -> None:
        self.validate()  # HardwareSettings is always valid from construction

    def validate(self) -> HardwareSettings:
        self.enabled = bool(self.enabled)
        self.interval = float(_clamp(self.interval, 0.1, 60.0, 1.0))
        self.disk_path = str(self.disk_path or "/")
        network_interface = self.network_interface
        if network_interface is not None:
            network_interface = str(network_interface).strip() or None
        self.network_interface = network_interface
        self.prefer_native = bool(self.prefer_native)
        self.history_size = _clamp_int(self.history_size, 10, 5000, 240)
        return self


@dataclass
class UISettings:
    """Presentation preferences for the desktop application."""

    theme: str = "dark"
    show_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_track_ids: bool = True
    show_fps: bool = True
    show_histograms: bool = True
    privacy_notice_accepted: bool = False
    start_with_camera: bool = False

    def __post_init__(self) -> None:
        self.validate()  # UISettings is always valid from construction

    def validate(self) -> UISettings:
        self.theme = _as_choice(self.theme, ("dark", "light", "system"), "dark")
        for attr in (
            "show_boxes",
            "show_labels",
            "show_confidence",
            "show_track_ids",
            "show_fps",
            "show_histograms",
            "privacy_notice_accepted",
            "start_with_camera",
        ):
            setattr(self, attr, bool(getattr(self, attr)))
        return self


@dataclass
class PrivacySettings:
    """Privacy defaults. Every storage or network path is opt-in."""

    store_frames: bool = False
    store_snapshots: bool = False
    store_face_crops: bool = False
    allow_network_upload: bool = False
    anonymize_log_payloads: bool = True
    retain_hours: int = 0
    wipe_on_exit: bool = True

    def __post_init__(self) -> None:
        self.validate()  # PrivacySettings is always valid from construction

    def validate(self) -> PrivacySettings:
        self.store_frames = bool(self.store_frames)
        self.store_snapshots = bool(self.store_snapshots)
        self.store_face_crops = bool(self.store_face_crops)
        # allow_network_upload gates the (intentionally unimplemented) upload
        # path only. It is deliberately independent of local storage, so that
        # enabling local snapshots never requires opting into a network path.
        self.allow_network_upload = bool(self.allow_network_upload)
        self.anonymize_log_payloads = bool(self.anonymize_log_payloads)
        self.retain_hours = _clamp_int(self.retain_hours, 0, 8760, 0)
        self.wipe_on_exit = bool(self.wipe_on_exit)
        return self

    @property
    def any_persistence_enabled(self) -> bool:
        return self.store_frames or self.store_snapshots or self.store_face_crops


@dataclass
class LoggingSettings:
    """Logging destinations and verbosity."""

    level: str = "INFO"
    console: bool = True
    file: bool = True
    directory: str | None = None
    max_bytes: int = 2_000_000
    backups: int = 3

    def __post_init__(self) -> None:
        self.validate()  # LoggingSettings is always valid from construction

    def validate(self) -> LoggingSettings:
        self.level = _as_choice(
            self.level, ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), "INFO"
        )
        self.console = bool(self.console)
        self.file = bool(self.file)
        if self.directory is not None:
            self.directory = str(self.directory).strip() or None
        self.max_bytes = _clamp_int(self.max_bytes, 10_000, 100_000_000, 2_000_000)
        self.backups = _clamp_int(self.backups, 0, 50, 3)
        return self


@dataclass
class Settings:
    """Root settings object grouping every configurable area."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    ui: UISettings = field(default_factory=UISettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    version: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> Settings:
        self.camera.validate()
        self.models.validate()
        self.pipeline.validate()
        self.hardware.validate()
        self.ui.validate()
        self.privacy.validate()
        self.logging.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> Settings:
        return Settings.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Settings:
        payload = copy.deepcopy(payload or {})
        known = {f.name for f in fields(cls)}
        instance = cls()
        for key, value in payload.items():
            if key not in known:
                continue
            if is_dataclass(getattr(instance, key)):
                setattr(instance, key, _build_group(type(getattr(instance, key)), value))
            else:
                setattr(instance, key, value)
        return instance.validate()

    def merge_overrides(self, overrides: dict[str, Any], *, strict: bool = False) -> Settings:
        """Apply dotted CLI overrides such as ``{"models.detector": "yolo"}``.

        Args:
            overrides: mapping of dotted setting paths to values.
            strict: report an unknown path as an error instead of ignoring it.
                Overrides also arrive from a hand-edited file, which must never
                wedge the application, so the default stays lenient. Explicit
                command-line input passes ``True`` so a typo is not silently
                dropped.

        Raises:
            ConfigError: if ``strict`` and a path does not name a real setting.
        """
        result = self.copy()
        for dotted, value in overrides.items():
            if value is None:
                continue
            target: Any = result
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = getattr(target, part, None)
                if target is None:
                    break
            if target is None or not _is_setting(target, parts[-1]):
                if strict:
                    raise ConfigError(f"unknown setting {dotted!r}{_valid_keys_hint(target, parts)}")
                LOG.warning("ignoring unknown settings path %r", dotted)
                continue
            setattr(target, parts[-1], value)
        return result.validate()

    def save(self, path: Path | str | None = None) -> Path:
        return save_settings(self, path)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Settings:
        return load_settings(path)


def _build_group(group_cls: type[T], payload: Any) -> T:
    # Deserialising JSON into a dataclass is inherently dynamic, so the
    # instance is treated as Any here rather than fighting the type checker.
    instance: Any = group_cls()
    if not isinstance(payload, dict):
        return instance
    known = {f.name for f in fields(instance)}
    for key, value in payload.items():
        if key in known:
            setattr(instance, key, value)
    return instance


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from disk, falling back to defaults on any problem."""
    target = Path(path) if path is not None else default_config_path()
    if not target.is_file():
        return Settings().validate()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("could not read config %s (%s); using defaults", target, exc)
        return Settings().validate()
    if not isinstance(payload, dict):
        LOG.warning("config %s is not a JSON object; using defaults", target)
        return Settings().validate()
    return Settings.from_dict(payload)


def save_settings(settings: Settings, path: Path | str | None = None) -> Path:
    """Atomically write settings, creating parent directories as needed."""
    target = Path(path) if path is not None else default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(settings.to_dict(), indent=2, sort_keys=True)
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(target)
    try:
        target.chmod(0o600)
    except OSError:
        LOG.debug("could not tighten permissions on %s", target)
    return target
