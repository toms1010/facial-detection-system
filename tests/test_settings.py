"""Settings validation, clamping and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionai.config.settings import (
    CameraSettings,
    HardwareSettings,
    LoggingSettings,
    ModelSettings,
    PipelineSettings,
    PrivacySettings,
    Settings,
    UISettings,
    load_settings,
    save_settings,
)


def test_defaults_are_valid() -> None:
    settings = Settings().validate()
    assert settings.models.detector == "yunet"
    assert settings.models.classifier == "heuristic"
    assert settings.camera.resolution == (1280, 720)
    assert settings.privacy.store_frames is False
    assert settings.privacy.allow_network_upload is False


def test_out_of_range_values_are_clamped() -> None:
    camera = CameraSettings(index=-5, width=10, height=99999, fps=100000, backend="nope")
    assert camera.index == 0
    assert camera.width == 160
    assert camera.height == 4320
    assert camera.fps == 240
    assert camera.backend == "auto"


def test_non_numeric_values_fall_back_to_defaults() -> None:
    camera = CameraSettings(index="not-a-number", width=None)
    assert camera.index == 0
    assert camera.width == 1280


def test_thresholds_are_clamped_to_unit_interval() -> None:
    models = ModelSettings(detection_confidence=5.0, detection_iou=-1.0, emotion_confidence=99.0)
    assert models.detection_confidence == 1.0
    assert models.detection_iou == 0.0
    assert models.emotion_confidence == 1.0


def test_unknown_choices_are_replaced() -> None:
    models = ModelSettings(detector="telepathy", classifier="crystal-ball", device="tpu")
    assert models.detector == "yunet"
    assert models.classifier == "heuristic"
    assert models.device == "auto"


def test_privacy_defaults_disable_every_storage_and_upload_path() -> None:
    privacy = PrivacySettings().validate()
    assert privacy.any_persistence_enabled is False
    assert privacy.allow_network_upload is False


def test_privacy_allows_storage_when_explicitly_enabled() -> None:
    privacy = PrivacySettings(store_frames=True).validate()
    assert privacy.store_frames is True
    assert privacy.any_persistence_enabled is True


def test_pipeline_bounds() -> None:
    pipeline = PipelineSettings(inference_fps=0.0, track_max_age=-3, smooth_alpha=5.0)
    assert pipeline.inference_fps == 0.5
    assert pipeline.track_max_age == 0
    assert pipeline.smooth_alpha == 1.0


def test_hardware_and_logging_bounds() -> None:
    hardware = HardwareSettings(interval=0.0, history_size=1)
    assert hardware.interval == 0.1
    assert hardware.history_size == 10

    logging_settings = LoggingSettings(level="chatty", max_bytes=1, backups=999)
    assert logging_settings.level == "INFO"
    assert logging_settings.max_bytes == 10_000
    assert logging_settings.backups == 50


def test_theme_validation() -> None:
    assert UISettings(theme="neon").validate().theme == "dark"
    assert UISettings(theme="light").validate().theme == "light"


def test_round_trip_through_dict() -> None:
    original = Settings()
    original.camera.index = 3
    original.models.emotion_confidence = 0.42
    original.ui.show_histograms = False
    restored = Settings.from_dict(original.to_dict())
    assert restored.camera.index == 3
    assert restored.models.emotion_confidence == pytest.approx(0.42)
    assert restored.ui.show_histograms is False


def test_unknown_keys_are_ignored_on_load() -> None:
    restored = Settings.from_dict({"camera": {"index": 2, "unknown": 1}, "nope": True})
    assert restored.camera.index == 2


def test_from_dict_survives_wrong_types() -> None:
    restored = Settings.from_dict({"camera": "not-a-dict", "version": "x"})
    assert isinstance(restored.camera, CameraSettings)


def test_copy_is_independent() -> None:
    original = Settings()
    clone = original.copy()
    clone.camera.index = 7
    assert original.camera.index != 7


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        ("camera.index", 5, 5),
        ("models.detector", "yolo", "yolo"),
        ("pipeline.inference_fps", 30.0, 30.0),
        ("ui.theme", "light", "light"),
        ("privacy.store_frames", True, True),
    ],
)
def test_merge_overrides(path: str, value: object, expected: object) -> None:
    merged = Settings().merge_overrides({path: value})
    node: object = merged
    for part in path.split("."):
        node = getattr(node, part)
    assert node == expected


def test_merge_overrides_clamps_and_ignores_unknown() -> None:
    merged = Settings().merge_overrides(
        {"camera.index": 999, "nonsense.path": 1, "models.emotion_confidence": 5.0}
    )
    assert merged.camera.index == 32
    assert merged.models.emotion_confidence == 1.0


def test_merge_overrides_ignores_none() -> None:
    merged = Settings().merge_overrides({"camera.index": None})
    assert merged.camera.index == 0


def test_save_and_load(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "settings.json"
    settings = Settings()
    settings.camera.index = 4
    written = save_settings(settings, target)
    assert written.exists()
    assert json.loads(written.read_text())["camera"]["index"] == 4
    assert load_settings(target).camera.index == 4


def test_save_tightens_permissions(tmp_path: Path) -> None:
    target = save_settings(Settings(), tmp_path / "settings.json")
    assert oct(target.stat().st_mode)[-3:] == "600"


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "absent.json").models.detector == "yunet"


def test_load_corrupt_file_returns_defaults(tmp_path: Path) -> None:
    broken = tmp_path / "settings.json"
    broken.write_text("{ this is not json", encoding="utf-8")
    assert load_settings(broken).camera.index == 0


def test_load_non_object_returns_defaults(tmp_path: Path) -> None:
    weird = tmp_path / "settings.json"
    weird.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_settings(weird).camera.fps == 30
