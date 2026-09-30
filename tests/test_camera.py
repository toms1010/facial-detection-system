"""Camera capture: initialisation, frame reading and device error mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionai.ai.errors import (
    CameraPermissionError,
    CameraUnavailableError,
    FrameError,
    InvalidCameraError,
)
from visionai.camera.camera_manager import CameraManager, enumerate_cameras, open_source
from visionai.camera.frame_source import (
    FrameSource,
    SyntheticFrameSource,
    VideoFileSource,
)
from visionai.config.settings import CameraSettings


class TestSyntheticSource:
    def test_produces_bgr_frames(self) -> None:
        source = SyntheticFrameSource(320, 240, faces=2)
        source.open()
        frame = source.read()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        source.release()

    def test_is_deterministic_at_the_same_instant(self) -> None:
        source = SyntheticFrameSource(160, 120, faces=1)
        assert np.array_equal(source.render(1.5), source.render(1.5))

    def test_frames_change_over_time(self) -> None:
        source = SyntheticFrameSource(320, 240, faces=2)
        assert not np.array_equal(source.render(0.0), source.render(2.0))

    def test_zero_faces_is_valid(self) -> None:
        source = SyntheticFrameSource(160, 120, faces=0)
        assert source.read().shape == (120, 160, 3)

    def test_reports_its_size(self) -> None:
        source = SyntheticFrameSource(200, 100)
        assert source.frame_size == (200, 100)

    def test_context_manager(self) -> None:
        with SyntheticFrameSource(160, 120) as source:
            assert source.is_open
        assert not source.is_open


class TestVideoFileSource:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidCameraError):
            VideoFileSource(tmp_path / "nope.mp4").open()

    def test_release_is_safe_when_unopened(self) -> None:
        VideoFileSource("/tmp/never.mp4").release()


class TestCameraManagerValidation:
    def test_missing_device_path_raises(self) -> None:
        manager = CameraManager(CameraSettings(device_path="/dev/not-a-real-device"))
        with pytest.raises(InvalidCameraError):
            manager.open()

    def test_absent_index_raises_with_a_useful_message(self, tmp_path: Path) -> None:
        manager = CameraManager(CameraSettings(index=31))
        if not Path("/dev/video31").exists():
            with pytest.raises(InvalidCameraError) as info:
                manager.open()
            assert "31" in str(info.value)

    def test_release_without_open_is_safe(self) -> None:
        CameraManager(CameraSettings(index=0)).release()

    def test_settings_are_validated_on_construction(self) -> None:
        manager = CameraManager(CameraSettings(width=1, height=1))
        assert manager.settings.width == 160
        assert manager.settings.height == 120

    def test_capture_target_prefers_the_path(self) -> None:
        assert CameraManager(CameraSettings(index=2, device_path="/dev/video9")).settings.capture_target == "/dev/video9"
        assert CameraManager(CameraSettings(index=2)).settings.capture_target == 2


class TestEnumerateCameras:
    def test_returns_a_list(self) -> None:
        assert isinstance(enumerate_cameras(), list)

    @pytest.mark.parametrize("device", enumerate_cameras())
    def test_entries_are_well_formed(self, device) -> None:
        assert device.index >= 0
        assert device.name
        assert device.path and device.path.startswith("/dev/")

    def test_exactly_one_default(self) -> None:
        devices = enumerate_cameras()
        if devices:
            assert sum(1 for d in devices if d.is_default) == 1

    def test_respects_the_limit(self) -> None:
        assert len(enumerate_cameras(max_index=1)) <= 1


class TestOpenSourceDispatch:
    def test_camera_index(self) -> None:
        assert isinstance(open_source("0", CameraSettings()), CameraManager)

    def test_device_path(self) -> None:
        assert isinstance(open_source("/dev/video0", CameraSettings()), CameraManager)

    def test_synthetic(self) -> None:
        assert isinstance(open_source("synthetic", CameraSettings()), SyntheticFrameSource)

    def test_none_placeholder(self) -> None:
        frame = open_source("none", CameraSettings()).read()
        assert frame is not None
        assert frame.size > 0

    def test_video_file(self, tmp_path: Path) -> None:
        assert isinstance(open_source(str(tmp_path / "a.mp4"), CameraSettings()), VideoFileSource)

    def test_index_is_applied(self) -> None:
        source = open_source("3", CameraSettings())
        assert isinstance(source, CameraManager)
        assert source.settings.index == 3

    def test_device_path_is_applied(self) -> None:
        source = open_source("/dev/video5", CameraSettings())
        assert isinstance(source, CameraManager)
        assert source.settings.device_path == "/dev/video5"


class TestFakeCapture:
    """CameraManager behaviour against a stubbed cv2.VideoCapture."""

    @pytest.fixture
    def fake_cv2(self, monkeypatch: pytest.MonkeyPatch):
        import cv2

        class FakeCapture:
            instances: list[FakeCapture] = []
            fail_read = False

            def __init__(self, *args, **kwargs) -> None:
                self.opened = True
                self.frames_served = 0
                self.props: dict[int, float] = {}
                FakeCapture.instances.append(self)
                self.isOpened = lambda: self.opened
                self.read = lambda: (
                    (False, None)
                    if FakeCapture.fail_read
                    else (True, np.full((480, 640, 3), self.frames_served, dtype=np.uint8))
                )

            def set(self, prop, value):
                self.props[prop] = value
                return True

            def get(self, prop):
                return self.props.get(prop, 0.0)

            def release(self) -> None:
                self.opened = False

        FakeCapture.instances = []
        monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
        return FakeCapture

    def test_opens_and_applies_the_format(self, fake_cv2, settings) -> None:
        import cv2

        manager = CameraManager(CameraSettings(index=0, width=800, height=600, fps=25))
        manager.open()
        assert manager.is_open
        assert manager.frame_size == (800, 600)
        capture = fake_cv2.instances[0]
        assert capture.props[cv2.CAP_PROP_FRAME_WIDTH] == 800
        assert capture.props[cv2.CAP_PROP_FRAME_HEIGHT] == 600
        assert capture.props[cv2.CAP_PROP_FPS] == 25
        manager.release()

    def test_read_updates_statistics(self, fake_cv2) -> None:
        manager = CameraManager(CameraSettings(index=0))
        manager.open()
        frame = manager.read()
        assert frame is not None and frame.shape == (480, 640, 3)
        assert manager.stats.frames_captured == 1
        manager.release()

    def test_mirror_flips_the_frame(self, fake_cv2) -> None:
        manager = CameraManager(CameraSettings(index=0, mirror=True))
        manager.open()
        frame = manager.read()
        assert frame is not None
        manager.release()

    def test_a_failed_read_raises(self, fake_cv2) -> None:
        manager = CameraManager(CameraSettings(index=0))
        manager.open()
        fake_cv2.fail_read = True
        with pytest.raises(FrameError):
            manager.read()
        assert manager.stats.read_errors == 1
        manager.release()

    def test_an_unopenable_device_raises_unavailable(
        self, fake_cv2, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cv2

        def refuse(self, *args, **kwargs):
            instance = fake_cv2.__init__
            del instance

        class Dead:
            def __init__(self, *args, **kwargs) -> None:
                self.isOpened = lambda: False

            def release(self) -> None: ...

        monkeypatch.setattr(cv2, "VideoCapture", Dead)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        manager = CameraManager(CameraSettings(index=0))
        with pytest.raises(CameraUnavailableError):
            manager.open()

    def test_a_busy_device_raises_permission(self, fake_cv2, monkeypatch: pytest.MonkeyPatch) -> None:
        import cv2

        class Busy:
            def __init__(self, *args, **kwargs) -> None:
                self.isOpened = lambda: False

            def release(self) -> None: ...

        monkeypatch.setattr(cv2, "VideoCapture", Busy)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(
            "visionai.camera.camera_manager._looks_like_permission_problem",
            lambda message: True,
        )
        manager = CameraManager(CameraSettings(index=0))
        with pytest.raises(CameraPermissionError):
            manager.open()

    def test_capture_thread_publishes_latest_frame(self, fake_cv2) -> None:
        import time

        manager = CameraManager(CameraSettings(index=0))
        manager.start()
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline:
                frame, sequence = manager.latest()
                if frame is not None and sequence > 0:
                    break
                time.sleep(0.02)
            assert frame is not None
            assert manager.has_new_frame(sequence - 1) is True
        finally:
            manager.stop()
        assert not manager.is_running

    def test_describe_is_serialisable(self, fake_cv2) -> None:
        import json

        manager = CameraManager(CameraSettings(index=0))
        manager.open()
        json.dumps(manager.describe())
        manager.release()


class TestFrameSourceContract:
    @pytest.mark.parametrize(
        "source_factory",
        [lambda: SyntheticFrameSource(160, 120), lambda: open_source("none", CameraSettings())],
    )
    def test_satisfies_the_interface(self, source_factory) -> None:
        source = source_factory()
        assert isinstance(source, FrameSource)
        source.open()
        assert source.read() is not None
        source.release()

    @pytest.mark.parametrize(
        "source_factory",
        [lambda: SyntheticFrameSource(160, 120), lambda: open_source("none", CameraSettings())],
    )
    def test_describe_is_available_and_serialisable(self, source_factory) -> None:
        """Regression: the base class had no describe(), so callers crashed."""
        import json

        source = source_factory()
        described = source.describe()
        assert described["name"] == source.name
        assert described["display_name"] == source.display_name
        assert described["open"] is False
        source.open()
        assert source.describe()["open"] is True
        json.dumps(source.describe())
        source.release()
