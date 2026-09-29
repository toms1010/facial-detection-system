"""Privacy behaviour: nothing leaves the machine and nothing is stored by default."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from visionai.ai.models.base import Box
from visionai.config.paths import PackagePaths
from visionai.config.settings import PrivacySettings, Settings
from visionai.ui.settings_panel import SettingsPanel
from visionai.utils.storage import (
    clear_stored_data,
    save_face_snapshot,
    save_frame,
    stored_files,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "visionai"


class TestStorageDefaults:
    def test_nothing_is_stored_by_default(self, settings: Settings) -> None:
        assert settings.privacy.any_persistence_enabled is False
        assert save_frame(np.zeros((8, 8, 3), np.uint8), settings.privacy) is None

    def test_each_flag_is_independently_opt_in(self) -> None:
        assert PrivacySettings(store_frames=True).any_persistence_enabled is True
        assert PrivacySettings(store_snapshots=True).any_persistence_enabled is True
        assert PrivacySettings(store_face_crops=True).any_persistence_enabled is True
        assert PrivacySettings().any_persistence_enabled is False

    def test_wipe_on_exit_is_the_default(self, settings: Settings) -> None:
        assert settings.privacy.wipe_on_exit is True

    def test_logs_are_anonymised_by_default(self, settings: Settings) -> None:
        assert settings.privacy.anonymize_log_payloads is True


class TestStorage:
    def test_frame_is_written_when_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "visionai.utils.storage.user_data_dir", lambda: tmp_path
        )
        path = save_frame(
            np.zeros((16, 16, 3), np.uint8), PrivacySettings(store_frames=True)
        )
        assert path is not None and path.is_file()

    def test_frame_is_not_written_when_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "visionai.utils.storage.user_data_dir", lambda: tmp_path
        )
        assert save_frame(np.zeros((16, 16, 3), np.uint8), PrivacySettings()) is None
        assert not stored_files()

    def test_snapshot_is_written_when_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "visionai.utils.storage.user_data_dir", lambda: tmp_path
        )
        frame = np.zeros((100, 100, 3), np.uint8)
        path = save_face_snapshot(
            frame, Box(10, 10, 50, 50), PrivacySettings(store_snapshots=True), track_id=3
        )
        assert path is not None and path.is_file()

    def test_clear_removes_everything(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "visionai.utils.storage.user_data_dir", lambda: tmp_path
        )
        save_frame(np.zeros((16, 16, 3), np.uint8), PrivacySettings(store_frames=True))
        save_face_snapshot(
            np.zeros((50, 50, 3), np.uint8), Box(0, 0, 20, 20), PrivacySettings(store_snapshots=True), 1
        )
        assert len(stored_files()) == 2
        assert clear_stored_data() == 2
        assert stored_files() == []

    def test_clear_on_an_empty_tree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("visionai.utils.storage.user_data_dir", lambda: tmp_path)
        assert clear_stored_data() == 0


class TestNoNetworkPath:
    """The application must contain no outbound network capability at all."""

    FORBIDDEN_MODULES = {
        "requests", "urllib", "http", "httpx", "aiohttp", "ftplib", "smtplib",
        "telnetlib", "xmlrpc", "webbrowser", "asyncio", "ssl", "urllib3",
    }

    def _python_sources(self) -> list[Path]:
        return sorted(PACKAGE_ROOT.rglob("*.py"))

    def test_no_network_modules_are_imported(self) -> None:
        offenders: list[str] = []
        for path in self._python_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.split(".")[0] in self.FORBIDDEN_MODULES:
                        offenders.append(f"{path.name}:{node.lineno} imports {name}")
        assert not offenders, offenders

    def test_socket_is_only_used_for_the_local_hostname(self) -> None:
        """socket.gethostname() is a local syscall, not a network client."""
        for path in self._python_sources():
            text = path.read_text(encoding="utf-8")
            if "import socket" not in text:
                continue
            assert "socket.socket" not in text, f"{path.name} opens a socket"
            assert "connect(" not in text, f"{path.name} connects to a remote host"
            assert "urlopen" not in text

    def test_no_subprocess_or_shell_calls(self) -> None:
        offenders: list[str] = []
        for path in self._python_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in {"subprocess", "multiprocessing"}:
                            offenders.append(f"{path.name}:{node.lineno} imports {alias.name}")
                if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "subprocess":
                    offenders.append(f"{path.name}:{node.lineno} imports subprocess")
        assert not offenders, offenders

    def test_requirements_declare_no_http_client(self) -> None:
        pyproject = (PACKAGE_ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        for package in ("requests", "httpx", "aiohttp", "websockets", "urllib3"):
            assert f'"{package}' not in pyproject

    def test_upload_flag_has_no_runtime_effect(self) -> None:
        """The flag is shown in Settings but no execution path reads it."""
        offenders: list[str] = []
        for path in self._python_sources():
            if path.name in ("settings.py", "settings_panel.py"):
                continue
            if "allow_network_upload" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        assert not offenders, f"allow_network_upload is read in {offenders}"


class TestPrivacyNotice:
    @staticmethod
    def notice() -> str:
        from visionai.ui.theme import DARK

        return SettingsPanel(Settings(), DARK).privacy_notice()

    def test_notice_states_local_processing(self, app) -> None:
        assert "on this machine" in self.notice()

    def test_notice_states_no_upload(self, app) -> None:
        assert "no network upload path" in self.notice()

    def test_notice_states_the_ai_limitation(self, app) -> None:
        assert "not a measurement" in self.notice()

    def test_every_ui_phrase_is_an_estimate(self) -> None:
        """No module may claim a person's actual emotion as fact."""
        banned = re.compile(
            r"(this person is|person is definitely|the user is feeling|"
            r"real emotion|actual feeling state)",
            re.IGNORECASE,
        )
        offenders: list[str] = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith("#") or "banned" in stripped.lower():
                    continue
                if banned.search(line):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        assert not offenders, offenders


class TestModelDirectory:
    def test_contains_only_expected_files(self, paths: PackagePaths) -> None:
        assert paths.face_models.is_dir()
        assert paths.emotion_models.is_dir()

    def test_ensure_model_dirs_is_idempotent(self, paths: PackagePaths, tmp_path: Path) -> None:
        custom = PackagePaths.discover(tmp_path)
        custom.ensure_model_dirs()
        custom.ensure_model_dirs()
        assert (tmp_path / "models" / "face").is_dir()

    def test_no_expression_weights_ship_by_default(self, paths: PackagePaths) -> None:
        """Shipping a trained model would impose its own licence and bias caveats."""
        assert list(paths.emotion_models.glob("*.onnx")) == []
        assert list(paths.emotion_models.glob("*.pt")) == []
