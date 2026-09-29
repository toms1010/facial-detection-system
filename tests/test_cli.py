"""Command-line interface behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionai import __version__
from visionai.main import build_parser, main


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every CLI test away from the real user configuration and the desktop.

    ``visionai run`` would otherwise open a real window and block in the Qt event
    loop, so the display variables are removed and the command reports that it
    needs a display instead.
    """
    path = tmp_path / "settings.json"
    monkeypatch.setenv("VISIONAI_CONFIG", str(path))
    monkeypatch.setenv("VISIONAI_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    return path


def run(argv: list[str]) -> int:
    return main(argv)


class TestParser:
    def test_default_command_is_run(self) -> None:
        args = build_parser().parse_args([])
        assert args.command is None

    @pytest.mark.parametrize(
        "command",
        ["run", "headless", "list-cameras", "devices", "models", "self-test", "benchmark", "settings"],
    )
    def test_every_command_parses(self, command: str) -> None:
        assert build_parser().parse_args([command]).command == command

    def test_version(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as info:
            build_parser().parse_args(["--version"])
        assert info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_help_mentions_the_ai_limitation(self) -> None:
        # argparse hard-wraps the epilog, so compare on collapsed whitespace.
        help_text = " ".join(build_parser().format_help().split())
        assert "AI estimate" in help_text
        assert "not a measurement" in help_text

    def test_model_flags(self) -> None:
        args = build_parser().parse_args(
            ["run", "--detector", "yolo", "--classifier", "onnx", "--device", "cuda"]
        )
        assert args.detector == "yolo"
        assert args.classifier == "onnx"
        assert args.device == "cuda"


class TestListCameras:
    def test_lists_or_explains(self, capsys: pytest.CaptureFixture) -> None:
        code = run(["--no-log-file", "list-cameras"])
        out = capsys.readouterr().out
        assert code in (0, 1)
        if code == 0:
            assert "capture device" in out
        else:
            assert "No capture devices" in out


class TestDevices:
    def test_reports_a_backend_and_values(self, capsys: pytest.CaptureFixture) -> None:
        assert run(["--no-log-file", "devices"]) == 0
        out = capsys.readouterr().out
        assert "active backend" in out
        assert "CPU" in out
        assert "RAM" in out


class TestModels:
    def test_lists_face_models(self, capsys: pytest.CaptureFixture) -> None:
        assert run(["--no-log-file", "models"]) == 0
        out = capsys.readouterr().out
        assert "Face models" in out
        assert "yunet" in out
        assert "heuristic fallback" in out


class TestSelfTest:
    @pytest.mark.slow
    def test_passes(self, capsys: pytest.CaptureFixture) -> None:
        assert run(["--no-log-file", "self-test"]) == 0
        out = capsys.readouterr().out
        assert "all checks passed" in out
        assert "FAIL" not in out

    def test_reports_the_model_quality(self, capsys: pytest.CaptureFixture) -> None:
        run(["--no-log-file", "self-test"])
        out = capsys.readouterr().out
        assert "inference runs" in out
        assert "detector loads" in out


class TestHeadless:
    def test_runs_for_a_fixed_frame_count(self, capsys: pytest.CaptureFixture) -> None:
        code = run(
            ["--no-log-file", "headless", "--source", "synthetic", "--frames", "8", "--no-colour"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Average FPS" in out
        assert "AI-estimated facial expression" in out

    def test_reports_performance_metrics(self, capsys: pytest.CaptureFixture) -> None:
        run(["--no-log-file", "headless", "--source", "synthetic", "--frames", "6", "--no-colour"])
        out = capsys.readouterr().out
        for label in ("Frames processed", "Average FPS", "Avg inference", "CPU usage", "RAM usage"):
            assert label in out

    def test_hardware_lines_are_printed(self, capsys: pytest.CaptureFixture) -> None:
        run(["--no-log-file", "headless", "--source", "synthetic", "--frames", "6", "--no-colour"])
        out = capsys.readouterr().out
        assert "Hardware" in out
        assert "Disk" in out
        assert "Network" in out

    def test_bad_source_reports_an_error(self, capsys: pytest.CaptureFixture) -> None:
        code = run(["--no-log-file", "headless", "--source", "/dev/not-a-camera", "--frames", "2"])
        assert code == 1

    def test_json_dump(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        target = tmp_path / "out" / "report.json"
        run(
            [
                "--no-log-file", "headless", "--source", "synthetic",
                "--frames", "4", "--no-colour", "--dump-json", str(target),
            ]
        )
        assert target.is_file()
        assert "average_fps" in json.loads(target.read_text())


class TestBenchmark:
    @pytest.mark.slow
    def test_produces_a_report(self, capsys: pytest.CaptureFixture) -> None:
        assert run(["--no-log-file", "benchmark", "--frames", "12"]) == 0
        out = capsys.readouterr().out
        assert "Average FPS" in out
        assert "P95 inference" in out

    def test_json_output(self, capsys: pytest.CaptureFixture) -> None:
        run(["--no-log-file", "benchmark", "--frames", "8", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["frames"] == 8
        assert "average_inference_ms" in payload
        assert "source" in payload


class TestSettings:
    def test_show(self, capsys: pytest.CaptureFixture) -> None:
        assert run(["--no-log-file", "settings", "show"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["models"]["detector"] == "yunet"

    def test_path(self, capsys: pytest.CaptureFixture, isolated_config: Path) -> None:
        assert run(["--no-log-file", "settings", "path"]) == 0
        assert capsys.readouterr().out.strip() == str(isolated_config)

    def test_set_writes_the_file(self, capsys: pytest.CaptureFixture, isolated_config: Path) -> None:
        assert run(["--no-log-file", "settings", "set", "camera.index=3", "models.detector=yolo"]) == 0
        payload = json.loads(isolated_config.read_text())
        assert payload["camera"]["index"] == 3
        assert payload["models"]["detector"] == "yolo"

    def test_set_clamps_out_of_range(self, isolated_config: Path) -> None:
        run(["--no-log-file", "settings", "set", "camera.fps=99999"])
        assert json.loads(isolated_config.read_text())["camera"]["fps"] == 240

    def test_set_ignores_malformed(self, isolated_config: Path) -> None:
        run(["--no-log-file", "settings", "set", "camera.index"])
        assert json.loads(isolated_config.read_text())["camera"]["index"] == 0

    def test_reset(self, isolated_config: Path) -> None:
        run(["--no-log-file", "settings", "set", "camera.index=9"])
        run(["--no-log-file", "settings", "reset"])
        assert json.loads(isolated_config.read_text())["camera"]["index"] == 0

    def test_defaults_when_no_file_exists(self, capsys: pytest.CaptureFixture) -> None:
        run(["--no-log-file", "settings", "show"])
        assert json.loads(capsys.readouterr().out)["camera"]["index"] == 0


class TestOverrides:
    def test_set_flags_override_the_config(self, capsys: pytest.CaptureFixture) -> None:
        run(
            [
                "--no-log-file", "--config", "/nonexistent.json",
                "settings", "show", "--set", "camera.index=2",
            ]
        )
        assert json.loads(capsys.readouterr().out)["camera"]["index"] == 2

    def test_overrides_are_not_persisted(self, isolated_config: Path) -> None:
        run(["--no-log-file", "settings", "show", "--set", "camera.index=2"])
        assert not isolated_config.exists()

    def test_type_coercion(self, capsys: pytest.CaptureFixture) -> None:
        run(
            [
                "--no-log-file", "settings", "show",
                "--set", "privacy.store_frames=true",
                "--set", "pipeline.inference_fps=22.5",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["privacy"]["store_frames"] is True
        assert payload["pipeline"]["inference_fps"] == 22.5

    def test_model_flags_reach_the_settings(self) -> None:
        """Every convenience flag must land in the right settings group."""
        from visionai.main import _settings_from_args, build_parser

        args = build_parser().parse_args(
            [
                "run",
                "--detector", "haar",
                "--classifier", "onnx",
                "--theme", "light",
                "--device", "cpu",
                "--detection-confidence", "0.8",
                "--emotion-confidence", "0.6",
                "--inference-fps", "20",
                "--detector-weights", "/tmp/w.pt",
            ]
        )
        settings = _settings_from_args(args)
        assert settings.models.detector == "haar"
        assert settings.models.classifier == "onnx"
        assert settings.models.device == "cpu"
        assert settings.models.detection_confidence == pytest.approx(0.8)
        assert settings.models.emotion_confidence == pytest.approx(0.6)
        assert settings.models.detector_weights == "/tmp/w.pt"
        assert settings.pipeline.inference_fps == pytest.approx(20.0)
        assert settings.ui.theme == "light"

    def test_flags_are_optional(self) -> None:
        from visionai.main import _settings_from_args, build_parser

        settings = _settings_from_args(build_parser().parse_args(["run"]))
        assert settings.models.detector == "yunet"


class TestRunFallback:
    def test_without_a_display_it_reports_clearly(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        code = run(["--no-log-file", "run"])
        err = capsys.readouterr().err
        assert code == 3
        assert "No display detected" in err
        assert "headless" in err
