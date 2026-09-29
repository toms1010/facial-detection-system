"""Hardware monitoring: backends, normalisation, formatting and thread safety."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from visionai.ai.errors import SensorUnavailableError
from visionai.config.settings import HardwareSettings, Settings
from visionai.hardware.formatting import (
    format_bytes,
    format_duration,
    format_percent,
    format_rate,
    format_temperature,
)
from visionai.hardware.hardware_bridge import HardwareBridge, HardwareSnapshot, format_snapshot
from visionai.hardware.native_loader import NativeLoader, describe_environment
from visionai.hardware.privacy import redact_mapping, scrub_text
from visionai.hardware.python_backend import PythonBackend


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1.0 KB"),
            (1024**2, "1.0 MB"),
            (1024**3, "1.0 GB"),
            (1024**4, "1.0 TB"),
            (int(6.4 * 1024**3), "6.4 GB"),
        ],
    )
    def test_format_bytes(self, value: int, expected: str) -> None:
        assert format_bytes(value) == expected

    def test_format_bytes_handles_none(self) -> None:
        assert format_bytes(None) == "n/a"

    def test_format_bytes_handles_negatives(self) -> None:
        assert format_bytes(-2048) == "-2.0 KB"

    def test_decimal_units(self) -> None:
        assert format_bytes(1_000_000, binary=False) == "1.0 MB"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0 B/s"), (512, "512 B/s"), (1024**2, "1.0 MB/s")],
    )
    def test_format_rate(self, value: float, expected: str) -> None:
        assert format_rate(value) == expected

    def test_format_rate_handles_none(self) -> None:
        assert format_rate(None) == "0 B/s"

    def test_format_temperature(self) -> None:
        assert format_temperature(51.4) == "51°C"
        assert format_temperature(None) == "n/a"

    def test_format_percent(self) -> None:
        assert format_percent(32.4) == "32%"
        assert format_percent(32.44, 1) == "32.4%"
        assert format_percent(None) == "n/a"

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(45, "45s"), (90, "1m 30s"), (3700, "1h 1m"), (90000, "1d 1h"), (None, "n/a")],
    )
    def test_format_duration(self, seconds: int | None, expected: str) -> None:
        assert format_duration(seconds) == expected


class TestPythonBackend:
    def test_returns_every_subsystem(self) -> None:
        backend = PythonBackend()
        snapshot = backend.sample_all("/", "")
        assert set(snapshot) >= {
            "cpu", "memory", "temperature", "disk", "network", "gpus", "system", "backend"
        }
        assert snapshot["backend"] == "python"

    def test_reports_memory(self) -> None:
        memory = PythonBackend().sample_memory()
        assert memory["total_bytes"] > 0
        assert 0 <= memory["usage_percent"] <= 100

    def test_cpu_usage_needs_two_samples(self) -> None:
        backend = PythonBackend()
        assert backend.sample_cpu()["usage_percent"] == 0.0
        time.sleep(0.05)
        second = backend.sample_cpu()
        assert 0.0 <= second["usage_percent"] <= 100.0

    def test_cpu_model_is_populated(self) -> None:
        assert PythonBackend().sample_cpu()["model_name"]

    def test_reports_temperature_sensors(self) -> None:
        temperature = PythonBackend().sample_temperature()
        for sensor in temperature["sensors"]:
            assert -40.0 <= sensor["celsius"] <= 150.0

    def test_rejects_implausible_temperatures(self) -> None:
        assert PythonBackend().sample_temperature()["max_celsius"] <= 150.0

    def test_disk_usage(self) -> None:
        disk = PythonBackend().sample_disk("/")
        assert disk["available"] is True
        assert disk["total_bytes"] > 0
        assert disk["usage_percent"] <= 100

    def test_unreadable_disk_path(self) -> None:
        assert PythonBackend().sample_disk("/no/such/path")["available"] is False

    def test_network_interfaces(self) -> None:
        network = PythonBackend().sample_network("")
        assert network["interfaces"]
        assert any(i["name"] == "lo" for i in network["interfaces"])

    def test_loopback_is_not_chosen_by_default(self) -> None:
        network = PythonBackend().sample_network("")
        if network["primary_interface"]:
            assert network["primary_interface"] != "lo"

    def test_requested_interface_is_honoured(self) -> None:
        assert PythonBackend().sample_network("lo")["primary_interface"] == "lo"

    def test_gpus_do_not_crash(self) -> None:
        assert isinstance(PythonBackend().sample("gpus", "")["gpus"], list)

    def test_system_information(self) -> None:
        system = PythonBackend().sample_system()
        assert system["hostname"]
        assert system["uptime_seconds"] > 0
        assert system["architecture"]

    def test_unknown_subsystem_raises(self) -> None:
        with pytest.raises(ValueError):
            PythonBackend().sample("nonsense", "")

    def test_capabilities_are_listed(self) -> None:
        assert "cpu" in PythonBackend().capabilities()

    def test_backend_priority_is_cython_free(self) -> None:
        assert PythonBackend.name == "python"


class TestNativeLoader:
    def test_reports_a_backend_even_without_the_native_build(self) -> None:
        loader = NativeLoader(prefer_native=False)
        report = loader.load()
        assert report.active == "python"
        assert report.is_native is False
        assert report.to_dict()["active"] == "python"

    def test_candidates_explain_themselves(self) -> None:
        report = NativeLoader(prefer_native=False).load()
        assert report.candidates
        assert all(c.detail for c in report.candidates)

    def test_describe_is_readable(self) -> None:
        assert "active backend" in NativeLoader(prefer_native=False).load().describe()

    @pytest.mark.requires_native
    def test_finds_a_working_backend(self) -> None:
        loader = NativeLoader()
        report = loader.load()
        assert report.active in ("pybind11", "cabi")
        assert loader.backend is not None
        assert loader.backend.sample_all("/", "").get("memory")


class TestHardwareBridge:
    def test_samples_every_field(self, settings: Settings) -> None:
        bridge = HardwareBridge(settings.hardware)
        snapshot = bridge.sample()
        assert snapshot.ram_total_bytes > 0
        assert 0 <= snapshot.ram_percent <= 100
        assert snapshot.disk_total_bytes > 0
        assert snapshot.backend in ("pybind11", "cabi", "python")

    def test_snapshot_is_serialisable(self) -> None:
        bridge = HardwareBridge(HardwareSettings())
        json.dumps(bridge.sample().to_dict())

    def test_latest_is_populated_after_sampling(self) -> None:
        bridge = HardwareBridge(HardwareSettings())
        assert bridge.latest is None
        bridge.sample()
        assert bridge.latest is not None

    def test_history_accumulates(self) -> None:
        bridge = HardwareBridge(HardwareSettings())
        for _ in range(3):
            bridge.sample()
        assert len(bridge.history("cpu_percent")) == 3

    def test_background_thread_samples(self) -> None:
        bridge = HardwareBridge(HardwareSettings(interval=0.1))
        bridge.start()
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline and bridge.latest is None:
                time.sleep(0.02)
            assert bridge.latest is not None
        finally:
            bridge.stop()

    def test_stop_is_safe_when_not_started(self) -> None:
        HardwareBridge(HardwareSettings()).stop()

    def test_context_manager(self) -> None:
        with HardwareBridge(HardwareSettings(interval=0.2)) as bridge:
            time.sleep(0.2)
            assert bridge.latest is not None

    def test_notes_explain_missing_sensors(self) -> None:
        snapshot = HardwareBridge(HardwareSettings()).sample()
        assert isinstance(snapshot.notes, list)

    def test_hostname_is_redacted_by_default(self) -> None:
        snapshot = HardwareBridge(HardwareSettings(), anonymize=True).sample()
        assert snapshot.hostname == "redacted"

    def test_anonymisation_can_be_disabled(self) -> None:
        snapshot = HardwareBridge(HardwareSettings(), anonymize=False).sample()
        assert snapshot.hostname != "redacted"

    def test_backend_failure_raises_a_sensor_error(self) -> None:
        class Broken:
            name = "broken"

            def sample_all(self, *args):
                raise OSError("procfs gone")

            def capabilities(self):
                return []

            def library_version(self):
                return "0"

        loader = NativeLoader(prefer_native=False)
        loader.load()
        loader._backend = Broken()  # type: ignore[assignment]
        monitor = HardwareBridge(HardwareSettings(), loader=loader)
        with pytest.raises(SensorUnavailableError):
            monitor.sample()
        assert monitor.error_count == 1
        assert "procfs gone" in monitor.last_error


class TestFormatSnapshot:
    def test_renders_the_documented_lines(self) -> None:
        snapshot = HardwareSnapshot(
            cpu_percent=32.0,
            cpu_temp_c=51.0,
            ram_used_bytes=int(6.4 * 1024**3),
            ram_total_bytes=16 * 1024**3,
            gpu_percent=48.0,
            gpu_name="Test GPU",
            gpu_temp_c=54.0,
            disk_percent=62.0,
            net_rx_bytes_per_second=int(4.2 * 1024**2),
            net_tx_bytes_per_second=int(0.8 * 1024**2),
        )
        lines = format_snapshot(snapshot)
        assert lines[0] == "CPU        32%"
        assert lines[1] == "CPU Temp   51°C"
        assert "6.4 GB / 16.0 GB" in lines[2]
        assert "48%" in lines[3]
        assert "GPU Temp   54°C" in lines[4]
        assert "Disk       62%" in lines[5]
        assert "↓ 4.2 MB/s" in lines[6] and "↑ 819.2 KB/s" in lines[6]

    def test_missing_values_render_as_na(self) -> None:
        snapshot = HardwareSnapshot()
        lines = format_snapshot(snapshot)
        assert "n/a" in lines[1]
        assert "n/a" in lines[3]

    def test_report_lines_before_the_first_sample(self) -> None:
        bridge = HardwareBridge(HardwareSettings())
        assert "not produced" in bridge.report_lines()[0]

    def test_report_lines_after_sampling(self) -> None:
        bridge = HardwareBridge(HardwareSettings())
        bridge.sample()
        assert any(line.startswith("CPU") for line in bridge.report_lines())


class TestPrivacyHelpers:
    def test_redact_mapping_hides_the_hostname(self) -> None:
        payload = {"system": {"hostname": "my-laptop", "kernel": "Linux"}}
        assert redact_mapping(payload)["system"]["hostname"] == "redacted"

    def test_redact_mapping_walks_lists(self) -> None:
        payload = {"items": [{"hostname": "x"}]}
        assert redact_mapping(payload)["items"][0]["hostname"] == "redacted"

    def test_redact_mapping_ignores_non_mappings(self) -> None:
        assert redact_mapping([1, 2]) == [1, 2]  # type: ignore[arg-type]

    def test_scrub_text_removes_home_paths(self) -> None:
        assert "/home/alice" not in scrub_text("reading /home/alice/config.json")

    def test_scrub_text_remips_ips(self) -> None:
        assert "192.168.1.5" not in scrub_text("connected to 192.168.1.5")

    def test_scrub_text_can_keep_ips(self) -> None:
        assert "192.168.1.5" in scrub_text("connected to 192.168.1.5", scrub_ips=False)


class TestEnvironmentDescription:
    def test_reports_installed_and_missing_packages(self) -> None:
        env = describe_environment()
        assert "opencv" in env and "numpy" in env
        assert env["python"]
        assert Path(env["project_root"]).exists()

    def test_describe_includes_the_environment(self) -> None:
        info = HardwareBridge(HardwareSettings()).describe()
        assert "environment" in info
        assert info["environment"]["python"]
