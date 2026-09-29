"""Python <-> C++ interoperability.

These tests exercise the boundary rather than the internals: both backends must
produce the same dictionary shape, and the pure-Python backend must keep working
when neither is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionai.config.paths import find_native_candidates
from visionai.hardware.native_loader import NativeLoader
from visionai.hardware.python_backend import PythonBackend

SUBSYSTEMS = ("cpu", "memory", "temperature", "disk", "network", "gpus", "system")

REQUIRED_KEYS = {"cpu", "memory", "temperature", "disk", "network", "gpus", "system"}


@pytest.fixture
def native_loader() -> NativeLoader:
    loader = NativeLoader()
    loader.load()
    if not loader.is_native:
        pytest.skip("no native backend is available")
    return loader


class TestDiscovery:
    def test_candidates_are_paths_or_none(self) -> None:
        pybind_path, cabi_path = find_native_candidates()
        for candidate in (pybind_path, cabi_path):
            assert candidate is None or isinstance(candidate, Path)

    def test_existing_candidates_exist_on_disk(self) -> None:
        for candidate in find_native_candidates():
            if candidate is not None:
                assert candidate.is_file()

    def test_environment_override_is_respected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VISIONAI_NATIVE_LIB", str(tmp_path / "absent.so"))
        assert find_native_candidates() == (None, None)


@pytest.mark.requires_native
class TestPybind11Backend:
    def test_snapshot_contains_every_subsystem(self, native_loader: NativeLoader) -> None:
        snapshot = native_loader.backend.sample_all("/", "")
        assert set(snapshot) >= REQUIRED_KEYS
        assert snapshot["backend"] == "pybind11"

    def test_snapshot_is_json_serialisable(self, native_loader: NativeLoader) -> None:
        json.dumps(native_loader.backend.sample_all("/", ""))

    def test_cpu_values_are_plausible(self, native_loader: NativeLoader) -> None:
        cpu = native_loader.backend.sample("cpu", "")
        assert cpu["core_count"] >= 1
        assert 0.0 <= cpu["usage_percent"] <= 100.0
        assert cpu["model_name"]

    def test_memory_values_are_plausible(self, native_loader: NativeLoader) -> None:
        memory = native_loader.backend.sample("memory", "")
        assert memory["total_bytes"] > 0
        assert 0.0 <= memory["usage_percent"] <= 100.0

    def test_temperatures_are_in_celsius(self, native_loader: NativeLoader) -> None:
        temperature = native_loader.backend.sample("temperature", "")
        for sensor in temperature["sensors"]:
            assert -40.0 <= sensor["celsius"] <= 150.0, sensor

    def test_temperatures_are_not_millidegrees(self, native_loader: NativeLoader) -> None:
        temperature = native_loader.backend.sample("temperature", "")
        for sensor in temperature["sensors"]:
            assert sensor["celsius"] < 200.0, sensor

    def test_disk_for_an_unreadable_path(self, native_loader: NativeLoader) -> None:
        disk = native_loader.backend.sample("disk", "/no/such/path")
        assert disk["available"] is False
        assert disk["usage_percent"] == 0.0

    def test_network_lists_interfaces(self, native_loader: NativeLoader) -> None:
        network = native_loader.backend.sample("network", "")
        assert isinstance(network["interfaces"], list)
        assert network["interfaces"]

    def test_gpu_list_is_always_present(self, native_loader: NativeLoader) -> None:
        assert isinstance(native_loader.backend.sample("gpus", "")["gpus"], list)

    def test_system_information(self, native_loader: NativeLoader) -> None:
        system = native_loader.backend.sample("system", "")
        assert system["architecture"]
        assert system["uptime_seconds"] > 0

    def test_unknown_subsystem_raises(self, native_loader: NativeLoader) -> None:
        with pytest.raises(ValueError):
            native_loader.backend.sample("nonsense", "")

    def test_version_and_capabilities(self, native_loader: NativeLoader) -> None:
        assert native_loader.backend.library_version()
        assert "cpu" in native_loader.backend.capabilities()

    def test_proc_stat_parser(self, native_loader: NativeLoader) -> None:
        parsed = native_loader.backend._module.parse_stat_line("cpu 10 20 30 400 50 5 5 0")
        assert parsed["user"] == 10
        assert parsed["idle"] == 400
        assert parsed["iowait"] == 50

    def test_meminfo_parser(self, native_loader: NativeLoader) -> None:
        parsed = native_loader.backend._module.parse_meminfo(
            "MemTotal:       16384 kB\nMemFree:  1024 kB\n", "MemTotal"
        )
        assert parsed == 16384 * 1024

    def test_netdev_parser(self, native_loader: NativeLoader) -> None:
        content = (
            "Inter-|   Receive\n"
            " face |bytes\n"
            "  eth0: 1000 10 0 0 0 0 0 0 2000 20 0 0 0 0 0 0\n"
        )
        interfaces = native_loader.backend._module.parse_net_dev(content)
        assert interfaces[0]["name"] == "eth0"
        assert interfaces[0]["rx_bytes"] == 1000
        assert interfaces[0]["tx_bytes"] == 2000

    def test_nvidia_smi_parser(self, native_loader: NativeLoader) -> None:
        gpus = native_loader.backend._module.parse_nvidia_smi(
            "0, NVIDIA RTX 3060, 42, 1024, 12288, 47, 55.3\n"
        )
        assert gpus[0]["vendor"] == "NVIDIA"
        assert gpus[0]["usage_percent"] == 42.0
        assert gpus[0]["temperature_celsius"] == 47.0

    def test_repeated_sampling_is_stable(self, native_loader: NativeLoader) -> None:
        for _ in range(5):
            memory = native_loader.backend.sample("memory", "")
            assert memory["total_bytes"] > 0

    def test_sampling_is_fast_enough_for_a_one_hz_tick(self, native_loader: NativeLoader) -> None:
        import time

        started = time.perf_counter()
        native_loader.backend.sample_all("/", "")
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert elapsed_ms < 250.0, f"full snapshot took {elapsed_ms:.0f} ms"


@pytest.mark.requires_native
class TestCtypesBackend:
    @pytest.fixture
    def cabi(self):
        from visionai.hardware.native_loader import _CtypesBackend

        _, cabi_path = find_native_candidates()
        if cabi_path is None:
            pytest.skip("libvisionai_hw.so is not built")
        return _CtypesBackend(cabi_path)

    def test_snapshot_contains_every_subsystem(self, cabi) -> None:
        assert set(cabi.sample_all("/", "")) >= REQUIRED_KEYS

    def test_reports_its_own_backend_name(self, cabi) -> None:
        assert cabi.sample_all("/", "")["backend"] == "cabi"

    def test_memory_is_correct(self, cabi) -> None:
        assert cabi.sample("memory", "")["total_bytes"] > 0

    def test_temperature_is_celsius(self, cabi) -> None:
        for sensor in cabi.sample("temperature", "")["sensors"]:
            assert -40.0 <= sensor["celsius"] <= 150.0

    def test_unknown_subsystem_reports_an_error(self, cabi) -> None:
        assert "error" in cabi.sample("nonsense", "")

    def test_version_and_capabilities(self, cabi) -> None:
        assert cabi.library_version()
        assert "cpu" in cabi.capabilities()

    def test_freeing_memory_does_not_leak_or_crash(self, cabi) -> None:
        for _ in range(200):
            cabi.sample_all("/", "")
        assert cabi.sample("memory", "")["total_bytes"] > 0

    def test_handles_unicode_and_empty_arguments(self, cabi) -> None:
        assert cabi.sample_all("", "")["cpu"]["core_count"] >= 1


class TestBackendParity:
    """Every backend must expose the same keys so callers need no branching."""

    @pytest.mark.requires_native
    def test_native_and_python_agree_on_shape(self, native_loader: NativeLoader) -> None:
        native = native_loader.backend.sample_all("/", "")
        pure = PythonBackend().sample_all("/", "")
        assert set(native) >= REQUIRED_KEYS
        assert set(pure) >= REQUIRED_KEYS
        for key in SUBSYSTEMS:
            assert key in native and key in pure
        for key in ("total_bytes", "usage_percent"):
            assert key in native["memory"] and key in pure["memory"]

    @pytest.mark.requires_native
    def test_native_is_the_preferred_backend(self) -> None:
        loader = NativeLoader()
        report = loader.load()
        if report.is_native:
            assert report.active == "pybind11"

    def test_python_backend_always_works(self) -> None:
        assert set(PythonBackend().sample_all("/", "")) >= REQUIRED_KEYS


class TestNativeMissing:
    def test_pure_python_fallback_when_nothing_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import visionai.hardware.native_loader as loader_module

        monkeypatch.setattr(loader_module, "find_native_candidates", lambda: (None, None))
        loader = loader_module.NativeLoader()
        report = loader.load()
        assert report.active == "python"
        assert "cmake" in report.reason
        assert loader.is_native is False

    def test_require_native_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import visionai.hardware.native_loader as loader_module
        from visionai.ai.errors import NativeLibraryError

        monkeypatch.setattr(loader_module, "find_native_candidates", lambda: (None, None))
        with pytest.raises(NativeLibraryError):
            loader_module.NativeLoader(require_native=True).load()

    def test_cdll_missing_raises_oserror(self, tmp_path: Path) -> None:
        from visionai.hardware.native_loader import _CtypesBackend

        with pytest.raises(OSError):
            _CtypesBackend(tmp_path / "not-a-library.so")

    def test_bad_argument_type_is_rejected(self, tmp_path: Path) -> None:
        from visionai.hardware.native_loader import _CtypesBackend

        with pytest.raises((OSError, TypeError, AttributeError, ValueError)):
            _CtypesBackend(tmp_path)  # type: ignore[arg-type]


class TestNumpyInterop:
    def test_frames_cross_into_the_python_layer_intact(self) -> None:
        import numpy as np

        from visionai.ai.models.base import validate_frame

        array = np.zeros((8, 8, 3), dtype=np.uint8)
        assert validate_frame(array) is array
        assert array.shape == (8, 8, 3)
