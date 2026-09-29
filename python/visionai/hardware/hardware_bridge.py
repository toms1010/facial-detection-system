"""Unified hardware monitoring API used by the UI and the CLI.

The bridge owns backend selection, sampling cadence, history and the display
formatting required by the spec. It never lets a missing sensor or a missing
native library take the application down: unavailable values are reported as
``None`` and explained in :attr:`HardwareSnapshot.notes`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from visionai.ai.errors import SensorUnavailableError
from visionai.config.settings import HardwareSettings
from visionai.hardware.native_loader import NativeLoader, describe_environment
from visionai.hardware.privacy import redact_snapshot

LOG = logging.getLogger(__name__)


@dataclass
class HardwareSnapshot:
    """A normalised view over whichever backend produced the data."""

    timestamp: float = field(default_factory=time.time)
    backend: str = "unknown"
    version: str = "0.0.0"

    cpu_percent: float = 0.0
    cpu_user_percent: float = 0.0
    cpu_system_percent: float = 0.0
    cpu_frequency_mhz: float = 0.0
    cpu_cores: int = 0
    cpu_model: str = ""
    load_average: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_core: list[dict[str, Any]] = field(default_factory=list)

    ram_total_bytes: int = 0
    ram_used_bytes: int = 0
    ram_percent: float = 0.0
    swap_total_bytes: int = 0
    swap_percent: float = 0.0

    cpu_temp_c: float | None = None
    gpu_temp_c: float | None = None
    temperature_sensors: list[dict[str, Any]] = field(default_factory=list)
    temperature_source: str = ""

    gpu_percent: float | None = None
    gpu_name: str = ""
    gpu_vendor: str = ""
    gpu_memory_percent: float | None = None
    gpu_memory_used_bytes: int = 0
    gpu_memory_total_bytes: int = 0
    gpu_count: int = 0
    gpu_detail: str = ""

    disk_percent: float = 0.0
    disk_total_bytes: int = 0
    disk_free_bytes: int = 0
    disk_path: str = "/"

    net_rx_bytes_per_second: float = 0.0
    net_tx_bytes_per_second: float = 0.0
    net_interface: str = ""
    net_interfaces: list[dict[str, Any]] = field(default_factory=list)

    hostname: str = ""
    kernel: str = ""
    distribution: str = ""
    uptime_seconds: float = 0.0

    sample_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0

    @property
    def has_temperature(self) -> bool:
        return self.cpu_temp_c is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "backend": self.backend,
            "version": self.version,
            "cpu_percent": round(self.cpu_percent, 1),
            "cpu_user_percent": round(self.cpu_user_percent, 1),
            "cpu_system_percent": round(self.cpu_system_percent, 1),
            "cpu_frequency_mhz": round(self.cpu_frequency_mhz, 1),
            "cpu_cores": self.cpu_cores,
            "cpu_model": self.cpu_model,
            "load_average": [round(v, 2) for v in self.load_average],
            "per_core": self.per_core,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_used_bytes": self.ram_used_bytes,
            "ram_percent": round(self.ram_percent, 1),
            "swap_total_bytes": self.swap_total_bytes,
            "swap_percent": round(self.swap_percent, 1),
            "cpu_temp_c": self.cpu_temp_c,
            "gpu_temp_c": self.gpu_temp_c,
            "temperature_sensors": self.temperature_sensors,
            "temperature_source": self.temperature_source,
            "gpu_percent": self.gpu_percent,
            "gpu_name": self.gpu_name,
            "gpu_vendor": self.gpu_vendor,
            "gpu_memory_percent": self.gpu_memory_percent,
            "gpu_count": self.gpu_count,
            "gpu_detail": self.gpu_detail,
            "disk_percent": round(self.disk_percent, 1),
            "disk_total_bytes": self.disk_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "disk_path": self.disk_path,
            "net_rx_bytes_per_second": self.net_rx_bytes_per_second,
            "net_tx_bytes_per_second": self.net_tx_bytes_per_second,
            "net_interface": self.net_interface,
            "net_interfaces": self.net_interfaces,
            "hostname": self.hostname,
            "kernel": self.kernel,
            "distribution": self.distribution,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "sample_ms": round(self.sample_ms, 2),
            "notes": self.notes,
        }


class HardwareBridge:
    """Samples hardware telemetry on a background thread."""

    def __init__(
        self,
        settings: HardwareSettings | None = None,
        loader: NativeLoader | None = None,
        anonymize: bool = True,
    ) -> None:
        self.settings = (settings or HardwareSettings()).validate()
        self.loader = loader or NativeLoader(prefer_native=self.settings.prefer_native)
        self.anonymize = anonymize
        self._snapshot: HardwareSnapshot | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: deque[HardwareSnapshot] = deque(maxlen=self.settings.history_size)
        self._error_count = 0
        self._last_error = ""

    @property
    def backend_name(self) -> str:
        return self.loader.load().active

    @property
    def is_native(self) -> bool:
        return self.loader.is_native

    def sample(self) -> HardwareSnapshot:
        """Take one synchronous sample of every subsystem."""
        started = time.perf_counter()
        report = self.loader.load()
        backend = self.loader.backend
        notes: list[str] = []
        if backend is None:
            raise SensorUnavailableError("no hardware backend available")

        try:
            raw = backend.sample_all(
                self.settings.disk_path, self.settings.network_interface or ""
            )
        except Exception as exc:  # noqa: BLE001
            self._error_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            LOG.warning("hardware sample failed: %s", exc)
            raise SensorUnavailableError(self._last_error) from exc

        snapshot = self._normalise(raw, report.active, str(raw.get("version", report.version)), notes)
        snapshot.sample_ms = (time.perf_counter() - started) * 1000.0
        if not report.is_native:
            notes.append(
                "Native C++ layer unavailable; using the pure-Python reader. "
                "Build it with `cmake -S . -B cpp/build && cmake --build cpp/build`."
            )
        if self.anonymize:
            snapshot = redact_snapshot(snapshot)

        with self._lock:
            self._snapshot = snapshot
            self._history.append(snapshot)
        return snapshot

    def _normalise(
        self, raw: dict[str, Any], backend: str, version: str, notes: list[str]
    ) -> HardwareSnapshot:
        snapshot = HardwareSnapshot(backend=backend, version=version)
        cpu = raw.get("cpu") or {}
        snapshot.cpu_percent = _f(cpu.get("usage_percent"))
        snapshot.cpu_user_percent = _f(cpu.get("user_percent"))
        snapshot.cpu_system_percent = _f(cpu.get("system_percent"))
        snapshot.cpu_frequency_mhz = _f(cpu.get("frequency_mhz"))
        snapshot.cpu_cores = int(_f(cpu.get("core_count")))
        snapshot.cpu_model = str(cpu.get("model_name") or "")
        snapshot.load_average = (
            _f(cpu.get("load_average_1")),
            _f(cpu.get("load_average_5")),
            _f(cpu.get("load_average_15")),
        )
        snapshot.per_core = list(cpu.get("cores") or [])

        memory = raw.get("memory") or {}
        snapshot.ram_total_bytes = int(_f(memory.get("total_bytes")))
        snapshot.ram_used_bytes = int(_f(memory.get("used_bytes")))
        snapshot.ram_percent = _f(memory.get("usage_percent"))
        snapshot.swap_total_bytes = int(_f(memory.get("swap_total_bytes")))
        snapshot.swap_percent = _f(memory.get("swap_usage_percent"))

        temperature = raw.get("temperature") or {}
        if temperature.get("available"):
            snapshot.cpu_temp_c = _f(temperature.get("cpu_celsius")) or None
            snapshot.temperature_source = str(temperature.get("cpu_source") or "")
        else:
            notes.append("No CPU temperature sensor is exposed on this machine.")
        snapshot.temperature_sensors = list(temperature.get("sensors") or [])

        gpus = list(raw.get("gpus") or [])
        snapshot.gpu_count = len(gpus)
        if gpus:
            primary = gpus[0]
            snapshot.gpu_vendor = str(primary.get("vendor") or "")
            snapshot.gpu_name = str(primary.get("name") or "")
            snapshot.gpu_percent = _f(primary.get("usage_percent"))
            snapshot.gpu_memory_percent = _f(primary.get("memory_usage_percent")) or None
            snapshot.gpu_memory_used_bytes = int(_f(primary.get("memory_used_bytes")))
            snapshot.gpu_memory_total_bytes = int(_f(primary.get("memory_total_bytes")))
            temperature_c = primary.get("temperature_celsius")
            snapshot.gpu_temp_c = _f(temperature_c) if temperature_c is not None else None
            snapshot.gpu_detail = str(primary.get("detail") or "")
            if snapshot.gpu_detail:
                notes.append(f"GPU: {snapshot.gpu_detail}")
        else:
            notes.append("No discrete GPU telemetry available; CPU inference is in use.")

        disk = raw.get("disk") or {}
        if disk.get("available"):
            snapshot.disk_percent = _f(disk.get("usage_percent"))
            snapshot.disk_total_bytes = int(_f(disk.get("total_bytes")))
            snapshot.disk_free_bytes = int(_f(disk.get("free_bytes")))
            snapshot.disk_path = str(disk.get("mount_point") or self.settings.disk_path)
        else:
            notes.append(f"Disk usage for {self.settings.disk_path} is not readable.")

        network = raw.get("network") or {}
        snapshot.net_rx_bytes_per_second = _f(network.get("rx_bytes_per_second"))
        snapshot.net_tx_bytes_per_second = _f(network.get("tx_bytes_per_second"))
        snapshot.net_interface = str(network.get("primary_interface") or "")
        snapshot.net_interfaces = list(network.get("interfaces") or [])

        system = raw.get("system") or {}
        snapshot.kernel = str(system.get("kernel") or "")
        snapshot.distribution = str(system.get("distribution") or "")
        snapshot.uptime_seconds = _f(system.get("uptime_seconds"))
        snapshot.hostname = str(system.get("hostname") or "")
        return snapshot

    @property
    def latest(self) -> HardwareSnapshot | None:
        with self._lock:
            return self._snapshot

    def history(self, attribute: str = "cpu_percent") -> list[tuple[float, float]]:
        with self._lock:
            return [(s.timestamp, float(getattr(s, attribute))) for s in self._history]

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def last_error(self) -> str:
        return self._last_error

    def start(self) -> None:
        """Begin sampling in the background."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hardware-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except SensorUnavailableError as exc:
                LOG.debug("hardware sample skipped: %s", exc)
            self._stop.wait(self.settings.interval)

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def report_lines(self) -> list[str]:
        """The exact label/value lines the Hardware panel renders."""
        snapshot = self.latest
        if snapshot is None:
            return ["Hardware monitoring has not produced a sample yet"]
        return format_snapshot(snapshot)

    def describe(self) -> dict[str, Any]:
        report = self.loader.report
        return {
            "backend": report.to_dict(),
            "environment": describe_environment(),
            "enabled": self.settings.enabled,
            "interval": self.settings.interval,
            "errors": self._error_count,
            "last_error": self._last_error or None,
        }

    def __enter__(self) -> HardwareBridge:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def format_snapshot(snapshot: HardwareSnapshot) -> list[str]:
    """Render a snapshot as the ``CPU 32%`` style lines from the spec."""
    from visionai.hardware.formatting import (
        format_bytes,
        format_rate,
        format_temperature,
    )

    lines = [f"CPU        {snapshot.cpu_percent:.0f}%"]
    lines.append(
        f"CPU Temp   {format_temperature(snapshot.cpu_temp_c)}"
    )
    lines.append(
        f"RAM        {format_bytes(snapshot.ram_used_bytes)} / {format_bytes(snapshot.ram_total_bytes)}"
    )
    gpu_text = f"{snapshot.gpu_percent:.0f}%" if snapshot.gpu_percent is not None else "n/a"
    lines.append(f"GPU        {gpu_text}  {snapshot.gpu_name}".rstrip())
    if snapshot.gpu_temp_c is not None:
        lines.append(f"GPU Temp   {snapshot.gpu_temp_c:.0f}°C")
    lines.append(f"Disk       {snapshot.disk_percent:.0f}%")
    lines.append(
        f"Network    ↓ {format_rate(snapshot.net_rx_bytes_per_second)}"
        f"  ↑ {format_rate(snapshot.net_tx_bytes_per_second)}"
    )
    return lines


def _f(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if result == result else 0.0
