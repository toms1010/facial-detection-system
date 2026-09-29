"""Pure-Python hardware backend.

Used when the C++ layer has not been built. It reads the same ``/proc`` and
``/sys`` files and returns the same dictionary shape as the native backends, so
nothing downstream needs to know which one is active.

It is intentionally simple rather than exhaustive: no per-core deltas beyond
what ``/proc/stat`` gives directly, and GPU utilisation is reported as
unavailable rather than guessed when the driver does not expose it.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import time
from pathlib import Path
from typing import Any

PROC_STAT = "/proc/stat"
PROC_MEMINFO = "/proc/meminfo"
PROC_NET_DEV = "/proc/net/dev"
PROC_LOADAVG = "/proc/loadavg"
PROC_UPTIME = "/proc/uptime"
SYS_HWMON = "/sys/class/hwmon"
SYS_THERMAL = "/sys/class/thermal"
SYS_CLASS_NET = "/sys/class/net"
SYS_CLASS_DRM = "/sys/class/drm"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _read_float(path: str) -> float | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def _read_int(path: str) -> int | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


class PythonBackend:
    """Fallback implementation of the native backend protocol."""

    name = "python"

    def __init__(self) -> None:
        self._previous_cpu: list[int] | None = None
        self._previous_cores: list[list[int]] = []
        self._previous_net: dict[str, tuple[int, int]] = {}
        self._last_net_time = 0.0
        self._sensors: list[tuple[str, str]] | None = None

    def library_version(self) -> str:
        return "py-1.0.0"

    def capabilities(self) -> list[str]:
        return ["cpu", "memory", "swap", "temperature", "hwmon", "disk", "network", "system"]

    def sample_all(self, disk_path: str, interface: str) -> dict[str, Any]:
        return {
            "cpu": self.sample_cpu(),
            "memory": self.sample_memory(),
            "temperature": self.sample_temperature(),
            "disk": self.sample_disk(disk_path),
            "network": self.sample_network(interface),
            "gpus": self.sample("gpus", "").get("gpus", []),
            "system": self.sample_system(),
            "sample_seconds": 0.0,
            "version": self.library_version(),
            "backend": "python",
        }

    def sample(self, subsystem: str, argument: str) -> dict[str, Any]:
        handlers = {
            "cpu": lambda: self.sample_cpu(),
            "memory": lambda: self.sample_memory(),
            "temperature": lambda: self.sample_temperature(),
            "disk": lambda: self.sample_disk(argument or "/"),
            "network": lambda: self.sample_network(argument or ""),
            "gpus": lambda: {"gpus": self._gpus()},
            "system": self.sample_system,
        }
        if subsystem not in handlers:
            raise ValueError(f"unknown subsystem {subsystem!r}")
        return handlers[subsystem]()

    def sample_cpu(self) -> dict[str, Any]:
        usage, user, system, idle, iowait, cores = self._cpu()
        load1, load5, load15 = self._loadavg()
        frequency = self._cpu_frequency()
        return {
            "usage_percent": usage,
            "user_percent": user,
            "system_percent": system,
            "idle_percent": idle,
            "iowait_percent": iowait,
            "frequency_mhz": frequency,
            "max_frequency_mhz": frequency,
            "load_average_1": load1,
            "load_average_5": load5,
            "load_average_15": load15,
            "core_count": max(1, os.cpu_count() or 1),
            "model_name": self._cpu_model(),
            "cores": cores,
        }

    def _cpu(self) -> tuple[float, float, float, float, float, list[dict[str, Any]]]:
        text = _read(PROC_STAT)
        cores: list[dict[str, Any]] = []
        if not text:
            return 0.0, 0.0, 0.0, 0.0, 0.0, cores

        total_fields: list[int] | None = None
        core_rows: list[list[int]] = []
        for line in text.splitlines():
            if line.startswith("cpu "):
                total_fields = [int(v) for v in line.split()[1:]]
            elif line.startswith("cpu") and len(line) > 3 and line[3].isdigit():
                core_rows.append([int(v) for v in line.split()[1:]])

        if total_fields is None or len(total_fields) < 4:
            return 0.0, 0.0, 0.0, 0.0, 0.0, cores

        cores = [
            {"index": i, "usage_percent": 0.0, "frequency_mhz": 0.0}
            for i in range(len(core_rows))
        ]
        self._previous_cores = core_rows

        previous = self._previous_cpu
        self._previous_cpu = total_fields
        if previous is None or len(previous) != len(total_fields):
            return 0.0, 0.0, 0.0, 0.0, 0.0, cores

        def delta(index: int) -> int:
            return total_fields[index] - previous[index] if index < len(previous) else 0

        total_delta = sum(delta(i) for i in range(len(total_fields)))
        idle_delta = delta(3) + (delta(4) if len(total_fields) > 4 else 0)
        if total_delta <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, cores

        user_delta = delta(0) + (delta(1) if len(total_fields) > 1 else 0)
        system_delta = delta(2)
        busy_delta = total_delta - idle_delta
        return (
            _clamp(busy_delta / total_delta * 100.0),
            _clamp(user_delta / total_delta * 100.0),
            _clamp(system_delta / total_delta * 100.0),
            _clamp(idle_delta / total_delta * 100.0),
            _clamp(delta(4) / total_delta * 100.0) if len(total_fields) > 4 else 0.0,
            cores,
        )

    def _cpu_model(self) -> str:
        text = _read("/proc/cpuinfo") or ""
        for line in text.splitlines():
            for key in ("model name", "Model", "Processor"):
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
        return platform.processor() or "Unknown CPU"

    def _cpu_frequency(self) -> float:
        for entry in sorted(Path(SYS_HWMON).glob("hwmon*")) if Path(SYS_HWMON).is_dir() else []:
            khz = _read_int(str(entry / "cpufreq1" / "cpuinfo_max_freq"))
            if khz:
                return khz / 1000.0
        for path in Path("/sys/devices/system/cpu/cpu0/cpufreq").glob("cpuinfo_max_freq"):
            khz = _read_int(str(path))
            if khz:
                return khz / 1000.0
        return 0.0

    def _loadavg(self) -> tuple[float, float, float]:
        text = _read(PROC_LOADAVG)
        if not text:
            return 0.0, 0.0, 0.0
        fields = text.split()
        try:
            return float(fields[0]), float(fields[1]), float(fields[2])
        except (IndexError, ValueError):
            return 0.0, 0.0, 0.0

    def sample_memory(self) -> dict[str, Any]:
        text = _read(PROC_MEMINFO)
        if not text:
            return _empty_memory()
        fields: dict[str, int] = {}
        for line in text.splitlines():
            # Lines look like "MemTotal:       24300848 kB".
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "kB" and parts[0].endswith(":"):
                try:
                    fields[parts[0].rstrip(":")] = int(parts[1]) * 1024
                except ValueError:
                    continue
        total = fields.get("MemTotal", 0)
        available = fields.get("MemAvailable", 0)
        if available == 0:
            available = fields.get("MemFree", 0) + fields.get("Cached", 0) + fields.get("Buffers", 0)
        used = max(0, total - available)
        swap_total = fields.get("SwapTotal", 0)
        swap_free = fields.get("SwapFree", 0)
        return {
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "free_bytes": fields.get("MemFree", 0),
            "cached_bytes": fields.get("Cached", 0),
            "buffers_bytes": fields.get("Buffers", 0),
            "swap_total_bytes": swap_total,
            "swap_free_bytes": swap_free,
            "usage_percent": _clamp(used / total * 100.0) if total else 0.0,
            "swap_usage_percent": _clamp(
                (swap_total - swap_free) / swap_total * 100.0) if swap_total else 0.0,
        }

    def sample_temperature(self) -> dict[str, Any]:
        sensors: list[dict[str, Any]] = []
        for label, path in self._temperature_probes():
            raw = _read(path)
            if raw is None:
                continue
            text = raw.strip()
            try:
                celsius = float(text) / 1000.0 if "." not in text else float(text)
            except ValueError:
                continue
            if not -40.0 <= celsius <= 150.0:
                continue
            sensors.append(
                {
                    "label": label,
                    "path": path,
                    "celsius": celsius,
                    "critical_celsius": 0.0,
                    "max_celsius": 0.0,
                }
            )
        cpu_source, cpu_celsius, hottest = "", 0.0, 0.0
        for sensor in sensors:
            hottest = max(hottest, sensor["celsius"])
            lowered = sensor["label"].lower()
            if cpu_celsius == 0.0 and any(
                key in lowered for key in ("pkg", "coretemp", "tctl", "tcpu", "k10temp", "zenpower")
            ):
                cpu_source, cpu_celsius = sensor["label"], sensor["celsius"]
        if cpu_celsius == 0.0 and sensors:
            best = max(sensors, key=lambda s: s["celsius"])
            cpu_source, cpu_celsius = best["label"], best["celsius"]
        return {
            "available": bool(sensors),
            "cpu_celsius": cpu_celsius,
            "max_celsius": hottest,
            "cpu_source": cpu_source,
            "sensors": sensors,
        }

    def _temperature_probes(self) -> list[tuple[str, str]]:
        if self._sensors is not None:
            return self._sensors
        probes: list[tuple[str, str]] = []
        for hwmon in sorted(Path(SYS_HWMON).glob("hwmon*")):
            chip = (_read(str(hwmon / "name")) or hwmon.name).strip()
            for entry in sorted(hwmon.glob("temp*_input")):
                name = entry.name.replace("_input", "")
                label = (_read(str(entry.with_name(f"{name}_label"))) or f"{chip}/{name}").strip()
                probes.append((label, str(entry)))
        for zone in sorted(Path(SYS_THERMAL).glob("thermal_zone*")):
            kind = (_read(str(zone / "type")) or zone.name).strip()
            probes.append((kind, str(zone / "temp")))
        self._sensors = probes
        return probes

    def sample_disk(self, path: str) -> dict[str, Any]:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            return {
                "available": False,
                "mount_point": path,
                "device": "",
                "filesystem": "",
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "usage_percent": 0.0,
            }
        return {
            "available": True,
            "mount_point": path,
            "device": self._device_for(path),
            "filesystem": "",
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "usage_percent": _clamp(usage.used / usage.total * 100.0) if usage.total else 0.0,
        }

    def _device_for(self, path: str) -> str:
        text = _read("/proc/self/mountinfo") or ""
        target = str(Path(path).resolve()) if Path(path).exists() else path
        best, best_len = "", -1
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            mount_point = fields[4]
            if not (mount_point == target or target.startswith(mount_point.rstrip("/") + "/")):
                continue
            if len(mount_point) > best_len and "-" in fields:
                separator = fields.index("-")
                if separator + 1 < len(fields):
                    best, best_len = fields[separator + 1], len(mount_point)
        return best

    def sample_network(self, preferred: str) -> dict[str, Any]:
        text = _read(PROC_NET_DEV) or ""
        now = time.monotonic()
        elapsed = now - self._last_net_time if self._last_net_time else 0.0
        interfaces: list[dict[str, Any]] = []
        current: dict[str, tuple[int, int]] = {}
        for line in text.splitlines():
            if ":" not in line or line.startswith("Inter-") or line.startswith(" face"):
                continue
            name, _, rest = line.partition(":")
            name = name.strip()
            fields = rest.split()
            if len(fields) < 16 or not name:
                continue
            try:
                rx, tx = int(fields[0]), int(fields[8])
            except ValueError:
                continue
            current[name] = (rx, tx)
            previous = self._previous_net.get(name)
            rx_rate = tx_rate = 0.0
            if previous and elapsed > 1e-6:
                rx_rate = max(0, rx - previous[0]) / elapsed
                tx_rate = max(0, tx - previous[1]) / elapsed
            state = (_read(f"{SYS_CLASS_NET}/{name}/operstate") or "unknown").strip()
            interfaces.append(
                {
                    "name": name,
                    "rx_bytes": rx,
                    "tx_bytes": tx,
                    "rx_bytes_per_second": rx_rate,
                    "tx_bytes_per_second": tx_rate,
                    "rx_packets_per_second": 0.0,
                    "tx_packets_per_second": 0.0,
                    "errors": 0,
                    "drops": 0,
                    "is_up": state != "down",
                }
            )
        self._previous_net = current
        self._last_net_time = now

        primary = preferred if any(i["name"] == preferred for i in interfaces) else ""
        if not primary:
            candidates = [i for i in interfaces if i["name"] != "lo"]
            best = max(candidates or interfaces, key=lambda i: i["rx_bytes"] + i["tx_bytes"], default=None)
            primary = best["name"] if best else ""
        selected = next((i for i in interfaces if i["name"] == primary), None)
        return {
            "primary_interface": primary,
            "rx_bytes_per_second": selected["rx_bytes_per_second"] if selected else 0.0,
            "tx_bytes_per_second": selected["tx_bytes_per_second"] if selected else 0.0,
            "interfaces": interfaces,
        }

    def _gpus(self) -> list[dict[str, Any]]:
        gpus: list[dict[str, Any]] = []
        drm = Path(SYS_CLASS_DRM)
        if not drm.is_dir():
            return gpus
        for index, card in enumerate(sorted(p for p in drm.glob("card*") if "-" not in p.name)):
            device = card / "device"
            if not device.exists():
                continue
            driver_link = device / "driver"
            driver = driver_link.resolve().name if driver_link.is_symlink() else ""
            vendor = "AMD" if "amdgpu" in driver else "Intel" if "i915" in driver else "Unknown"
            busy = _read_float(str(device / "gpu_busy_percent"))
            vram_total = _read_int(str(device / "mem_info_vram_total"))
            vram_used = _read_int(str(device / "mem_info_vram_used"))
            temp_path = device / "hwmon" / "hwmon0" / "temp1_input"
            temperature = _read_float(str(temp_path))
            if temperature is not None and temperature > 1000:
                temperature /= 1000.0
            gpus.append(
                {
                    "present": True,
                    "vendor": vendor,
                    "name": f"{vendor} GPU" + (f" ({driver})" if driver else ""),
                    "usage_percent": busy if busy is not None else 0.0,
                    "memory_usage_percent": (
                        vram_used / vram_total * 100.0 if vram_total and vram_used is not None else 0.0
                    ),
                    "memory_used_bytes": vram_used or 0,
                    "memory_total_bytes": vram_total or 0,
                    "temperature_celsius": temperature,
                    "power_watts": None,
                    "index": index,
                    "source": "python-sysfs",
                    "detail": "" if busy is not None else "driver exposes no utilisation counter",
                }
            )
        return gpus

    def sample_system(self) -> dict[str, Any]:
        uptime = 0.0
        text = _read(PROC_UPTIME)
        if text:
            try:
                uptime = float(text.split()[0])
            except (IndexError, ValueError):
                uptime = 0.0
        distribution = ""
        for line in (_read("/etc/os-release") or "").splitlines():
            if line.startswith("PRETTY_NAME="):
                distribution = line.split("=", 1)[1].strip().strip('"')
                break
        try:
            processes = sum(1 for entry in Path("/proc").iterdir() if entry.name.isdigit())
        except OSError:
            processes = 0
        return {
            "hostname": socket.gethostname(),
            "kernel": f"{platform.system()} {platform.release()}",
            "distribution": distribution,
            "uptime_seconds": uptime,
            "process_count": processes,
            "architecture": platform.machine(),
        }


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _empty_memory() -> dict[str, Any]:
    return {
        "total_bytes": 0,
        "used_bytes": 0,
        "available_bytes": 0,
        "free_bytes": 0,
        "cached_bytes": 0,
        "buffers_bytes": 0,
        "swap_total_bytes": 0,
        "swap_free_bytes": 0,
        "usage_percent": 0.0,
        "swap_usage_percent": 0.0,
    }
