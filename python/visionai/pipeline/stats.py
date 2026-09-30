"""Rolling performance statistics.

Rates use an exponentially weighted moving average so the displayed FPS reacts
quickly but does not flicker, while the raw history ring keeps exact values for
the performance report.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Sample:
    timestamp: float
    total_ms: float
    detection_ms: float
    classification_ms: float
    face_count: int


class PipelineStats:
    """Thread-safe FPS and latency accounting for the inference path."""

    def __init__(self, window: int = 120, inference_fps: float = 15.0) -> None:
        self.window = int(max(10, window))
        self.target_inference_fps = float(max(0.1, inference_fps))
        self._samples: deque[_Sample] = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._ema_total = 0.0
        self._ema_detection = 0.0
        self._ema_classification = 0.0
        self._ema_fps = 0.0
        self._last_timestamp: float | None = None
        self._error_count = 0
        self._frame_count = 0
        self._started = time.perf_counter()

    def record(
        self,
        total_ms: float,
        detection_ms: float = 0.0,
        classification_ms: float = 0.0,
        face_count: int = 0,
    ) -> None:
        now = time.perf_counter()
        with self._lock:
            self._samples.append(
                _Sample(now, float(total_ms), float(detection_ms), float(classification_ms), int(face_count))
            )
            self._frame_count += 1
            alpha = 0.2
            self._ema_total = _ema(self._ema_total, float(total_ms), alpha)
            self._ema_detection = _ema(self._ema_detection, float(detection_ms), alpha)
            self._ema_classification = _ema(self._ema_classification, float(classification_ms), alpha)
            if self._last_timestamp is not None:
                delta = now - self._last_timestamp
                if delta > 1e-6:
                    self._ema_fps = _ema(self._ema_fps, 1.0 / delta, alpha)
            self._last_timestamp = now

    def record_error(self, message: str = "") -> None:
        with self._lock:
            self._error_count += 1
        if message:
            from visionai.utils.logging_setup import log_throttled

            log_throttled("pipeline-error", f"pipeline error: {message}", interval=30.0)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._ema_total = 0.0
            self._ema_detection = 0.0
            self._ema_classification = 0.0
            self._ema_fps = 0.0
            self._last_timestamp = None
            self._error_count = 0
            self._frame_count = 0
            self._started = time.perf_counter()

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def error_count(self) -> int:
        with self._lock:
            return self._error_count

    def average(self, attribute: str) -> float:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return 0.0
        return sum(getattr(s, attribute) for s in samples) / len(samples)

    def percentile(self, attribute: str, fraction: float) -> float:
        with self._lock:
            values = sorted(getattr(s, attribute) for s in self._samples)
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
        return values[index]

    def observed_fps(self) -> float:
        with self._lock:
            samples = list(self._samples)
        if len(samples) < 2:
            return 0.0
        span = samples[-1].timestamp - samples[0].timestamp
        return (len(samples) - 1) / span if span > 1e-6 else 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            frame_count, errors = self._frame_count, self._error_count
            ema = (self._ema_total, self._ema_detection, self._ema_classification, self._ema_fps)
        latencies = [s.total_ms for s in samples]
        return {
            "frames": frame_count,
            "errors": errors,
            "window": len(samples),
            "inference_fps_ema": round(ema[3], 2),
            "inference_ms_ema": round(ema[0], 2),
            "detection_ms_ema": round(ema[1], 2),
            "classification_ms_ema": round(ema[2], 2),
            "inference_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "inference_ms_p95": round(self.percentile("total_ms", 0.95), 2),
            "inference_ms_max": round(max(latencies), 2) if latencies else 0.0,
            "faces_avg": round(self.average("face_count"), 2),
            "elapsed_s": round(time.perf_counter() - self._started, 2),
        }

    def history(self, attribute: str = "total_ms") -> list[tuple[float, float]]:
        with self._lock:
            return [(s.timestamp, getattr(s, attribute)) for s in self._samples]


def _ema(previous: float, current: float, alpha: float) -> float:
    return current if previous <= 0 else (1.0 - alpha) * previous + alpha * current


@dataclass
class PerformanceReport:
    """Aggregate view used by the benchmark command and the performance test."""

    frames: int = 0
    duration_s: float = 0.0
    average_fps: float = 0.0
    average_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    max_inference_ms: float = 0.0
    average_detection_ms: float = 0.0
    average_classification_ms: float = 0.0
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    ram_label: str = "System RAM used"
    gpu_percent: float | None = None
    gpu_temperature: float | None = None
    faces_average: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            f"Frames processed : {self.frames}",
            f"Duration         : {self.duration_s:.2f} s",
            f"Average FPS      : {self.average_fps:.2f}",
            f"Avg inference    : {self.average_inference_ms:.2f} ms",
            f"P95 inference    : {self.p95_inference_ms:.2f} ms",
            f"Max inference    : {self.max_inference_ms:.2f} ms",
            f"Avg detection    : {self.average_detection_ms:.2f} ms",
            f"Avg classifier   : {self.average_classification_ms:.2f} ms",
            f"Average faces    : {self.faces_average:.2f}",
            f"CPU usage        : {self.cpu_percent:.1f} %",
            f"{self.ram_label:<17}: {self.ram_mb:.1f} MB",
        ]
        if self.gpu_percent is not None:
            lines.append(f"GPU usage        : {self.gpu_percent:.1f} %")
        if self.gpu_temperature is not None:
            lines.append(f"GPU temperature  : {self.gpu_temperature:.1f} C")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "duration_s": round(self.duration_s, 3),
            "average_fps": round(self.average_fps, 2),
            "average_inference_ms": round(self.average_inference_ms, 2),
            "p95_inference_ms": round(self.p95_inference_ms, 2),
            "max_inference_ms": round(self.max_inference_ms, 2),
            "average_detection_ms": round(self.average_detection_ms, 2),
            "average_classification_ms": round(self.average_classification_ms, 2),
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_mb": round(self.ram_mb, 1),
            "ram_label": self.ram_label,
            "gpu_percent": self.gpu_percent,
            "gpu_temperature": self.gpu_temperature,
            "faces_average": round(self.faces_average, 2),
            **self.extras,
        }
