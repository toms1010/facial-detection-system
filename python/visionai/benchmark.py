"""Throughput and latency benchmark used by the CLI and the performance test."""

from __future__ import annotations

import logging
import time

import numpy as np

from visionai.ai.inference import InferencePipeline
from visionai.ai.models.registry import ModelRegistry
from visionai.config.paths import PackagePaths
from visionai.config.settings import Settings
from visionai.pipeline.stats import PerformanceReport

LOG = logging.getLogger(__name__)


def run_benchmark(
    settings: Settings,
    source_spec: str = "synthetic",
    frames: int = 200,
    warmup: int = 10,
) -> PerformanceReport:
    """Process ``frames`` frames and report throughput and resource usage."""
    from visionai.camera.camera_manager import open_source
    from visionai.hardware.hardware_bridge import HardwareBridge

    paths = PackagePaths.discover()
    registry = ModelRegistry(paths.face_models, paths.emotion_models)
    detector, detector_report = registry.build_detector(settings)
    classifier, classifier_report = registry.build_classifier(settings)
    pipeline = InferencePipeline(
        settings, registry, detector, classifier, detector_report, classifier_report
    )

    source = open_source(source_spec, settings.camera)
    source.open()
    bridge = HardwareBridge(settings.hardware, anonymize=True)
    try:
        bridge.sample()
    except Exception as exc:  # noqa: BLE001
        LOG.debug("hardware sampling unavailable during benchmark: %s", exc)

    cpu_before = _process_cpu_seconds()
    try:
        bridge.sample()
    except Exception as exc:  # noqa: BLE001
        LOG.debug("hardware sampling unavailable during benchmark: %s", exc)

    latencies: list[float] = []
    detection: list[float] = []
    classification: list[float] = []
    face_counts: list[int] = []
    started = time.perf_counter()

    try:
        frame = source.read()
        for _ in range(min(warmup, frames)):
            if frame is None:
                break
            pipeline.process(frame)
            frame = source.read()

        started = time.perf_counter()
        for _ in range(frames):
            if frame is None:
                break
            result = pipeline.process(frame)
            latencies.append(result.total_ms)
            detection.append(result.detection_ms)
            classification.append(result.classification_ms)
            face_counts.append(result.face_count)
            frame = source.read()
        duration = max(1e-6, time.perf_counter() - started)
    finally:
        source.release()
        pipeline.close()

    cpu_after = _process_cpu_seconds()
    ram_bytes = _ram_bytes()
    try:
        hardware = bridge.sample()
        cpu_percent = hardware.cpu_percent
        gpu_percent = hardware.gpu_percent
        gpu_temperature = hardware.gpu_temp_c
    except Exception:  # noqa: BLE001
        cpu_percent, gpu_percent, gpu_temperature = 0.0, None, None

    processed = len(latencies)
    sorted_latencies = sorted(latencies)
    p95 = (
        sorted_latencies[min(len(sorted_latencies) - 1, int(0.95 * len(sorted_latencies)))]
        if sorted_latencies
        else 0.0
    )
    return PerformanceReport(
        frames=processed,
        duration_s=duration,
        average_fps=(processed / duration) if processed else 0.0,
        average_inference_ms=float(np.mean(latencies)) if latencies else 0.0,
        p95_inference_ms=p95,
        max_inference_ms=max(latencies) if latencies else 0.0,
        average_detection_ms=float(np.mean(detection)) if detection else 0.0,
        average_classification_ms=float(np.mean(classification)) if classification else 0.0,
        cpu_percent=cpu_percent,
        ram_mb=ram_bytes / 1024**2,
        gpu_percent=gpu_percent,
        gpu_temperature=gpu_temperature,
        faces_average=float(np.mean(face_counts)) if face_counts else 0.0,
        extras={
            "source": source_spec,
            "detector": detector_report.active,
            "classifier": classifier_report.active,
            "trained_expression_model": pipeline.uses_trained_expression_model,
            "process_cpu_seconds": round(cpu_after - cpu_before, 3),
        },
    )


def _process_cpu_seconds() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _ram_bytes() -> int:
    """Resident set size, read from procfs so no extra dependency is needed."""
    import os
    from pathlib import Path

    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    except (OSError, IndexError, ValueError):
        return 0
    return pages * os.sysconf("SC_PAGE_SIZE")
