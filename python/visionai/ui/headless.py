"""Text-only interface for servers, containers and CI.

Renders the same overlay as the desktop UI into a terminal-friendly summary, so
the pipeline can be exercised on a machine with no display server.
"""

from __future__ import annotations

import logging
import time

from visionai.ai.inference import PipelineResult
from visionai.ai.taxonomy import DISCLAIMER, describe_track
from visionai.config.settings import Settings
from visionai.hardware.hardware_bridge import HardwareBridge, format_snapshot
from visionai.pipeline.engine import PipelineEngine
from visionai.pipeline.stats import PerformanceReport

LOG = logging.getLogger(__name__)

CLEAR = "\033[2J\033[H"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLOURS = {
    "happy": "\033[92m",
    "joy": "\033[92m",
    "love": "\033[95m",
    "sad": "\033[94m",
    "fear": "\033[95m",
    "surprise": "\033[93m",
    "angry": "\033[91m",
    "disgust": "\033[93m",
    "neutral": "\033[90m",
}


def colour_for(label: str) -> str:
    return COLOURS.get(label, "")


def describe_result(result: PipelineResult, model_name: str, heuristic: bool) -> str:
    lines = [
        f"{BOLD}Detected facial expressions (AI estimates){RESET}",
        f"  frame {result.frame_number:>6}   "
        f"inference {result.total_ms:5.1f} ms   "
        f"(detect {result.detection_ms:.1f} / classify {result.classification_ms:.1f})",
    ]
    if not result.tracks:
        lines.append(f"  {DIM}no faces detected in this frame{RESET}")
    for track in result.tracks:
        expression = track.expression
        accepted = bool(expression and expression.is_confident)
        label = track.label if accepted else "below threshold"
        colour = colour_for(track.label) if accepted else ""
        text = describe_track(track.track_id, label, track.confidence if accepted else None)
        lines.append(f"  {colour}{text}{RESET}")
        if expression is not None and expression.is_heuristic:
            lines.append(f"    {DIM}untrained heuristic, not a model prediction{RESET}")
    suffix = " (heuristic fallback)" if heuristic else ""
    lines.append(f"  {DIM}expression model: {model_name}{suffix}{RESET}")
    return "\n".join(lines)


def render_text_frame(
    result: PipelineResult | None,
    bridge: HardwareBridge,
    model_name: str = "",
    heuristic: bool = False,
    colour: bool = True,
) -> str:
    """A single screen of status, ready to print."""
    def paint(text: str) -> str:
        return text if colour else _strip_ansi(text)

    sections = []
    if result is not None:
        sections.append(paint(describe_result(result, model_name, heuristic)))
    hardware = bridge.latest
    if hardware is not None:
        header = f"{BOLD}Hardware{RESET}"
        rows = "\n".join(f"  {line}" for line in format_snapshot(hardware))
        sections.append(paint(f"{header}\n{rows}"))
    else:
        sections.append(paint(f"{BOLD}Hardware{RESET}\n  waiting for the first sample"))
    sections.append(paint(f"{DIM}{DISCLAIMER}{RESET}"))
    return "\n\n".join(sections)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def run_headless(
    settings: Settings,
    source_spec: str = "synthetic",
    duration: float | None = None,
    frames: int | None = None,
    colour: bool = True,
    report: bool = True,
) -> int:
    """Run the pipeline in the terminal for ``duration`` seconds or ``frames``."""
    bridge = HardwareBridge(settings.hardware, anonymize=settings.privacy.anonymize_log_payloads)
    engine = PipelineEngine(settings)

    if settings.hardware.enabled:
        bridge.start()

    started = time.perf_counter()
    deadline = started + duration if duration else None
    processed = 0
    printed = False
    interval = max(0.1, 1.0 / max(1.0, settings.pipeline.inference_fps))

    try:
        engine.start(source_spec)
        if engine.state.value == "error":
            print(f"error: {engine.status.last_error}", flush=True)
            return 1
        print(
            f"{BOLD}Linux AI Vision - headless mode{RESET}\n"
            f"source={source_spec}  inference_fps={settings.pipeline.inference_fps}\n"
            f"press Ctrl+C to stop\n",
            flush=True,
        )
        while True:
            result = engine.latest()
            if result is not None:
                now = time.perf_counter()
                if now - started >= interval:
                    if colour:
                        print(CLEAR, end="")
                    print(
                        render_text_frame(
                            result,
                            bridge,
                            engine.pipeline.classifier_stage.classifier.name
                            if engine.pipeline
                            else "",
                            not engine.pipeline.uses_trained_expression_model
                            if engine.pipeline
                            else True,
                            colour,
                        ),
                        flush=True,
                    )
                    printed = True
                processed = result.frame_number
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if frames is not None and processed >= frames:
                break
            if engine.state.value == "error":
                print(f"\nerror: {engine.status.last_error}", flush=True)
                return 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        engine.close()
        bridge.stop()

    if report:
        print(_performance_summary(engine, bridge, started), flush=True)
    return 0 if printed else 1


def _performance_summary(
    engine: PipelineEngine, bridge: HardwareBridge, started: float
) -> str:
    if engine.pipeline is None:
        return "no performance data collected"
    snapshot = engine.pipeline.stats.snapshot()
    hardware = bridge.latest
    performance = PerformanceReport(
        frames=snapshot["frames"],
        duration_s=snapshot["elapsed_s"],
        average_fps=engine.pipeline.stats.observed_fps(),
        average_inference_ms=snapshot["inference_ms_avg"],
        p95_inference_ms=snapshot["inference_ms_p95"],
        max_inference_ms=snapshot["inference_ms_max"],
        average_detection_ms=snapshot["detection_ms_ema"],
        average_classification_ms=snapshot["classification_ms_ema"],
        faces_average=snapshot["faces_avg"],
        cpu_percent=hardware.cpu_percent if hardware else 0.0,
        ram_mb=(hardware.ram_used_bytes / 1024**2) if hardware else 0.0,
        gpu_percent=hardware.gpu_percent if hardware else None,
        gpu_temperature=hardware.gpu_temp_c if hardware else None,
    )
    del started
    return "\n" + performance.describe()
