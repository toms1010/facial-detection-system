# Performance

## Measured

On the development machine: 12-core Intel i5-12450H, 24 GB RAM, Intel i915
integrated graphics, no CUDA, Ubuntu 26.04, Python 3.14, OpenCV 5.0.

| Stage | Time |
|---|---|
| YuNet detection (640×480) | ~8–30 ms |
| Heuristic classification (7 faces) | < 1 ms |
| **Full pipeline** | **~14 ms per frame (≈26 FPS)** |
| Hardware snapshot (native) | ~4 ms |
| Hardware snapshot (pure Python) | ~9 ms |

> These were taken on an idle machine. Confirm with `visionai benchmark`; if
> your load average is high, expect several times these numbers. That is not a
> regression in the code.

## Why it is fast

- **Stale frames are dropped.** Capture publishes only the newest frame, so a
  slow consumer degrades in latency rather than accumulating a backlog.
- **Detection is resolution-capped.** `pipeline.processing_width` (default 960)
  downscales before inference; boxes are scaled back afterwards.
- **Inference is decoupled from rendering.** A slow model lowers the label
  update rate, not the frame rate of the window.
- **The hardware layer caches its expensive probe.** `nvidia-smi` is invoked at
  most once.
- **The overlay re-renders only when a new frame arrives**, not per timer tick.

## Tuning

| Knob | Default | Effect |
|---|---|---|
| `pipeline.inference_fps` | `15` | lower it to save CPU |
| `pipeline.processing_width` | `960` | lower to `640` for a large CPU saving |
| `models.imgsz` | `640` | YOLO only; `320` is usually fine for faces |
| `models.max_faces` | `16` | caps classification work per frame |
| `models.device` | `auto` | set `cuda` when a supported GPU is present |
| `models.half_precision` | `false` | FP16 on CUDA |
| `hardware.interval` | `1.0` | raise to 5.0 to sample less often |

## With a real model

A trained expression model is far more expensive than the heuristic: budget
10–40 ms per face on CPU, less on GPU. `ExpressionClassifier.supports_batching`
lets a backend submit a whole batch in one call, and the classification stage
uses it when a face is available.

## Profiling

```bash
visionai benchmark --frames 300 --json > bench.json
```

Reports average FPS, mean and p95 inference latency, per-stage timings, CPU,
RAM and GPU usage. `--source 0` benchmarks the real camera path.
