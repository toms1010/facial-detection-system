# Architecture

## Layers

```
                 PySide6 desktop UI            (never runs inference)
                          |
                 PipelineEngine (worker thread)
                          |
        +--------+---------+---------+------------------+
        |        |         |         |                  |
   Detector  Classifier Tracker  Overlay          HardwareBridge
  (YuNet/    (heuristic/ (IoU)   (OpenCV,          (pybind11 -> ctypes
   Haar/     ONNX/TS)             testable         -> pure Python)
   YOLO)                                                |
        |                                                 |
     OpenCV frames                                   C++17 hardware layer
   (drop-stale)                                    /proc, /sys, statvfs
```

## Threading

Three concerns run independently so a slow model degrades gracefully instead of
freezing the window:

1. **Capture thread** reads the device and publishes only the newest frame.
   Stale frames are dropped rather than queued, so back-pressure never turns
   into latency.
2. **Inference worker** consumes frames at the configured `inference_fps`.
3. **UI thread** polls `engine.latest()` on a Qt timer and paints. It performs
   no inference.

The Hardware bridge samples on its own thread at `hardware.interval`.

Consequence: a model that takes 200 ms still leaves the window responsive; the
labels just update less often. The status bar reports the real rates rather than
an idealised one.

## The model contract

Adding a detector or classifier means implementing one interface and
registering it:

- `FaceDetector`: `load()`, `is_ready`, `detect(frame) -> list[FaceDetection]`
- `ExpressionClassifier`: `load()`, `is_ready`, `predict(crop) -> ExpressionResult`

Nothing else changes. `ModelRegistry` handles construction, availability probing
and fallback, and records *why* a backend was downgraded so the UI can say
`yunet unavailable, using haar (ModelMissingError: ...)`.

Fallback chain: `yunet -> haar -> yolo -> onnx` for detectors and
`heuristic -> onnx -> torchscript` for classifiers. A backend that cannot load
is skipped with a logged reason, never a crash.

## The hardware layer

Three backends, tried in order:

| Backend | Artefact | Needs |
|---|---|---|
| `pybind11` | `_visionai_native*.so` | Python headers at build time |
| `cabi` | `libvisionai_hw.so` via `ctypes` | no Python headers |
| `python` | pure Python | nothing |

All three return **the same dictionary shape**, so callers never branch. Every
parser is a pure static function taking a string, which is why the C++ tests can
assert against synthetic `/proc` content.

Two things the C++ layer gets right that are easy to get wrong:

- **Temperatures are millidegrees.** `hwmon/temp*_input` and `thermal_zone/temp`
  are in millidegrees unless a driver writes a decimal. Reading them as plain
  floats yields 50000 °C; the scale is detected from the text and readings outside
  −40…150 °C are rejected.
- **`nvidia-smi` is probed once.** Spawning it per tick cost 523 ms per sample
  and made the whole telemetry path useless. Detection is cached at startup, and
  a full snapshot now takes ~4 ms.

## Data flow

A `PipelineResult` carries the frame, detections, tracked faces, expression
results and timings. The same object feeds the overlay, the tables, the headless
renderer and (optionally) the database. The overlay is pure OpenCV with no Qt
dependency, which is why the visual output is testable without a display server.

## Database

One schema, two dialects. The SQL in `migrations/` is portable; the migrator
rewrites the handful of constructs that differ (`AUTOINCREMENT`, `SMALLINT`/
`TINYINT`, table options). Tests run against a throwaway SQLite store, so the
whole data layer is exercised with no server.

Repositories are explicit parameterised SQL — no ORM, so the statement that
reaches the database is the statement you can read.

## Configuration

`Settings` validates and clamps **on construction**, not only when you remember
to call `validate()`. A hand-edited config file cannot produce an invalid
object, which removes a whole class of intermittent bugs.

## Testing

715 Python tests plus 72 native checks, needing no camera, display or network.
Optional dependencies are covered by markers that skip themselves
(`requires_qt`, `requires_native`, `requires_ai`, `camera`), so a partial
install still gets a meaningful suite.
