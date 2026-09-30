# Linux AI Vision

Real-time face detection with **AI-estimated** facial-expression labels for Linux
desktops. Python handles inference, a C++17/pybind11 layer handles Linux
hardware telemetry, and a PySide6 desktop application puts them together.

> **Read this first.** Every expression label is a model's guess from visible
> facial features in one camera frame. It is **not** a statement about how a
> person actually feels, and it cannot see things a camera cannot see — tone of
> voice, posture in context, or anything behind a face covering. Do not use these
> labels to make decisions about people. The application repeats this message on
> every frame and in every dialog; see [PRIVACY.md](docs/PRIVACY.md) and
> [AI_MODEL.md](docs/AI_MODEL.md).

---

## What you get out of the box

| | |
|---|---|
| **Face detection** | YuNet (a 227 KB CNN, ships in `models/face/`) — no download needed |
| **Expression labels** | An **untrained geometric heuristic** by default, clearly marked as such |
| **Tracking** | Multi-face IoU tracking with stable `ID 01`, `ID 02`, … |
| **Hardware** | CPU, RAM, swap, GPU, temperatures, disk, network via a native C++ layer |
| **Storage** | Nothing. No frames are written and there is no network code at all |
| **Database** | Optional MySQL 8 schema (26 tables) for datasets, models, runs and experiments |

### The one thing to understand first

**No trained expression model ships with this repository.** None is
redistributable without its own licence and bias caveats, and none was
downloadable during development. So the default classifier is a hand-written
heuristic that measures mouth curvature, eye aperture, brow geometry and lower-face
tension. It is honest about this: confidence is capped at 55%, labels are marked
`heuristic, untrained` in the table, and the AI panel carries a warning banner.

To get real accuracy, drop your own weights into `models/emotion/` — see
[Adding a real model](#adding-a-real-expression-model).

---

## Requirements

- Linux (developed and tested on Ubuntu 26.04, kernel 7.0)
- Python **3.10+** (developed on 3.14)
- CMake **3.16+** and a C++17 compiler (GCC 12+)
- A webcam (optional — a synthetic source works without one)
- MySQL 8 (optional — only for the persistence features)

Camera access requires membership of the `video` group:

```bash
sudo usermod -aG video "$USER"   # then log out and back in
```

---

## Install

```bash
git clone <repo-url> linux-ai-vision
cd linux-ai-vision

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[qt,dev]"      # desktop UI + test tooling
pip install pymysql             # only if you will use MySQL
```

`opencv-python-headless` and `numpy` are the only hard runtime dependencies, so
the core pipeline installs and runs even with nothing else present.

### Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `[qt]` | PySide6 | the desktop application |
| `[ai]` | ultralytics, torch | the YOLO detector and training |
| `[onnx]` | onnxruntime | running your own ONNX expression model |
| `[dev]` | pytest, ruff, pybind11 | running the tests and linting |

### Build the native hardware layer

```bash
cmake -S . -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE="$(command -v python)"
cmake --build cpp/build
ctest --test-dir cpp/build          # 73 native checks
```

This produces two artefacts, and the Python side picks whichever it finds:

- `_visionai_native…so` — the pybind11 extension (preferred)
- `libvisionai_hw.so` — the same C++ behind a JSON C ABI, loaded with `ctypes`

> If CMake reports `pybind11 not found`, the shared library is still built and
> the app works through the ctypes backend. Check which one is active with
> `visionai devices`.

**The native layer is optional.** Without it the Hardware panel falls back to a
pure-Python `/proc` and `/sys` reader and says so. Point `VISIONAI_NATIVE_LIB` at
a specific `.so` to override discovery.

---

## How to run

### Desktop application

```bash
source .venv/bin/activate
visionai                                   # webcam, GUI
visionai --source 1                        # pick a different camera
visionai --source synthetic                # no camera needed
visionai --start                           # begin capturing immediately
visionai --theme light
```

If PySide6 is missing, or there is no display, the command explains what to
install and exits rather than failing obscurely.

### Terminal mode

Works over SSH, in containers and in CI, with no display at all:

```bash
visionai headless --source synthetic --frames 200
visionai headless --source 0 --duration 30
visionai headless --source clip.mp4 --no-colour
```

### Check everything works

```bash
visionai self-test        # loads models, runs a frame, samples hardware
visionai list-cameras
visionai devices          # which hardware backend is active, and what it sees
visionai models           # which model files were found
visionai benchmark --frames 200 --json
```

`self-test` is the one to run first if something looks wrong — it exercises the
detector, classifier, overlay, hardware bridge and the threaded engine.

---

## Command reference

| Command | Purpose |
|---|---|
| `run` | desktop application (the default when no command is given) |
| `headless` | run the pipeline in the terminal |
| `list-cameras` | enumerate capture devices |
| `devices` | hardware backend status and live telemetry |
| `models` | model files found, and available backends |
| `self-test` | verify every component loads and runs |
| `benchmark` | FPS, latency and resource usage |
| `settings show/set/reset/path` | inspect and change settings |
| `db status/migrate/seed` | MySQL schema and migrations |
| `user add/list` | account administration |

Every command accepts `--config PATH`, `--log-level LEVEL` and `--no-log-file`.
`run`, `headless` and `benchmark` also accept `--detector`, `--classifier`,
`--device`, `--detection-confidence`, `--emotion-confidence`, `--inference-fps`,
`--theme` and repeatable `--set path.to.key=value`.

---

## Configuration

Settings live in `~/.config/linux-ai-vision/settings.json` (override with
`VISIONAI_CONFIG`). Every value is clamped on load, so a hand-edited file cannot
wedge the application: a number that cannot be parsed, or one outside its range,
falls back to the default and says so on stderr.

```bash
visionai settings show
visionai settings set pipeline.inference_fps=20 models.device=cuda
visionai settings set --help
```

An unknown key is an error, not a silent no-op — `visionai settings set
models.typo_key=1` prints the valid keys in that group and writes nothing.

Most-used keys:

| Key | Default | Effect |
|---|---|---|
| `camera.index` | `0` | capture device |
| `camera.width` / `camera.height` | `1280` / `720` | capture resolution |
| `pipeline.inference_fps` | `15` | how often inference runs |
| `models.detector` | `yunet` | `yunet`, `haar`, `yolo`, `onnx` |
| `models.classifier` | `heuristic` | `heuristic`, `onnx`, `torchscript` |
| `models.detection_confidence` | `0.5` | detector threshold |
| `models.emotion_confidence` | `0.35` | below this, the label is withheld |
| `models.device` | `auto` | `auto`, `cpu`, `cuda` |
| `ui.theme` | `dark` | `dark`, `light` |
| `privacy.store_frames` | `false` | nothing is stored unless you set this |

Useful environment variables:

| Variable | Purpose |
|---|---|
| `VISIONAI_CONFIG` | settings file location |
| `VISIONAI_NATIVE_LIB` | explicit path to the compiled layer |
| `VISIONAI_LOG_DIR` | log directory |
| `VISIONAI_SQLITE_PATH` | use a local store instead of MySQL |
| `MYSQL_HOST` / `_PORT` / `_DATABASE` / `_USER` / `_PASSWORD` | database connection |
| `QT_QPA_PLATFORM=offscreen` | run Qt widgets headless (used by the tests) |

---

## Adding a real expression model

1. Put the weights in `models/emotion/`, e.g. `affectnet.onnx`.
2. Put a descriptor beside it at `models/emotion/affectnet.json`:

```json
{
  "name": "affectnet-resnet18",
  "framework": "onnx",
  "classes": ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"],
  "input_size": [224, 224],
  "color_order": "rgb",
  "scale": 0.017529,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "license": "your licence here",
  "notes": "training data and known limitations"
}
```

3. Select it:

```bash
visionai run --classifier onnx --classifier-weights models/emotion/affectnet.onnx
```

**The `classes` order must match the model's output order.** Common dataset
spellings are mapped automatically (`anger` → `angry`, `happiness` → `happy`).

A wrong descriptor is reported as a clear error rather than silently producing
nonsense labels. TorchScript (`.pt`) works the same way with
`"framework": "torchscript"`.

### Using a YOLO face model

```bash
visionai run --detector yolo --detector-weights models/face/yolov8n-face.pt
```

No face-specific YOLO weights are bundled, because the well-known ones are not
reliably downloadable. A generic COCO model also works: it detects `person`
boxes, and the detector derives an approximate face region from the top of each
box, reporting that it is doing so.

---

## Database (optional)

Built against MySQL 8, but every test runs against a throwaway SQLite store, so
no server is needed to work on the code.

```bash
export MYSQL_HOST=127.0.0.1 MYSQL_DATABASE=linux_ai_vision \
       MYSQL_USER=visionai MYSQL_PASSWORD='...'
visionai db migrate          # 7 migrations, 26 tables, reference data
visionai db status
visionai user add admin admin@example.com --role administrator
```

For development without a server:

```bash
visionai db --sqlite migrate
visionai --sqlite user list
```

Migrations are numbered `.sql` files in `python/visionai/database/migrations/`.
Add a new one rather than editing the schema by hand; `visionai db status` lists
what is pending and the application refuses to run against a stale schema.

> **Image bytes are never stored in the database.** Datasets record filesystem
> paths; the database holds metadata, labels, metrics and provenance. No table
> has a column for a face image or an embedding, and a test enforces that.

---

## Tests

```bash
pytest                    # 776 tests
pytest -m "not slow"      # skip the benchmarks
ruff check python tests
ctest --test-dir cpp/build
```

The suite needs no camera, no display and no network. Tests that need one of
those carry a marker (`requires_qt`, `requires_native`, `camera`) and skip
themselves when it is unavailable.

---

## Troubleshooting

**The expression labels are wrong / low confidence**
That is the untrained fallback. Install a real model — see above. The AI tab
shows exactly which classifier is active.

**`visionai: No display detected`**
Set `DISPLAY` (X11) or `WAYLAND_DISPLAY` (Wayland), or use `visionai headless`.

**`could not open camera 0`**
Run `visionai list-cameras`. If the device is listed, check that you are in the
`video` group and that no other application holds the device (`fuser /dev/video0`).

**`no Haar cascade XML found` / detector problems**
YuNet is the default and ships in `models/face/`. Haar needs OpenCV < 5.0,
which removed `CascadeClassifier`. Use `--detector yunet`.

**Hardware shows `n/a` for temperature or GPU**
Those sensors are not exposed on every machine. The Hardware panel lists a note
for each one that is missing, and `visionai devices` explains it.

**`the database schema is out of date`**
Run `visionai db migrate`.

**Everything is slow**
Check `visionai benchmark`. On a busy machine expect much higher latency than
the figures in [PERFORMANCE.md](docs/PERFORMANCE.md) — verify with
`uptime` before assuming a regression.

---

## Project layout

```
python/visionai/
  ai/            detection, classification, tracking, the model registry
  camera/        capture, frame sources
  hardware/      bridge to the C++ layer, with a pure-Python fallback
  database/      driver, migrations, seed data, repositories
  security/      password hashing, roles and permissions
  pipeline/      threaded engine and statistics
  ui/            PySide6 panels, OpenCV overlay, headless renderer
  config/        typed, validated settings
cpp/
  hardware/      C++17 monitors: cpu, gpu, temperature, memory, network, disk
  bindings/      pybind11 module and a JSON C ABI
models/
  face/          YuNet weights (bundled)
  emotion/       your expression model goes here (not bundled)
tests/           776 tests
```

Architecture notes are in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Privacy

- Inference is **entirely local**. There is no HTTP client, no telemetry and no
  upload path anywhere in the codebase; tests enforce this.
- Camera frames are **never written to disk** unless you explicitly enable it in
  Settings, and there is a button that deletes anything already stored.
- The hardware panel redacts the host name by default so telemetry can be shared
  in a bug report.
- The camera can be disabled from Settings, which stops capture immediately.

Details and the limits of what a camera can infer: [PRIVACY.md](docs/PRIVACY.md).

---

## Licence

MIT. Bundled third-party assets keep their own licences — see
[THIRD_PARTY.md](docs/THIRD_PARTY.md).
# facial-detection-system
