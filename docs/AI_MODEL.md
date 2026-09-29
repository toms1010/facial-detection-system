# AI models

## The two stages are different things

```
camera frame
   -> DETECTOR      locates faces, returns boxes        (YuNet / Haar / YOLO / ONNX)
   -> CLASSIFIER    labels each face crop              (heuristic / ONNX / TorchScript)
   -> TRACKER       keeps identities stable
```

A detector answers *where are the faces*. A classifier answers *what expression
does this crop look like*. They are separate models with separate weights, and
conflating them is the most common way this kind of system misleads people.

**YOLO is used only as a detector here.** A stock COCO YOLO model knows the
`person` class; it does not classify expressions, and treating it as though it
does would be wrong.

## Detection

### YuNet (default)

A compact CNN from the OpenCV model zoo, ~227 KB, vendored in `models/face/`.
It runs on CPU in a few milliseconds and needs no download, which is why it is
the default: a fresh checkout works offline.

### Haar cascades

Present in `models/face/` for OpenCV < 5.0, which still exposes
`CascadeClassifier`. **OpenCV 5.0 removed that class**, so on a modern install
the Haar backend is hidden from the registry rather than failing at runtime.

### YOLO

Requires weights you supply:

```bash
visionai run --detector yolo --detector-weights models/face/yolov8n-face.pt
```

If the weights expose a `face` class, boxes are used directly. If they only know
`person` (a COCO model), the detector derives an approximate face region from the
top of each person box and says so in its `describe()` output — it never presents
that as a real face detection.

## Expression classification

### The default is NOT a trained model

`HeuristicExpressionClassifier` measures mouth curvature, eye aperture, brow
geometry and lower-face tension with OpenCV, then maps them onto the seven
primary classes with a hand-written scoring function. It is untrained.

It exists so the pipeline, the tracker, the overlay and the tests all have a
deterministic offline backend. It is marked as untrained everywhere it appears:
`is_trained_model = False`, confidence capped at 0.55, `is_confident` forced to
`False`, a warning banner on the AI tab, and a `heuristic, untrained` note in the
results table. **Expect it to be wrong often.** It is a placeholder, not a
feature.

### Adding a trained model

Drop weights in `models/emotion/` with a JSON descriptor beside them:

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
  "license": "...",
  "notes": "training data, known limitations"
}
```

| Field | Meaning |
|---|---|
| `framework` | `onnx` or `torchscript` |
| `classes` | **Must match the model's output order.** The last entry is the most likely class. |
| `input_size` | `[width, height]` |
| `color_order` | `rgb` (default) or `bgr` |
| `scale`, `mean`, `std` | Applied as `pixel * scale`, then `(x - mean) / std` |

Label aliases are resolved automatically, so `anger` becomes `angry` and
`happiness` becomes `happy`. A label count that disagrees with the descriptor is
a hard error, not a silent mislabel.

Select it with `--classifier onnx --classifier-weights models/emotion/<file>`.

**No expression model is bundled**, because each carries its own training-data
licence and bias caveats that this project cannot speak for. See
[THIRD_PARTY.md](THIRD_PARTY.md).

## The taxonomy

Seven primary classes: `happy`, `sad`, `angry`, `fear`, `surprise`, `disgust`,
`neutral`. Forty more (`joy`, `calm`, `exhausted`, `skeptical`, …) are defined in
`visionai/ai/taxonomy.py` and seeded into the database as **inactive** — they are
part of the taxonomy, but a label becomes active only when a model that predicts
it exists. Adding a class is a descriptor edit plus an `UPDATE`, not a code
change.

## What the output means

The application says:

> Detected facial expression: Happy (91% confidence)

and never:

> This person is definitely happy.

The distinction is enforced in one place, `visionai/ai/taxonomy.py`, so every
widget inherits it, and a test greps the whole package to make sure no module
snaps back into claiming certainty.

A confidence value is the model's own score for that crop. It is not calibrated
probability of a correct emotion, and it is not a statement about a person.
