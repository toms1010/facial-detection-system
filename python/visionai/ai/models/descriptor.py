"""Model descriptor: the contract between a weights file and its label set.

A trained expression model is not distributed with this repository because it
would carry its own training-data licence and bias caveats. Instead you drop
your own weights plus a small JSON descriptor into ``models/emotion/``.

Descriptor schema::

    {
      "name": "affectnet-resnet18",
      "architecture": "resnet18",
      "framework": "onnx",
      "classes": ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"],
      "input_size": [224, 224],
      "color_order": "rgb",
      "scale": 0.017529,
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225],
      "license": "...",
      "notes": "..."
    }
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visionai.ai import taxonomy
from visionai.ai.errors import ModelMissingError

LOG = logging.getLogger(__name__)

SUPPORTED_FRAMEWORKS = ("onnx", "torchscript")


@dataclass
class ModelDescriptor:
    """Everything needed to feed pixels to a trained model and label its output."""

    name: str
    framework: str
    weights: Path
    classes: tuple[str, ...]
    input_size: tuple[int, int] = (224, 224)
    color_order: str = "rgb"
    scale: float = 1.0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    architecture: str = ""
    license: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_classes(self) -> tuple[str, ...]:
        """Class names mapped into the project taxonomy where a mapping exists."""
        mapped = [taxonomy.normalize_label(c) or c for c in self.classes]
        return tuple(dict.fromkeys(mapped))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "framework": self.framework,
            "weights": str(self.weights),
            "classes": list(self.classes),
            "canonical_classes": list(self.canonical_classes),
            "input_size": list(self.input_size),
            "architecture": self.architecture,
            "license": self.license,
            "notes": self.notes,
        }


def _tuple3(values: Sequence[float], default: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(values) != 3:
        return default
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def parse_descriptor(payload: dict[str, Any], weights: Path) -> ModelDescriptor:
    framework = str(payload.get("framework", "onnx")).strip().lower()
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ModelMissingError(
            f"descriptor framework {framework!r} is not supported; "
            f"expected one of {SUPPORTED_FRAMEWORKS}"
        )
    classes = [str(c) for c in payload.get("classes", [])]
    if not classes:
        classes = list(taxonomy.DEFAULT_CLASS_NAMES)
    size = payload.get("input_size", [224, 224])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        size = [224, 224]
    return ModelDescriptor(
        name=str(payload.get("name") or weights.stem),
        framework=framework,
        weights=weights,
        classes=tuple(classes),
        input_size=(int(size[0]), int(size[1])),
        color_order=str(payload.get("color_order", "rgb")).strip().lower(),
        scale=float(payload.get("scale", 1.0)),
        mean=_tuple3(payload.get("mean", []), (0.0, 0.0, 0.0)),
        std=_tuple3(payload.get("std", []), (1.0, 1.0, 1.0)),
        architecture=str(payload.get("architecture", "")),
        license=str(payload.get("license", "")),
        notes=str(payload.get("notes", "")),
        extra={
            k: v
            for k, v in payload.items()
            if k
            not in {
                "name",
                "framework",
                "classes",
                "input_size",
                "color_order",
                "scale",
                "mean",
                "std",
                "architecture",
                "license",
                "notes",
            }
        },
    )


def load_descriptor(
    model_path: str | Path, descriptor_path: str | Path | None = None
) -> ModelDescriptor:
    """Load a descriptor for ``model_path``, defaulting to ``<model>.json``."""
    weights = Path(model_path).expanduser()
    if not weights.is_file():
        raise ModelMissingError(f"model weights not found: {weights}")

    if descriptor_path is not None:
        descriptor_file = Path(descriptor_path).expanduser()
    else:
        candidates = [
            weights.with_suffix(".json"),
            weights.parent / f"{weights.stem}.json",
            weights.parent / "descriptor.json",
        ]
        descriptor_file = next((c for c in candidates if c.is_file()), None)

    if descriptor_file is None or not descriptor_file.is_file():
        payload: dict[str, Any] = {
            "name": weights.stem,
            "framework": "onnx" if weights.suffix == ".onnx" else "torchscript",
            "classes": list(taxonomy.DEFAULT_CLASS_NAMES),
        }
        LOG.warning(
            "no descriptor next to %s; assuming the 7 primary classes in the "
            "taxonomy order. Add a JSON descriptor if the label order differs",
            weights.name,
        )
        return parse_descriptor(payload, weights)

    try:
        payload = json.loads(descriptor_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelMissingError(f"could not read descriptor {descriptor_file}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelMissingError(f"descriptor {descriptor_file} is not a JSON object")
    return parse_descriptor(payload, weights)


def discover_models(directory: str | Path) -> list[Path]:
    """List candidate model files in ``directory``, newest-looking first."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    found: list[Path] = []
    for pattern in ("*.onnx", "*.pt", "*.pth"):
        found.extend(p for p in sorted(root.glob(pattern)) if p.is_file())
    return found


def preprocess(crop: np.ndarray, descriptor: ModelDescriptor):
    """Resize and normalise a BGR crop into the model's expected input tensor."""
    import cv2

    height, width = descriptor.input_size
    image = crop
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if descriptor.color_order == "bgr":
        rgb = resized
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) * float(descriptor.scale)
    mean = np.asarray(descriptor.mean, dtype=np.float32)
    std = np.asarray(descriptor.std, dtype=np.float32)
    if np.any(std):
        tensor = (tensor - mean) / np.where(std == 0, 1.0, std)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)
