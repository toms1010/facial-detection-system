# Third-party components

## Bundled assets

| File | Source | Licence |
|---|---|---|
| `models/face/face_detection_yunet_2023mar.onnx` | [opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT (OpenCV model zoo) |
| `models/face/face_detection_yunet_2023mar_int8.onnx` | [opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT |
| `models/face/haarcascade_*.xml` | [opencv/opencv](https://github.com/opencv/opencv) `data/haarcascades` | Apache-2.0 / MIT |

These are vendored because **OpenCV 5.0 no longer ships the cascade data files**
inside the wheel, so a fresh checkout would otherwise have no working offline
detector. The Haar cascades are only used on OpenCV < 5.0, which still exposes
`CascadeClassifier`.

## Python dependencies

Runtime: `numpy`, `opencv-python-headless` (both permissive). Optional:
`PySide6` (LGPL-3.0), `torch` (BSD-3-Clause), `ultralytics` (AGPL-3.0),
`onnxruntime` (MIT), `pymysql` (MIT).

> `ultralytics` is **AGPL-3.0**. It is an optional extra, never imported unless
> you select the YOLO detector, and it is not required for the default YuNet
> pipeline.

## Expression models

**None are bundled.** Expression models carry their own training-data licences
and demographic bias caveats, which the project cannot speak for on the
model's behalf. You supply your own; see [AI_MODEL.md](AI_MODEL.md).
