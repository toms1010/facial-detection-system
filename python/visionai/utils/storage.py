"""On-disk capture storage.

Nothing here runs unless the user explicitly enables a storage option in
Settings. The default configuration writes no camera data at all, and the
"delete stored data" action in Settings wipes whatever exists.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from visionai.config.paths import user_data_dir
from visionai.config.settings import PrivacySettings

LOG = logging.getLogger(__name__)

CAPTURE_SUBDIR = "captures"
SNAPSHOT_SUBDIR = "snapshots"
ALLOWED_SUFFIXES = (".jpg", ".png", ".avi", ".mp4")


def capture_root() -> Path:
    return user_data_dir() / CAPTURE_SUBDIR


def snapshot_root() -> Path:
    return user_data_dir() / SNAPSHOT_SUBDIR


def storage_enabled(privacy: PrivacySettings) -> bool:
    return bool(privacy.any_persistence_enabled)


def ensure_root() -> Path:
    root = capture_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_frame(frame: np.ndarray, privacy: PrivacySettings, tag: str = "frame") -> Path | None:
    """Write a single frame when frame storage is enabled. Returns the path."""
    if not privacy.store_frames or frame is None or frame.size == 0:
        return None
    root = ensure_root()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = root / f"{tag}-{stamp}-{int(time.time() * 1000) % 1000:03d}.jpg"
    try:
        if not cv2.imwrite(str(path), frame):
            LOG.warning("could not write frame to %s", path)
            return None
    except cv2.error as exc:
        LOG.warning("frame write failed: %s", exc)
        return None
    return path


def save_face_snapshot(
    frame: np.ndarray, box, privacy: PrivacySettings, track_id: int
) -> Path | None:
    """Write a cropped face image when snapshot storage is enabled."""
    if not (privacy.store_snapshots or privacy.store_face_crops) or frame is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box.clamp(width, height).as_int()
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    root = snapshot_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"id{track_id:02d}-{int(time.time() * 1000) % 100000:05d}.jpg"
    try:
        if not cv2.imwrite(str(path), crop):
            return None
    except cv2.error as exc:
        LOG.warning("snapshot write failed: %s", exc)
        return None
    return path


def stored_files() -> list[Path]:
    """Every file this module could have written."""
    found: list[Path] = []
    for root in (capture_root(), snapshot_root()):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
                found.append(path)
    return found


def clear_stored_data() -> int:
    """Delete every stored capture and snapshot. Returns the file count removed."""
    removed = 0
    for root in (capture_root(), snapshot_root()):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                LOG.warning("could not delete %s: %s", path, exc)
        with contextlib.suppress(OSError):
            shutil.rmtree(root, ignore_errors=True)
    if removed:
        LOG.info("deleted %d stored capture file(s) on user request", removed)
    return removed
