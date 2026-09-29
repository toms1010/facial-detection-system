"""Datasets, images and labels.

Image bytes are never stored. :class:`DatasetRepository` records the dataset
root on the filesystem and every image by its path relative to that root, so a
dataset can be moved, backed up or exported as a directory without touching the
database.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from visionai.database.repositories.base import (
    NotFoundError,
    Repository,
    ValidationError,
    now,
)

LOG = logging.getLogger(__name__)

SPLITS = ("train", "val", "test", "unassigned")
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.7, 0.15, 0.15)
HEALTH_OK = "ok"
HEALTH_WARNING = "warning"
HEALTH_ERROR = "error"


class DatasetRepository(Repository):
    """Dataset records and their aggregate statistics."""

    table = "datasets"

    def __init__(self, database) -> None:
        super().__init__(database)
        self.images = DatasetImageRepository(database)
        self.labels = DatasetLabelRepository(database)

    def create(
        self,
        name: str,
        root_path: str,
        description: str = "",
        version: int = 1,
        licence: str = "",
        attribution: str = "",
        created_by: int | None = None,
    ) -> int:
        name = str(name).strip()
        if not name:
            raise ValidationError("a dataset needs a name")
        if self.db.query_one(
            "SELECT 1 AS ok FROM datasets WHERE name = %s AND version = %s", (name, int(version))
        ):
            raise ValidationError(f"dataset {name!r} version {version} already exists")
        return self.db.insert(
            "INSERT INTO datasets (name, version, description, root_path, licence, attribution, "
            "status, created_by, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                name,
                int(version),
                description,
                str(root_path),
                licence,
                attribution,
                "draft",
                created_by,
                now(),
                now(),
            ),
        )

    def get(self, dataset_id: int) -> dict[str, Any]:
        return self.require(dataset_id, "dataset")

    def find_by_name(self, name: str, version: int | None = None) -> dict[str, Any] | None:
        if version is None:
            return self.db.query_one(
                "SELECT * FROM datasets WHERE name = %s ORDER BY version DESC LIMIT 1",
                (str(name),),
            )
        return self.db.query_one(
            "SELECT * FROM datasets WHERE name = %s AND version = %s", (str(name), int(version))
        )

    def versions(self, name: str) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM datasets WHERE name = %s ORDER BY version DESC", (str(name),)
        )

    def list_datasets(
        self, status: str | None = None, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        where = "status = %s" if status else ""
        params: Sequence[Any] = (status,) if status else ()
        return self.paginate(where, params, "name ASC, version DESC", page, page_size)

    def set_status(self, dataset_id: int, status: str) -> None:
        if status not in ("draft", "validating", "ready", "archived"):
            raise ValidationError(f"unknown dataset status {status!r}")
        updated = self.db.execute(
            "UPDATE datasets SET status = %s, updated_at = %s WHERE id = %s",
            (status, now(), dataset_id),
        )
        if not updated:
            raise NotFoundError(f"dataset {dataset_id} does not exist")

    def rename(self, dataset_id: int, name: str) -> None:
        updated = self.db.execute(
            "UPDATE datasets SET name = %s, updated_at = %s WHERE id = %s",
            (str(name).strip(), now(), dataset_id),
        )
        if not updated:
            raise NotFoundError(f"dataset {dataset_id} does not exist")

    def delete(self, dataset_id: int) -> None:
        if not self.db.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,)):
            raise NotFoundError(f"dataset {dataset_id} does not exist")

    def import_folder(
        self,
        dataset_id: int,
        folder: str | Path,
        extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        limit: int | None = None,
    ) -> dict[str, int]:
        """Register every image under ``folder`` that is not already recorded.

        Only paths and metadata are stored; nothing is copied.
        """
        self.require(dataset_id, "dataset")
        root = Path(folder).expanduser()
        if not root.is_dir():
            raise ValidationError(f"{root} is not a directory")

        known = {
            str(row["relative_path"])
            for row in self.db.query(
                "SELECT relative_path FROM dataset_images WHERE dataset_id = %s",
                (dataset_id,),
            )
        }
        suffixes = {e.lower() for e in extensions}
        inserted = 0
        skipped = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = str(path.relative_to(root))
            if relative in known:
                skipped += 1
                continue
            self.images.add(
                dataset_id,
                relative,
                str(path),
                file_size_bytes=path.stat().st_size,
            )
            inserted += 1
            if limit is not None and inserted >= limit:
                break
        self.refresh_statistics(dataset_id)
        LOG.info("imported %d image(s) into dataset %d (%d already present)", inserted, dataset_id, skipped)
        return {"inserted": inserted, "skipped": skipped}

    def refresh_statistics(self, dataset_id: int) -> dict[str, Any]:
        """Recompute the cached counts from the image and label tables."""
        totals = self.db.query_one(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN split = 'train' THEN 1 ELSE 0 END) AS train_count, "
            "SUM(CASE WHEN split = 'val' THEN 1 ELSE 0 END) AS val_count, "
            "SUM(CASE WHEN split = 'test' THEN 1 ELSE 0 END) AS test_count, "
            "SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) AS valid_count "
            "FROM dataset_images WHERE dataset_id = %s",
            (dataset_id,),
        ) or {}
        classes = self.db.query_one(
            "SELECT COUNT(DISTINCT class_name) AS n FROM dataset_labels dl "
            "JOIN dataset_images di ON di.id = dl.image_id WHERE di.dataset_id = %s",
            (dataset_id,),
        ) or {"n": 0}
        self.db.execute(
            "UPDATE datasets SET total_images = %s, train_count = %s, val_count = %s, "
            "test_count = %s, class_count = %s, updated_at = %s WHERE id = %s",
            (
                int(totals.get("total") or 0),
                int(totals.get("train_count") or 0),
                int(totals.get("val_count") or 0),
                int(totals.get("test_count") or 0),
                int(classes.get("n") or 0),
                now(),
                dataset_id,
            ),
        )
        return dict(totals) | {"class_count": int(classes.get("n") or 0)}

    def class_distribution(self, dataset_id: int) -> dict[str, int]:
        """How many labels each expression has, for the imbalance check."""
        rows = self.db.query(
            "SELECT dl.class_name AS name, COUNT(*) AS total FROM dataset_labels dl "
            "JOIN dataset_images di ON di.id = dl.image_id "
            "WHERE di.dataset_id = %s GROUP BY dl.class_name ORDER BY total DESC",
            (dataset_id,),
        )
        return {str(row["name"]): int(row["total"]) for row in rows}

    def assign_splits(
        self,
        dataset_id: int,
        ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        seed: int = 0,
    ) -> dict[str, int]:
        """Split images into train/val/test deterministically.

        Uses a stable hash of the relative path so re-running produces the same
        assignment, which matters for reproducible training runs.
        """
        train_ratio, val_ratio, _ = ratios
        if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
            raise ValidationError("split ratios must be between 0 and 1")

        rows = self.db.query(
            "SELECT id, relative_path FROM dataset_images WHERE dataset_id = %s "
            "ORDER BY relative_path",
            (dataset_id,),
        )
        counts = {"train": 0, "val": 0, "test": 0}
        for row in rows:
            digest = hashlib.sha256(f"{seed}:{row['relative_path']}".encode()).hexdigest()
            bucket = int(digest[:8], 16) % 10_000 / 10_000
            if bucket < train_ratio:
                split = "train"
            elif bucket < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"
            self.db.execute(
                "UPDATE dataset_images SET split = %s WHERE id = %s", (split, row["id"])
            )
            counts[split] += 1
        self.refresh_statistics(dataset_id)
        return counts

    def statistics(self, dataset_id: int) -> dict[str, Any]:
        dataset = self.get(dataset_id)
        distribution = self.class_distribution(dataset_id)
        return {
            "id": dataset_id,
            "name": dataset["name"],
            "version": dataset["version"],
            "status": dataset["status"],
            "health_status": dataset["health_status"],
            "images": int(dataset["total_images"] or 0),
            "classes": int(dataset["class_count"] or 0),
            "train": int(dataset["train_count"] or 0),
            "val": int(dataset["val_count"] or 0),
            "test": int(dataset["test_count"] or 0),
            "class_distribution": distribution,
        }

    def search(self, term: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        like = f"%{str(term).strip()}%"
        return self.paginate(
            "name LIKE %s OR description LIKE %s", (like, like), "name ASC", page, page_size
        )


class DatasetImageRepository(Repository):
    """One row per image file, addressed by path."""

    table = "dataset_images"

    def add(
        self,
        dataset_id: int,
        relative_path: str,
        absolute_path: str = "",
        file_size_bytes: int | None = None,
        width: int | None = None,
        height: int | None = None,
        checksum: str | None = None,
        split: str = "train",
    ) -> int:
        if split not in SPLITS:
            raise ValidationError(f"unknown split {split!r}")
        return self.db.insert(
            "INSERT INTO dataset_images (dataset_id, relative_path, absolute_path, "
            "file_size_bytes, width, height, checksum, split, imported_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                dataset_id,
                str(relative_path),
                str(absolute_path),
                file_size_bytes,
                width,
                height,
                checksum,
                split,
                now(),
            ),
        )

    def get(self, image_id: int) -> dict[str, Any]:
        return self.require(image_id, "dataset image")

    def for_dataset(
        self, dataset_id: int, split: str | None = None, page: int = 1, page_size: int = 100
    ) -> dict[str, Any]:
        if split:
            return self.paginate(
                "dataset_id = %s AND split = %s",
                (dataset_id, split),
                "relative_path ASC",
                page,
                page_size,
            )
        return self.paginate("dataset_id = %s", (dataset_id,), "relative_path ASC", page, page_size)

    def find(self, dataset_id: int, relative_path: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM dataset_images WHERE dataset_id = %s AND relative_path = %s",
            (dataset_id, str(relative_path)),
        )

    def set_valid(self, image_id: int, valid: bool) -> None:
        self.db.execute(
            "UPDATE dataset_images SET is_valid = %s WHERE id = %s",
            (1 if valid else 0, image_id),
        )

    def set_duplicate(self, image_id: int, duplicate: bool) -> None:
        self.db.execute(
            "UPDATE dataset_images SET is_duplicate = %s WHERE id = %s",
            (1 if duplicate else 0, image_id),
        )

    def mark_prelabelled(self, image_id: int, model_name: str) -> None:
        self.db.execute(
            "UPDATE dataset_images SET prelabel_model = %s, annotated_by = %s WHERE id = %s",
            (str(model_name), "ai", image_id),
        )

    def progress(self, dataset_id: int) -> dict[str, int]:
        """Annotation progress, for the dataset progress bar."""
        row = self.db.query_one(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN EXISTS (SELECT 1 FROM dataset_labels l WHERE l.image_id = dataset_images.id) "
            "THEN 1 ELSE 0 END) AS labelled "
            "FROM dataset_images WHERE dataset_id = %s",
            (dataset_id,),
        ) or {}
        return {"total": int(row.get("total") or 0), "labelled": int(row.get("labelled") or 0)}


class DatasetLabelRepository(Repository):
    """Normalised bounding-box labels, each optionally with an expression class."""

    table = "dataset_labels"

    def set_label(
        self,
        image_id: int,
        class_name: str,
        x_center: float = 0.5,
        y_center: float = 0.5,
        width_norm: float = 1.0,
        height_norm: float = 1.0,
        confidence: float | None = None,
        source: str = "manual",
    ) -> int:
        if source not in ("manual", "ai", "import"):
            raise ValidationError(f"unknown label source {source!r}")
        for name, value in (
            ("x_center", x_center),
            ("y_center", y_center),
            ("width_norm", width_norm),
            ("height_norm", height_norm),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValidationError(f"{name} must be between 0 and 1")

        class_id = self.db.scalar(
            "SELECT id FROM emotion_classes WHERE name = %s", (str(class_name),)
        )
        return self.db.insert(
            "INSERT INTO dataset_labels (image_id, class_id, class_name, x_center, y_center, "
            "width_norm, height_norm, confidence, source, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                image_id,
                int(class_id) if class_id else None,
                str(class_name),
                float(x_center),
                float(y_center),
                float(width_norm),
                float(height_norm),
                confidence,
                source,
                now(),
            ),
        )

    def replace_for_image(self, image_id: int) -> None:
        self.db.execute("DELETE FROM dataset_labels WHERE image_id = %s", (image_id,))

    def for_image(self, image_id: int) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM dataset_labels WHERE image_id = %s ORDER BY id", (image_id,)
        )

    def for_dataset(self, dataset_id: int) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT dl.* FROM dataset_labels dl "
            "JOIN dataset_images di ON di.id = dl.image_id "
            "WHERE di.dataset_id = %s",
            (dataset_id,),
        )

    def rename_class(self, dataset_id: int, old: str, new: str) -> int:
        return self.db.execute(
            "UPDATE dataset_labels SET class_name = %s, updated_at = %s "
            "WHERE class_name = %s AND image_id IN "
            "(SELECT id FROM dataset_images WHERE dataset_id = %s)",
            (str(new), now(), str(old), dataset_id),
        )

    def delete(self, label_id: int) -> None:
        self.db.execute("DELETE FROM dataset_labels WHERE id = %s", (label_id,))


class DatasetHealthReport(dict):
    """Validation outcome for one dataset, ready to render."""

    @property
    def status(self) -> str:
        if self.get("errors"):
            return HEALTH_ERROR
        if self.get("warnings"):
            return HEALTH_WARNING
        return HEALTH_OK

    def describe(self) -> str:
        lines = [f"Dataset health: {self.status}"]
        for key in ("images_valid", "labels_valid", "missing_files", "corrupt_images",
                    "missing_labels", "invalid_labels", "duplicates", "class_imbalance"):
            value = self.get(key)
            if value:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class DatasetValidator:
    """Checks a dataset for the problems that silently ruin a training run."""

    def __init__(self, datasets: DatasetRepository) -> None:
        self.datasets = datasets

    def validate(self, dataset_id: int) -> DatasetHealthReport:
        dataset = self.datasets.get(dataset_id)
        root = Path(dataset["root_path"]).expanduser()
        images = self.datasets.db.query(
            "SELECT * FROM dataset_images WHERE dataset_id = %s", (dataset_id,)
        )
        labelled_ids = {
            int(row["image_id"])
            for row in self.datasets.db.query(
                "SELECT DISTINCT image_id FROM dataset_labels WHERE image_id IN "
                "(SELECT id FROM dataset_images WHERE dataset_id = %s)",
                (dataset_id,),
            )
        }

        missing: list[str] = []
        corrupt: list[str] = []
        bad_size: list[str] = []
        for row in images:
            path = Path(str(row["absolute_path"] or ""))
            if not path.is_file() and root.is_dir():
                path = root / str(row["relative_path"])
            if not path.is_file():
                missing.append(str(row["relative_path"]))
                self.datasets.images.set_valid(int(row["id"]), False)
                continue
            if not self._readable(path):
                corrupt.append(str(row["relative_path"]))
                self.datasets.images.set_valid(int(row["id"]), False)
            if not row.get("width") or not row.get("height"):
                size = self._image_size(path)
                if size:
                    self.datasets.db.execute(
                        "UPDATE dataset_images SET width = %s, height = %s WHERE id = %s",
                        (size[0], size[1], row["id"]),
                    )
                else:
                    bad_size.append(str(row["relative_path"]))

        duplicates = self._find_duplicates(dataset_id)
        for image_id in duplicates:
            self.datasets.images.set_duplicate(image_id, True)

        missing_labels = [
            str(row["relative_path"])
            for row in images
            if int(row["id"]) not in labelled_ids
        ]
        invalid_labels = self._invalid_labels(dataset_id)
        imbalance = self._class_imbalance(dataset_id)

        warnings: list[str] = []
        if imbalance:
            warnings.append("class imbalance detected")
        if missing_labels:
            warnings.append(f"{len(missing_labels)} image(s) have no label")
        if bad_size:
            warnings.append(f"{len(bad_size)} image(s) could not be measured")

        report = DatasetHealthReport(
            dataset_id=dataset_id,
            images_checked=len(images),
            images_valid=len(images) - len(missing) - len(corrupt),
            labels_valid=len(images) - len(missing_labels),
            missing_files=missing,
            corrupt_images=corrupt,
            missing_labels=missing_labels,
            invalid_labels=invalid_labels,
            duplicates=duplicates,
            unmeasurable=bad_size,
            class_imbalance=imbalance,
            errors=missing + corrupt + invalid_labels,
            warnings=warnings,
        )
        self.datasets.db.execute(
            "UPDATE datasets SET health_status = %s, updated_at = %s WHERE id = %s",
            (report.status, now(), dataset_id),
        )
        return report

    @staticmethod
    def _readable(path: Path) -> bool:
        try:
            import cv2

            image = cv2.imread(str(path))
            return image is not None
        except Exception as exc:  # noqa: BLE001
            LOG.debug("could not probe %s: %s", path, exc)
            return path.stat().st_size > 0

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int] | None:
        try:
            import cv2

            image = cv2.imread(str(path))
            if image is None:
                return None
            height, width = image.shape[:2]
            return width, height
        except Exception:  # noqa: BLE001
            return None

    def _find_duplicates(self, dataset_id: int) -> list[int]:
        rows = self.datasets.db.query(
            "SELECT id, checksum FROM dataset_images WHERE dataset_id = %s AND checksum IS NOT NULL",
            (dataset_id,),
        )
        by_checksum: dict[str, list[int]] = {}
        for row in rows:
            by_checksum.setdefault(str(row["checksum"]), []).append(int(row["id"]))
        return [image_id for ids in by_checksum.values() if len(ids) > 1 for image_id in ids]

    def _invalid_labels(self, dataset_id: int) -> list[str]:
        known = {
            str(row["name"])
            for row in self.datasets.db.query("SELECT name FROM emotion_classes")
        }
        rows = self.datasets.db.query(
            "SELECT dl.id, dl.class_name FROM dataset_labels dl "
            "JOIN dataset_images di ON di.id = dl.image_id WHERE di.dataset_id = %s",
            (dataset_id,),
        )
        return [
            str(row["id"])
            for row in rows
            if str(row["class_name"]) not in known
            or str(row["class_name"]).strip() == ""
        ]

    def _class_imbalance(self, dataset_id: int, factor: float = 3.0) -> dict[str, Any]:
        distribution = self.datasets.class_distribution(dataset_id)
        if len(distribution) < 2:
            return {}
        counts = sorted(distribution.values())
        smallest, largest = counts[0], counts[-1]
        if smallest <= 0 or largest / smallest < factor:
            return {}
        return {
            "min": smallest,
            "max": largest,
            "ratio": round(largest / smallest, 2),
            "rarest": [name for name, total in distribution.items() if total == smallest],
        }


def checksum_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, used to spot duplicate images."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def scan_undocumented_images(
    folder: str | Path, database: Any | None = None
) -> list[Path]:
    """Image files on disk that are not recorded in any dataset.

    With a database connection the known paths are read from
    ``dataset_images``; without one every image file is returned, because there
    is nothing to compare against.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    known: set[str] = set()
    if database is not None:
        known = {
            str(row["absolute_path"])
            for row in database.query(
                "SELECT absolute_path FROM dataset_images WHERE absolute_path <> ''"
            )
        }
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        and str(path) not in known
    ]


def default_split_ratios() -> Mapping[str, float]:
    return {"train": 0.7, "val": 0.15, "test": 0.15}


def free_space(path: str | Path = "/") -> int:
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
