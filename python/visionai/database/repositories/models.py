"""Model registry, versions, training runs, evaluation and the confusion matrix."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from visionai.database.repositories.base import (
    NotFoundError,
    Repository,
    ValidationError,
    decode_json,
    encode_json,
    now,
)

LOG = logging.getLogger(__name__)

TASK_DETECTION = "detection"
TASK_EXPRESSION = "expression"
TASK_EMBEDDING = "embedding"
TASK_OTHER = "other"

STATUS_REGISTERED = "registered"
STATUS_VALIDATED = "validated"
STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"
STATUS_FAILED = "failed"

RUN_QUEUED = "queued"
RUN_RUNNING = "running"
RUN_PAUSED = "paused"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_CANCELLED = "cancelled"


class ModelRepository(Repository):
    """Logical model families, e.g. "Face Detector"."""

    table = "models"

    def __init__(self, database) -> None:
        super().__init__(database)
        self.versions = ModelVersionRepository(database)

    def create(
        self,
        name: str,
        task_type: str = TASK_EXPRESSION,
        description: str = "",
        owner_id: int | None = None,
    ) -> int:
        name = str(name).strip()
        if not name:
            raise ValidationError("a model needs a name")
        if task_type not in (TASK_DETECTION, TASK_EXPRESSION, TASK_EMBEDDING, TASK_OTHER):
            raise ValidationError(f"unknown task type {task_type!r}")
        if self.db.query_one("SELECT 1 AS ok FROM models WHERE name = %s", (name,)):
            raise ValidationError(f"model {name!r} already exists")
        return self.db.insert(
            "INSERT INTO models (name, task_type, description, owner_id, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (name, task_type, description, owner_id, now(), now()),
        )

    def get(self, model_id: int) -> dict[str, Any]:
        return self.require(model_id, "model")

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM models WHERE name = %s", (str(name),))

    def list_models(
        self, task_type: str | None = None, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        where = "task_type = %s" if task_type else ""
        params: Sequence[Any] = (task_type,) if task_type else ()
        return self.paginate(where, params, "name ASC", page, page_size)

    def summary(self, model_id: int) -> dict[str, Any]:
        """A model with its versions and the active one, for the registry table."""
        model = self.get(model_id)
        versions = self.versions.for_model(model_id)
        active = next((v for v in versions if int(v.get("is_active") or 0) == 1), None)
        latest_test = self.db.query_one(
            "SELECT accuracy, precision, recall, f1, map50, map50_95, finished_at "
            "FROM test_runs WHERE model_version_id = %s AND status = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (active["id"] if active else 0, RUN_COMPLETED),
        )
        return {
            "id": model_id,
            "name": model["name"],
            "task_type": model["task_type"],
            "description": model.get("description") or "",
            "version_count": len(versions),
            "active_version": active["version"] if active else None,
            "status": active["status"] if active else "no versions",
            "created_at": model.get("created_at"),
            "metrics": dict(latest_test) if latest_test else None,
        }

    def delete(self, model_id: int) -> None:
        if not self.db.execute("DELETE FROM models WHERE id = %s", (model_id,)):
            raise NotFoundError(f"model {model_id} does not exist")


class ModelVersionRepository(Repository):
    """Immutable, individually addressable model artefacts."""

    table = "model_versions"

    def register(
        self,
        model_id: int,
        version: str,
        weights_path: str,
        framework: str = "onnx",
        descriptor_path: str | None = None,
        class_set: Iterable[str] | None = None,
        input_size: int = 224,
        dataset_id: int | None = None,
        status: str = STATUS_REGISTERED,
        file_size_bytes: int | None = None,
        checksum: str | None = None,
        licence: str = "",
        notes: str = "",
        created_by: int | None = None,
    ) -> int:
        version = str(version).strip()
        if not version or not str(weights_path).strip():
            raise ValidationError("a model version needs a version name and weights path")
        if self.db.query_one(
            "SELECT 1 AS ok FROM model_versions WHERE model_id = %s AND version = %s",
            (model_id, version),
        ):
            raise ValidationError(f"version {version!r} already exists for this model")
        return self.db.insert(
            "INSERT INTO model_versions (model_id, version, weights_path, framework, "
            "descriptor_path, class_set, input_size, dataset_id, status, file_size_bytes, "
            "checksum, licence, notes, created_by, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                model_id,
                version,
                str(weights_path),
                framework,
                descriptor_path,
                encode_json(list(class_set) if class_set else None),
                int(input_size),
                dataset_id,
                status,
                file_size_bytes,
                checksum,
                licence,
                notes,
                created_by,
                now(),
            ),
        )

    def get(self, version_id: int) -> dict[str, Any]:
        return self.require(version_id, "model version")

    def for_model(self, model_id: int) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM model_versions WHERE model_id = %s ORDER BY created_at DESC",
            (model_id,),
        )

    def find(self, model_id: int, version: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM model_versions WHERE model_id = %s AND version = %s",
            (model_id, str(version)),
        )

    def active_version(self, model_id: int) -> dict[str, Any] | None:
        """The single version currently marked active, if any."""
        return self.db.query_one(
            "SELECT * FROM model_versions WHERE model_id = %s AND is_active = 1 LIMIT 1",
            (model_id,),
        )

    def activate(self, version_id: int) -> dict[str, Any]:
        """Make one version active, retiring the previous one.

        Activating is always explicit. Nothing here promotes a version
        automatically, and the previous version is marked retired rather than
        deleted so its metrics stay attributable.
        """
        version = self.get(version_id)
        model_id = int(version["model_id"])
        with self.db.transaction():
            self.db.execute(
                "UPDATE model_versions SET is_active = 0, status = %s "
                "WHERE model_id = %s AND is_active = 1 AND id <> %s",
                (STATUS_RETIRED, model_id, version_id),
            )
            self.db.execute(
                "UPDATE model_versions SET is_active = 1, status = %s WHERE id = %s",
                (STATUS_ACTIVE, version_id),
            )
        LOG.info("activated model %s version %s", model_id, version["version"])
        return self.get(version_id)

    def deactivate(self, version_id: int) -> None:
        updated = self.db.execute(
            "UPDATE model_versions SET is_active = 0, status = %s WHERE id = %s",
            (STATUS_RETIRED, version_id),
        )
        if not updated:
            raise NotFoundError(f"model version {version_id} does not exist")

    def set_status(self, version_id: int, status: str) -> None:
        if status not in (
            STATUS_REGISTERED, STATUS_VALIDATED, STATUS_ACTIVE, STATUS_RETIRED, STATUS_FAILED
        ):
            raise ValidationError(f"unknown model status {status!r}")
        updated = self.db.execute(
            "UPDATE model_versions SET status = %s WHERE id = %s", (status, version_id)
        )
        if not updated:
            raise NotFoundError(f"model version {version_id} does not exist")

    def delete(self, version_id: int) -> None:
        row = self.get(version_id)
        if int(row.get("is_active") or 0) == 1:
            raise ValidationError(
                "refusing to delete the active version; deactivate it first"
            )
        self.db.execute("DELETE FROM model_versions WHERE id = %s", (version_id,))

    def registry(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """The registry table: every model with its latest metrics."""
        page = max(1, int(page))
        page_size = max(1, min(500, int(page_size)))
        total = int(self.db.scalar("SELECT COUNT(*) AS n FROM models", default=0) or 0)
        rows = self.db.query(
            "SELECT m.id AS model_id, m.name, m.task_type, mv.id AS version_id, mv.version, "
            "mv.status, mv.is_active, mv.created_at, "
            "tr.accuracy, tr.precision, tr.recall, tr.f1, tr.map50, tr.map50_95 "
            "FROM models m "
            "LEFT JOIN model_versions mv ON mv.is_active = 1 AND mv.model_id = m.id "
            "LEFT JOIN test_runs tr ON tr.id = ("
            "  SELECT id FROM test_runs WHERE model_version_id = mv.id AND status = 'completed' "
            "  ORDER BY created_at DESC LIMIT 1) "
            f"ORDER BY m.name LIMIT {page_size} OFFSET {(page - 1) * page_size}"
        )
        return {
            "items": [dict(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, -(-total // page_size)),
        }


class TrainingRunRepository(Repository):
    """Training runs and their per-epoch metrics.

    A run is a queue entry as well as a record: status moves queued -> running ->
    completed/failed/cancelled, and the job manager picks up ``queued`` rows.
    """

    table = "training_runs"

    METRIC_FIELDS = (
        "train_loss", "val_loss", "precision", "recall", "f1",
        "map50", "map50_95", "accuracy", "learning_rate", "gpu_percent", "gpu_memory_mb",
    )

    def enqueue(
        self,
        model_id: int,
        dataset_id: int,
        run_name: str,
        config: Mapping[str, Any] | None = None,
        device: str = "cpu",
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        learning_rate: float = 0.01,
        optimizer: str = "auto",
        workers: int = 4,
        early_stopping: bool = False,
        launched_by: int | None = None,
    ) -> int:
        if not str(run_name).strip():
            raise ValidationError("a training run needs a name")
        dataset = self.db.query_one(
            "SELECT version FROM datasets WHERE id = %s", (dataset_id,)
        )
        if dataset is None:
            raise ValidationError(f"dataset {dataset_id} does not exist")
        return self.db.insert(
            "INSERT INTO training_runs (model_id, dataset_id, dataset_version, run_name, "
            "status, config_json, device, epochs, batch_size, image_size, learning_rate, "
            "optimizer, workers, early_stopping, launched_by, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                model_id,
                dataset_id,
                int(dataset["version"]),
                str(run_name),
                RUN_QUEUED,
                encode_json(dict(config) if config else None),
                device,
                int(epochs),
                int(batch_size),
                int(image_size),
                float(learning_rate),
                optimizer,
                int(workers),
                1 if early_stopping else 0,
                launched_by,
                now(),
            ),
        )

    def get(self, run_id: int) -> dict[str, Any]:
        row = self.require(run_id, "training run")
        row["config"] = decode_json(row.get("config_json"), {})
        return row

    def claim_next_queued(self) -> dict[str, Any] | None:
        """Atomically take the oldest queued run, for the job manager."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT * FROM training_runs WHERE status = %s ORDER BY created_at LIMIT 1",
                (RUN_QUEUED,),
            )
            if row is None:
                return None
            self.db.execute(
                "UPDATE training_runs SET status = %s, started_at = %s WHERE id = %s",
                (RUN_RUNNING, now(), row["id"]),
            )
        return self.get(int(row["id"]))

    def set_status(self, run_id: int, status: str) -> None:
        if status not in (RUN_QUEUED, RUN_RUNNING, RUN_PAUSED, RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED):
            raise ValidationError(f"unknown training status {status!r}")
        finished = now() if status in (RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED) else None
        updated = self.db.execute(
            "UPDATE training_runs SET status = %s, finished_at = %s WHERE id = %s",
            (status, finished, run_id),
        )
        if not updated:
            raise NotFoundError(f"training run {run_id} does not exist")

    def set_error(self, run_id: int, message: str) -> None:
        self.db.execute(
            "UPDATE training_runs SET status = %s, error_message = %s, finished_at = %s "
            "WHERE id = %s",
            (RUN_FAILED, str(message)[:2000], now(), run_id),
        )

    def record_metric(self, run_id: int, epoch: int, **values: float) -> int:
        unknown = set(values) - set(self.METRIC_FIELDS)
        if unknown:
            raise ValidationError(f"unknown training metrics: {sorted(unknown)}")
        columns = ["run_id", "epoch", *self.METRIC_FIELDS, "recorded_at"]
        params: list[Any] = [run_id, int(epoch)]
        params.extend(values.get(field) for field in self.METRIC_FIELDS)
        params.append(now())
        placeholders = ", ".join(["%s"] * len(columns))
        return self.db.insert(
            f"INSERT INTO training_metrics ({', '.join(columns)}) VALUES ({placeholders})",
            params,
        )

    def metrics(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM training_metrics WHERE run_id = %s ORDER BY epoch", (run_id,)
        )

    def progress(self, run_id: int) -> dict[str, Any]:
        run = self.get(run_id)
        latest = self.db.query_one(
            "SELECT * FROM training_metrics WHERE run_id = %s ORDER BY epoch DESC LIMIT 1",
            (run_id,),
        )
        epochs = int(run.get("epochs") or 0)
        done = int(latest["epoch"]) if latest else 0
        return {
            "run_id": run_id,
            "name": run["run_name"],
            "status": run["status"],
            "epoch": done,
            "epochs": epochs,
            "percent": round(done / epochs * 100.0, 1) if epochs else 0.0,
            "latest": dict(latest) if latest else None,
        }

    def list_runs(
        self, status: str | None = None, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        where = "status = %s" if status else ""
        params: Sequence[Any] = (status,) if status else ()
        return self.paginate(where, params, "created_at DESC", page, page_size)

    def queue_view(self) -> list[dict[str, Any]]:
        """The job-queue list shown on the training page."""
        return self.db.query(
            "SELECT tr.id, tr.run_name, tr.status, tr.device, tr.epochs, tr.created_at, "
            "m.name AS model_name, d.name AS dataset_name, d.version AS dataset_version "
            "FROM training_runs tr "
            "LEFT JOIN models m ON m.id = tr.model_id "
            "LEFT JOIN datasets d ON d.id = tr.dataset_id "
            "ORDER BY tr.created_at DESC"
        )


class TestRunRepository(Repository):
    """Model evaluation runs, per-image results and the confusion matrix."""

    table = "test_runs"

    METRIC_FIELDS = (
        "accuracy", "precision", "recall", "f1", "map50", "map50_95", "average_latency_ms"
    )

    def __init__(self, database) -> None:
        super().__init__(database)
        self.results = TestResultRepository(database)
        self.confusion = ConfusionMatrixRepository(database)

    def start(
        self,
        model_version_id: int,
        run_name: str,
        source_kind: str = "dataset",
        source_path: str = "",
        device: str = "cpu",
        image_count: int = 0,
        created_by: int | None = None,
    ) -> int:
        if source_kind not in ("image", "folder", "video", "webcam", "dataset"):
            raise ValidationError(f"unknown test source {source_kind!r}")
        if self.db.query_one(
            "SELECT 1 AS ok FROM model_versions WHERE id = %s", (model_version_id,)
        ) is None:
            raise ValidationError(f"model version {model_version_id} does not exist")
        return self.db.insert(
            "INSERT INTO test_runs (model_version_id, run_name, source_kind, source_path, "
            "image_count, status, device, started_at, created_by, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                model_version_id,
                str(run_name),
                source_kind,
                str(source_path),
                int(image_count),
                RUN_RUNNING,
                device,
                now(),
                created_by,
                now(),
            ),
        )

    def get(self, run_id: int) -> dict[str, Any]:
        return self.require(run_id, "test run")

    def finish(self, run_id: int, **metrics: float) -> None:
        unknown = set(metrics) - set(self.METRIC_FIELDS)
        if unknown:
            raise ValidationError(f"unknown test metrics: {sorted(unknown)}")
        assignments = ", ".join(f"{field} = %s" for field in self.METRIC_FIELDS)
        params: list[Any] = [metrics.get(field) for field in self.METRIC_FIELDS]
        params.append(RUN_COMPLETED)
        params.append(now())
        params.append(run_id)
        self.db.execute(
            f"UPDATE test_runs SET {assignments}, status = %s, finished_at = %s WHERE id = %s",
            params,
        )

    def fail(self, run_id: int, message: str) -> None:
        self.db.execute(
            "UPDATE test_runs SET status = %s, error_message = %s, finished_at = %s WHERE id = %s",
            (RUN_FAILED, str(message)[:2000], now(), run_id),
        )

    def list_runs(self, model_version_id: int | None = None, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        if model_version_id is None:
            return self.paginate(order_by="created_at DESC", page=page, page_size=page_size)
        return self.paginate(
            "model_version_id = %s", (model_version_id,), "created_at DESC", page, page_size
        )

    def confusion_matrix(self, run_id: int) -> list[list[int]]:
        """A dense matrix with rows = actual, columns = predicted."""
        return self.confusion.as_matrix(run_id)

    def compare_versions(self, version_ids: Sequence[int]) -> list[dict[str, Any]]:
        """Side-by-side metrics for the comparison view.

        Returns the raw numbers only. Deciding which version is "best" depends on
        the use case, so no verdict is attached here.
        """
        rows: list[dict[str, Any]] = []
        for version_id in version_ids:
            version = self.db.query_one(
                "SELECT mv.id, mv.version, m.name AS model_name FROM model_versions mv "
                "JOIN models m ON m.id = mv.model_id WHERE mv.id = %s",
                (version_id,),
            )
            if version is None:
                continue
            latest = self.db.query_one(
                "SELECT accuracy, precision, recall, f1, map50, map50_95, average_latency_ms, "
                "finished_at FROM test_runs WHERE model_version_id = %s AND status = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (version_id, RUN_COMPLETED),
            )
            row = dict(version)
            for field in self.METRIC_FIELDS:
                row[field] = (latest or {}).get(field)
            rows.append(row)
        return rows


class TestResultRepository(Repository):
    """One row per evaluated image."""

    table = "test_results"

    def add(
        self,
        test_run_id: int,
        image_path: str,
        actual_class: str | None,
        predicted_class: str | None,
        confidence: float | None = None,
        latency_ms: float | None = None,
        face_count: int = 0,
    ) -> int:
        correct = 1 if actual_class and actual_class == predicted_class else 0
        return self.db.insert(
            "INSERT INTO test_results (test_run_id, image_path, correct, actual_class, "
            "predicted_class, confidence, latency_ms, face_count) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                test_run_id,
                str(image_path),
                correct,
                actual_class,
                predicted_class,
                confidence,
                latency_ms,
                int(face_count),
            ),
        )

    def for_run(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM test_results WHERE test_run_id = %s", (run_id,))

    def accuracy(self, run_id: int) -> float | None:
        row = self.db.query_one(
            "SELECT COUNT(*) AS total, SUM(correct) AS correct FROM test_results "
            "WHERE test_run_id = %s",
            (run_id,),
        )
        total = int((row or {}).get("total") or 0)
        if total == 0:
            return None
        return int((row or {}).get("correct") or 0) / total * 100.0


class ConfusionMatrixRepository(Repository):
    """Actual-versus-predicted counts, stored one cell per row."""

    table = "confusion_matrix"

    def record(self, test_run_id: int, actual: str, predicted: str, count: int = 1) -> None:
        existing = self.db.query_one(
            "SELECT id, count FROM confusion_matrix WHERE test_run_id = %s "
            "AND actual_class = %s AND predicted_class = %s",
            (test_run_id, actual, predicted),
        )
        if existing is None:
            self.db.insert(
                "INSERT INTO confusion_matrix (test_run_id, actual_class, predicted_class, count) "
                "VALUES (%s, %s, %s, %s)",
                (test_run_id, actual, predicted, int(count)),
            )
            return
        self.db.execute(
            "UPDATE confusion_matrix SET count = %s WHERE id = %s",
            (int(existing["count"]) + int(count), existing["id"]),
        )

    def build_from_results(self, test_run_id: int) -> list[list[int]]:
        """Recompute the matrix from the per-image results."""
        self.db.execute("DELETE FROM confusion_matrix WHERE test_run_id = %s", (test_run_id,))
        rows = self.db.query(
            "SELECT actual_class, predicted_class, COUNT(*) AS total FROM test_results "
            "WHERE test_run_id = %s GROUP BY actual_class, predicted_class",
            (test_run_id,),
        )
        for row in rows:
            self.record(
                test_run_id,
                str(row["actual_class"]),
                str(row["predicted_class"]),
                int(row["total"]),
            )
        return self.as_matrix(test_run_id)

    def labels(self, test_run_id: int) -> list[str]:
        rows = self.db.query(
            "SELECT DISTINCT actual_class AS name FROM confusion_matrix "
            "WHERE test_run_id = %s ORDER BY actual_class",
            (test_run_id,),
        )
        rows += self.db.query(
            "SELECT DISTINCT predicted_class AS name FROM confusion_matrix "
            "WHERE test_run_id = %s AND predicted_class NOT IN "
            "(SELECT actual_class FROM confusion_matrix WHERE test_run_id = %s) "
            "ORDER BY predicted_class",
            (test_run_id, test_run_id),
        )
        return [str(row["name"]) for row in rows]

    def as_matrix(self, test_run_id: int) -> list[list[int]]:
        labels = self.labels(test_run_id)
        if not labels:
            return []
        index = {label: position for position, label in enumerate(labels)}
        size = len(labels)
        matrix = [[0] * size for _ in range(size)]
        rows = self.db.query(
            "SELECT actual_class, predicted_class, count FROM confusion_matrix "
            "WHERE test_run_id = %s",
            (test_run_id,),
        )
        for row in rows:
            row_index = index.get(str(row["actual_class"]))
            column_index = index.get(str(row["predicted_class"]))
            if row_index is not None and column_index is not None:
                matrix[row_index][column_index] = int(row["count"])
        return matrix

    def per_class_accuracy(self, test_run_id: int) -> dict[str, float]:
        """Diagonal share per class, the standard per-class accuracy."""
        matrix = self.as_matrix(test_run_id)
        labels = self.labels(test_run_id)
        out: dict[str, float] = {}
        for position, label in enumerate(labels):
            total = sum(matrix[position])
            out[label] = (matrix[position][position] / total * 100.0) if total else 0.0
        return out
