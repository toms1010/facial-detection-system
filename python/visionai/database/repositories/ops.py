"""Experiments, detection sessions and operational logging."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
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

STATUS_DRAFT = "draft"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_ABANDONED = "abandoned"


class ExperimentRepository(Repository):
    """A recorded comparison of dataset version, model version and results."""

    table = "experiments"

    def create(
        self,
        name: str,
        dataset_id: int | None = None,
        dataset_version: int = 1,
        model_version_id: int | None = None,
        training_run_id: int | None = None,
        test_run_id: int | None = None,
        config: Mapping[str, Any] | None = None,
        hardware: Mapping[str, Any] | None = None,
        notes: str = "",
        status: str = STATUS_DRAFT,
        created_by: int | None = None,
    ) -> int:
        if not str(name).strip():
            raise ValidationError("an experiment needs a name")
        if status not in (STATUS_DRAFT, STATUS_RUNNING, STATUS_COMPLETE, STATUS_ABANDONED):
            raise ValidationError(f"unknown experiment status {status!r}")
        return self.db.insert(
            "INSERT INTO experiments (name, dataset_id, dataset_version, model_version_id, "
            "training_run_id, test_run_id, config_json, hardware_json, status, notes, "
            "created_by, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                str(name),
                dataset_id,
                int(dataset_version),
                model_version_id,
                training_run_id,
                test_run_id,
                encode_json(dict(config) if config else None),
                encode_json(dict(hardware) if hardware else None),
                status,
                notes,
                created_by,
                now(),
                now(),
            ),
        )

    def get(self, experiment_id: int) -> dict[str, Any]:
        row = self.require(experiment_id, "experiment")
        row["config"] = decode_json(row.get("config_json"), {})
        row["hardware"] = decode_json(row.get("hardware_json"), {})
        row["metrics"] = self.metrics(experiment_id)
        return row

    def metrics(self, experiment_id: int) -> dict[str, float]:
        rows = self.db.query(
            "SELECT metric_name, metric_value FROM experiment_metrics WHERE experiment_id = %s",
            (experiment_id,),
        )
        return {str(row["metric_name"]): row["metric_value"] for row in rows}

    def set_metric(self, experiment_id: int, name: str, value: float, unit: str = "") -> None:
        existing = self.db.query_one(
            "SELECT id FROM experiment_metrics WHERE experiment_id = %s AND metric_name = %s",
            (experiment_id, name),
        )
        if existing is None:
            self.db.insert(
                "INSERT INTO experiment_metrics (experiment_id, metric_name, metric_value, unit) "
                "VALUES (%s, %s, %s, %s)",
                (experiment_id, name, value, unit),
            )
            return
        self.db.execute(
            "UPDATE experiment_metrics SET metric_value = %s, unit = %s WHERE id = %s",
            (value, unit, existing["id"]),
        )

    def set_status(self, experiment_id: int, status: str) -> None:
        if status not in (STATUS_DRAFT, STATUS_RUNNING, STATUS_COMPLETE, STATUS_ABANDONED):
            raise ValidationError(f"unknown experiment status {status!r}")
        updated = self.db.execute(
            "UPDATE experiments SET status = %s, updated_at = %s WHERE id = %s",
            (status, now(), experiment_id),
        )
        if not updated:
            raise NotFoundError(f"experiment {experiment_id} does not exist")

    def list_experiments(
        self, status: str | None = None, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        where = "status = %s" if status else ""
        params: Sequence[Any] = (status,) if status else ()
        return self.paginate(where, params, "created_at DESC", page, page_size)

    def delete(self, experiment_id: int) -> None:
        if not self.db.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,)):
            raise NotFoundError(f"experiment {experiment_id} does not exist")


class DetectionSessionRepository(Repository):
    """Detection sessions and their per-face results.

    Only measurements are stored. No image, embedding or biometric template is
    ever written by this layer.
    """

    table = "detection_sessions"

    def __init__(self, database) -> None:
        super().__init__(database)
        self.faces = FaceDetectionRepository(database)

    def start(
        self,
        session_name: str = "",
        camera_id: int | None = None,
        camera_label: str = "",
        source_kind: str = "webcam",
        detector_name: str = "",
        classifier_name: str = "",
        model_version_id: int | None = None,
        is_trained_model: bool = False,
        user_id: int | None = None,
    ) -> int:
        if source_kind not in ("webcam", "video", "synthetic", "image"):
            raise ValidationError(f"unknown session source {source_kind!r}")
        return self.db.insert(
            "INSERT INTO detection_sessions (session_name, camera_id, camera_label, source_kind, "
            "detector_name, classifier_name, model_version_id, is_trained_model, started_at, "
            "user_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                session_name,
                camera_id,
                camera_label,
                source_kind,
                detector_name,
                classifier_name,
                model_version_id,
                1 if is_trained_model else 0,
                now(),
                user_id,
            ),
        )

    def finish(
        self,
        session_id: int,
        frame_count: int = 0,
        max_face_count: int = 0,
        average_fps: float | None = None,
        average_latency_ms: float | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE detection_sessions SET finished_at = %s, frame_count = %s, "
            "max_face_count = %s, average_fps = %s, average_latency_ms = %s WHERE id = %s",
            (now(), int(frame_count), int(max_face_count), average_fps, average_latency_ms, session_id),
        )

    def get(self, session_id: int) -> dict[str, Any]:
        return self.require(session_id, "detection session")

    def expression_distribution(self, session_id: int) -> dict[str, int]:
        """How many times each expression was reported in a session."""
        rows = self.db.query(
            "SELECT er.expression AS name, COUNT(*) AS total "
            "FROM expression_results er "
            "JOIN face_detections fd ON fd.id = er.detection_id "
            "WHERE fd.session_id = %s AND er.is_confident = 1 "
            "GROUP BY er.expression ORDER BY total DESC",
            (session_id,),
        )
        return {str(row["name"]): int(row["total"]) for row in rows}

    def summary(self, session_id: int) -> dict[str, Any]:
        session = self.get(session_id)
        return {
            "id": session_id,
            "name": session.get("session_name") or "",
            "source": session.get("source_kind"),
            "detector": session.get("detector_name"),
            "classifier": session.get("classifier_name"),
            "is_trained_model": bool(int(session.get("is_trained_model") or 0)),
            "frame_count": int(session.get("frame_count") or 0),
            "max_face_count": int(session.get("max_face_count") or 0),
            "average_fps": session.get("average_fps"),
            "average_latency_ms": session.get("average_latency_ms"),
            "expressions": self.expression_distribution(session_id),
        }


class FaceDetectionRepository(Repository):
    """Per-frame face boxes and their expression results."""

    table = "face_detections"

    def __init__(self, database) -> None:
        super().__init__(database)
        self.expressions = ExpressionResultRepository(database)

    def add(
        self,
        session_id: int,
        frame_number: int,
        tracking_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        detection_score: float = 0.0,
        first_seen_frame: int | None = None,
        last_seen_frame: int | None = None,
    ) -> int:
        return self.db.insert(
            "INSERT INTO face_detections (session_id, frame_number, tracking_id, x_center, "
            "y_center, width, height, detection_score, first_seen_frame, last_seen_frame, "
            "hit_count, recorded_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s)",
            (
                session_id,
                int(frame_number),
                int(tracking_id),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
                float(detection_score),
                int(first_seen_frame if first_seen_frame is not None else frame_number),
                int(last_seen_frame if last_seen_frame is not None else frame_number),
                now(),
            ),
        )

    def for_frame(self, session_id: int, frame_number: int) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM face_detections WHERE session_id = %s AND frame_number = %s "
            "ORDER BY tracking_id",
            (session_id, int(frame_number)),
        )

    def history_for_track(self, session_id: int, tracking_id: int) -> list[dict[str, Any]]:
        """Per-face timeline, for the track history view."""
        return self.db.query(
            "SELECT fd.*, er.expression, er.confidence, er.is_confident, er.is_heuristic "
            "FROM face_detections fd "
            "LEFT JOIN expression_results er ON er.detection_id = fd.id "
            "WHERE fd.session_id = %s AND fd.tracking_id = %s ORDER BY fd.frame_number",
            (session_id, int(tracking_id)),
        )

    def record_pipeline_result(self, session_id: int, result: Any) -> int:
        """Store one :class:`PipelineResult` worth of measurements."""
        frame_number = int(getattr(result, "frame_number", 0))
        highest = 0
        for track in getattr(result, "tracks", []) or []:
            box = track.box
            width, height = _frame_size(result)
            detection_id = self.add(
                session_id,
                frame_number,
                int(track.track_id),
                _center(box.x1, width),
                _center(box.y1, height),
                box.width / width if width else 0.0,
                box.height / height if height else 0.0,
                detection_score=float(getattr(track, "detection_score", 0.0)),
            )
            highest = max(highest, 1)
            expression = getattr(track, "expression", None)
            if expression is not None:
                self.expressions.add(
                    detection_id,
                    expression.label,
                    float(expression.confidence),
                    is_confident=bool(expression.is_confident),
                    is_heuristic=bool(expression.is_heuristic),
                    is_trained_model=not bool(expression.is_heuristic),
                    alternative=(expression.alternatives[0] if expression.alternatives else None),
                )
        return highest


def _center(value: float, total: float) -> float:
    return (value / total) if total else 0.0


def _frame_size(result: Any) -> tuple[float, float]:
    frame = getattr(result, "frame", None)
    if frame is None or getattr(frame, "size", 0) == 0:
        return 0.0, 0.0
    height, width = frame.shape[:2]
    return float(width), float(height)


class ExpressionResultRepository(Repository):
    """One expression prediction attached to a face detection."""

    table = "expression_results"

    def add(
        self,
        detection_id: int,
        expression: str,
        confidence: float,
        is_confident: bool = True,
        is_heuristic: bool = False,
        is_trained_model: bool = True,
        alternative: str | None = None,
    ) -> int:
        return self.db.insert(
            "INSERT INTO expression_results (detection_id, expression, confidence, "
            "is_confident, is_heuristic, is_trained_model, alternative, recorded_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                detection_id,
                str(expression),
                float(confidence),
                1 if is_confident else 0,
                1 if is_heuristic else 0,
                1 if is_trained_model else 0,
                alternative,
                now(),
            ),
        )


class ApplicationLogRepository(Repository):
    """Structured application events."""

    table = "application_logs"

    def log(
        self,
        level: str,
        category: str,
        message: str,
        module: str = "",
        line_number: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> int:
        if level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            level = "INFO"
        return self.db.insert(
            "INSERT INTO application_logs (level, category, message, module, line_number, "
            "context_json, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                level.upper(),
                str(category),
                str(message)[:4000],
                str(module)[:160],
                line_number,
                encode_json(dict(context) if context else None),
                now(),
            ),
        )

    def query_log(
        self,
        level: str | None = None,
        category: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []
        if level:
            clauses.append("level = %s")
            params.append(level.upper())
        if category:
            clauses.append("category = %s")
            params.append(category)
        if search:
            clauses.append("message LIKE %s")
            params.append(f"%{search}%")
        return self.paginate(
            " AND ".join(clauses), params, "created_at DESC, id DESC", page, page_size
        )

    def purge_older_than(self, cutoff_iso: str) -> int:
        return self.db.execute("DELETE FROM application_logs WHERE created_at < %s", (cutoff_iso,))

    def counts_by_level(self) -> dict[str, int]:
        rows = self.db.query(
            "SELECT level, COUNT(*) AS total FROM application_logs GROUP BY level"
        )
        return {str(row["level"]): int(row["total"]) for row in rows}


class SystemMetricRepository(Repository):
    """Time series of performance and resource samples."""

    table = "system_metrics"

    def record(self, **values: Any) -> int:
        columns = (
            "cpu_percent", "cpu_temperature", "ram_percent", "ram_used_bytes",
            "ram_total_bytes", "gpu_percent", "gpu_temperature", "gpu_memory_used",
            "gpu_memory_total", "disk_percent", "net_rx_bytes_per_second",
            "net_tx_bytes_per_second", "inference_latency_ms", "inference_fps",
            "face_count", "backend",
        )
        unknown = set(values) - set(columns)
        if unknown:
            raise ValidationError(f"unknown system metric fields: {sorted(unknown)}")
        # face_count and backend are NOT NULL, so they get a real default rather
        # than a NULL that would fail the insert.
        defaults: dict[str, Any] = {"face_count": 0, "backend": ""}
        params = [values.get(column, defaults.get(column)) for column in columns]
        params.append(now())
        return self.db.insert(
            f"INSERT INTO system_metrics ({', '.join(columns)}, recorded_at) "
            f"VALUES ({', '.join(['%s'] * (len(columns) + 1))})",
            params,
        )

    def recent(
        self, minutes: int = 60, page: int = 1, page_size: int = 500
    ) -> list[dict[str, Any]]:
        cutoff = time_minus_minutes(minutes)
        return self.db.query(
            "SELECT * FROM system_metrics WHERE recorded_at >= %s ORDER BY recorded_at",
            (cutoff,),
        )

    def latest(self) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM system_metrics ORDER BY id DESC LIMIT 1")

    def averages(self, minutes: int = 15) -> dict[str, float | None]:
        cutoff = time_minus_minutes(minutes)
        row = self.db.query_one(
            "SELECT AVG(cpu_percent) AS cpu, AVG(ram_percent) AS ram, AVG(gpu_percent) AS gpu, "
            "AVG(inference_latency_ms) AS latency, AVG(inference_fps) AS fps, "
            "AVG(face_count) AS faces FROM system_metrics WHERE recorded_at >= %s",
            (cutoff,),
        )
        if not row:
            return {}
        return {key: row.get(key) for key in row if row.get(key) is not None}


class HardwareLogRepository(Repository):
    """Sensor and hardware events worth keeping."""

    table = "hardware_logs"

    def log(
        self,
        category: str,
        message: str,
        severity: str = "info",
        sensor_name: str = "",
        sensor_value: float | None = None,
    ) -> int:
        return self.db.insert(
            "INSERT INTO hardware_logs (category, severity, message, sensor_name, "
            "sensor_value, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                str(category),
                str(severity)[:16],
                str(message)[:2000],
                str(sensor_name)[:160],
                sensor_value,
                now(),
            ),
        )

    def recent(self, category: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if category:
            return self.db.query(
                "SELECT * FROM hardware_logs WHERE category = %s ORDER BY id DESC "
                f"LIMIT {int(limit)}",
                (category,),
            )
        return self.db.query(
            f"SELECT * FROM hardware_logs ORDER BY id DESC LIMIT {int(limit)}"
        )


def time_minus_minutes(minutes: int) -> str:
    """A DATETIME string ``minutes`` in the past, portable across dialects."""
    import datetime

    return (datetime.datetime.now() - datetime.timedelta(minutes=int(minutes))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
