"""Dashboard counters aggregated from the database."""

from __future__ import annotations

import logging
from typing import Any

from visionai.database.repositories import Repositories

LOG = logging.getLogger(__name__)


def dashboard_counters(repositories: Repositories) -> dict[str, Any]:
    """The headline numbers for the dashboard.

    Every counter is a COUNT against a real table, so the dashboard cannot drift
    away from the stored data.
    """
    db = repositories.database
    return {
        "models": _count(db, "models"),
        "model_versions": _count(db, "model_versions"),
        "active_versions": _count(db, "model_versions", "is_active = 1"),
        "datasets": _count(db, "datasets"),
        "dataset_images": _count(db, "dataset_images"),
        "training_runs": _count(db, "training_runs"),
        "training_running": _count(db, "training_runs", "status = 'running'"),
        "test_runs": _count(db, "test_runs"),
        "test_completed": _count(db, "test_runs", "status = 'completed'"),
        "experiments": _count(db, "experiments"),
        "detection_sessions": _count(db, "detection_sessions"),
        "users": _count(db, "users"),
        "detected_faces": _count(db, "face_detections"),
        "expression_results": _count(db, "expression_results"),
    }


def expression_summary(repositories: Repositories) -> list[dict[str, Any]]:
    """Expression distribution across all confident results."""
    rows = repositories.database.query(
        "SELECT expression, COUNT(*) AS total, AVG(confidence) AS mean_confidence "
        "FROM expression_results WHERE is_confident = 1 "
        "GROUP BY expression ORDER BY total DESC"
    )
    return [
        {
            "expression": str(row["expression"]),
            "count": int(row["total"]),
            "mean_confidence": round(float(row["mean_confidence"] or 0.0), 4),
        }
        for row in rows
    ]


def training_history(repositories: Repositories, limit: int = 20) -> list[dict[str, Any]]:
    """Loss curves for the training chart."""
    rows = repositories.database.query(
        "SELECT tr.id, tr.run_name, tm.epoch, tm.train_loss, tm.val_loss, tm.precision, "
        "tm.recall, tm.map50 FROM training_metrics tm "
        "JOIN training_runs tr ON tr.id = tm.run_id "
        "ORDER BY tr.id DESC, tm.epoch LIMIT %s",
        (int(limit),),
    )
    return [dict(row) for row in rows]


def recent_activity(repositories: Repositories, limit: int = 20) -> list[dict[str, Any]]:
    """A merged feed of the most recent events across the platform."""
    db = repositories.database
    feed: list[dict[str, Any]] = []
    feed += [
        {"kind": "training", "label": str(row["run_name"]), "status": str(row["status"]),
         "at": str(row["created_at"])}
        for row in db.query(
            "SELECT run_name, status, created_at FROM training_runs "
            f"ORDER BY created_at DESC, id DESC LIMIT {int(limit)}"
        )
    ]
    feed += [
        {"kind": "test", "label": str(row["run_name"]), "status": str(row["status"]),
         "at": str(row["created_at"])}
        for row in db.query(
            "SELECT run_name, status, created_at FROM test_runs "
            f"ORDER BY created_at DESC, id DESC LIMIT {int(limit)}"
        )
    ]
    feed += [
        {"kind": "session", "label": str(row["session_name"] or "detection"),
         "status": "completed" if row["finished_at"] else "running",
         "at": str(row["started_at"])}
        for row in db.query(
            "SELECT session_name, started_at, finished_at FROM detection_sessions "
            f"ORDER BY started_at DESC, id DESC LIMIT {int(limit)}"
        )
    ]
    feed.sort(key=lambda item: str(item["at"]), reverse=True)
    return feed[:limit]


def dataset_summary(repositories: Repositories) -> list[dict[str, Any]]:
    rows = repositories.database.query(
        "SELECT name, version, status, health_status, total_images, class_count, "
        "train_count, val_count, test_count FROM datasets ORDER BY name, version"
    )
    return [dict(row) for row in rows]


def _count(db, table: str, where: str = "") -> int:
    sql = f"SELECT COUNT(*) AS n FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return int(db.scalar(sql, default=0) or 0)
