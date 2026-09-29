"""Reference data seeded after migration.

Roles, permissions and expression classes are *data*, not code. Seeding them
into tables means new permissions and new expression labels can be introduced
without a schema change, which is what the platform needs as the taxonomy grows.

Seeding is idempotent: it inserts only what is missing and never overwrites a
row a user has already customised.
"""

from __future__ import annotations

import json
import logging

from visionai.ai import taxonomy
from visionai.database.driver import Database
from visionai.security.passwords import Permission, permissions_for_role

LOG = logging.getLogger(__name__)

PERMISSION_DESCRIPTIONS: dict[str, str] = {
    "user:manage": "Create, edit, suspend and delete user accounts",
    "role:manage": "Assign roles and change role permissions",
    "database:manage": "Run migrations and manage the database",
    "settings:manage": "Change application-wide settings",
    "dataset:read": "View datasets and labels",
    "dataset:write": "Import datasets and edit labels",
    "dataset:delete": "Delete datasets",
    "model:read": "View the model registry",
    "model:write": "Register models and versions",
    "model:activate": "Activate or retire a model version",
    "model:delete": "Delete models and versions",
    "training:start": "Queue and start training runs",
    "training:stop": "Pause, resume or cancel training runs",
    "testing:run": "Run model evaluation",
    "experiment:read": "View experiments and their metrics",
    "experiment:write": "Create and edit experiments",
    "report:generate": "Generate and export reports",
    "hardware:read": "View hardware telemetry",
    "detection:run": "Run live detection sessions",
    "log:read": "View application and system logs",
}

ROLE_DESCRIPTIONS: dict[str, str] = {
    "administrator": "Full access including users, database and system settings",
    "ai_engineer": "Datasets, training, testing and model management",
    "researcher": "Read data, run experiments and export reports",
    "viewer": "Read-only access to dashboards and results",
}

DEFAULT_SETTINGS: dict[str, tuple[str, str]] = {
    "app.theme": ("dark", "Default interface theme"),
    "app.language": ("en", "Interface language"),
    "app.detector": ("yunet", "Default face detector backend"),
    "app.classifier": ("heuristic", "Default expression classifier backend"),
    "app.privacy.store_frames": ("false", "Whether camera frames may be written to disk"),
    "app.privacy.store_snapshots": ("false", "Whether face snapshots may be written to disk"),
    "app.telemetry.enabled": ("false", "Whether usage telemetry is collected"),
}


def seed_reference_data(database: Database) -> dict[str, int]:
    """Insert reference rows that are missing. Safe to run repeatedly."""
    counts = {
        "roles": seed_roles(database),
        "permissions": seed_permissions(database),
        "emotion_classes": seed_emotion_classes(database),
        "settings": seed_settings(database),
    }
    counts["role_permissions"] = seed_role_permissions(database)
    LOG.info("seeded reference data: %s", counts)
    return counts


def seed_roles(database: Database) -> int:
    existing = {
        str(row["name"])
        for row in database.query("SELECT name FROM roles")
    }
    missing = [(name, ROLE_DESCRIPTIONS.get(name, "")) for name in taxonomy_default_roles() if name not in existing]
    if not missing:
        return 0
    database.executemany(
        "INSERT INTO roles (name, description) VALUES (%s, %s)", missing
    )
    return len(missing)


def taxonomy_default_roles() -> tuple[str, ...]:
    from visionai.security.passwords import known_roles

    return known_roles()


def seed_permissions(database: Database) -> int:
    existing = {
        str(row["code"])
        for row in database.query("SELECT code FROM permissions")
    }
    rows = [
        (permission.value, PERMISSION_DESCRIPTIONS.get(permission.value, ""))
        for permission in Permission
        if permission.value not in existing
    ]
    if not rows:
        return 0
    database.executemany(
        "INSERT INTO permissions (code, description) VALUES (%s, %s)", rows
    )
    return len(rows)


def seed_role_permissions(database: Database) -> int:
    role_ids = {
        str(row["name"]): int(row["id"])
        for row in database.query("SELECT id, name FROM roles")
    }
    permission_ids = {
        str(row["code"]): int(row["id"])
        for row in database.query("SELECT id, code FROM permissions")
    }
    existing = {
        (int(row["role_id"]), int(row["permission_id"]))
        for row in database.query("SELECT role_id, permission_id FROM role_permissions")
    }
    rows: list[tuple[int, int]] = []
    for role, permissions in ((name, permissions_for_role(name)) for name in role_ids):
        role_id = role_ids[role]
        for permission in permissions:
            permission_id = permission_ids.get(permission.value)
            if permission_id is not None and (role_id, permission_id) not in existing:
                rows.append((role_id, permission_id))
    if not rows:
        return 0
    database.executemany(
        "INSERT INTO role_permissions (role_id, permission_id) VALUES (%s, %s)", rows
    )
    return len(rows)


def seed_emotion_classes(database: Database) -> int:
    """Load the seven primary classes plus the documented extended set.

    The extended classes are inserted as inactive: they are part of the
    taxonomy, but they are not enabled until a model that predicts them exists.
    """
    existing = {
        str(row["name"])
        for row in database.query("SELECT name FROM emotion_classes")
    }
    rows: list[tuple] = []
    for order, entry in enumerate(taxonomy.all_classes()):
        if entry.name in existing:
            continue
        rows.append(
            (
                entry.name,
                entry.name.title(),
                entry.emoji,
                ",".join(str(channel) for channel in entry.color),
                entry.group,
                entry.description,
                0 if entry in taxonomy.EXTENDED_CLASSES else 1,
                order,
            )
        )
    if not rows:
        return 0
    database.executemany(
        "INSERT INTO emotion_classes "
        "(name, display_name, emoji, color_rgb, polarity, description, is_active, sort_order) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        rows,
    )
    return len(rows)


def seed_settings(database: Database) -> int:
    existing = {
        str(row["key_name"])
        for row in database.query("SELECT key_name FROM settings")
    }
    rows = [
        (key, value, "app", description)
        for key, (value, description) in DEFAULT_SETTINGS.items()
        if key not in existing
    ]
    if not rows:
        return 0
    database.executemany(
        "INSERT INTO settings (key_name, value, scope, description) VALUES (%s, %s, %s, %s)",
        rows,
    )
    return len(rows)


def active_class_names(database: Database) -> tuple[str, ...]:
    """Expression names a model may predict right now."""
    rows = database.query(
        "SELECT name FROM emotion_classes WHERE is_active = 1 ORDER BY sort_order, name"
    )
    return tuple(str(row["name"]) for row in rows)


def all_class_names(database: Database) -> tuple[str, ...]:
    rows = database.query("SELECT name FROM emotion_classes ORDER BY sort_order, name")
    return tuple(str(row["name"]) for row in rows)


def set_setting(database: Database, key: str, value: str, scope: str = "app") -> None:
    payload = value if isinstance(value, str) else json.dumps(value)
    updated = database.execute(
        "UPDATE settings SET value = %s, updated_at = %s WHERE key_name = %s",
        (payload, "now", key),
    )
    if not updated:
        database.execute(
            "INSERT INTO settings (key_name, value, scope) VALUES (%s, %s, %s)",
            (key, payload, scope),
        )
