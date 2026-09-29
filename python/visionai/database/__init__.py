"""Database access layer: schema migrations, seed data and repositories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from visionai.database.driver import (
    Database,
    DatabaseConfig,
    DatabaseError,
    ddl_suffix,
    placeholder,
    temporary_database,
)
from visionai.database.migrator import Migrator, apply_all, require_up_to_date
from visionai.database.seed import seed_reference_data

if TYPE_CHECKING:  # pragma: no cover - resolved lazily at runtime
    from visionai.database.repositories import Repository, open_repositories

__all__ = [
    "Database",
    "DatabaseConfig",
    "DatabaseError",
    "Migrator",
    "Repository",
    "apply_all",
    "ddl_suffix",
    "open_repositories",
    "placeholder",
    "require_up_to_date",
    "seed_reference_data",
    "temporary_database",
]


def __getattr__(name: str) -> Any:
    if name in ("Repository", "open_repositories"):
        import importlib

        return getattr(importlib.import_module("visionai.database.repositories"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
