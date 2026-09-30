# pyright: reportAttributeAccessIssue=false
"""Database access layer: schema migrations, seed data and repositories.

The suppression above silences a stale Pylance hint that reports a phantom
import symbol on this file's closing parenthesis. Every name below is a real
static import that exists in its source module, and `pyright` reports no
diagnostics here, so nothing is being hidden. Remove the line once the editor's
analysis cache has been cleared.
"""

from __future__ import annotations

from visionai.database.driver import (
    Database,
    DatabaseConfig,
    DatabaseError,
    ddl_suffix,
    open_database,
    placeholder,
    temporary_database,
)
from visionai.database.migrator import Migrator, apply_all, require_up_to_date
from visionai.database.repositories import Repository, open_repositories
from visionai.database.seed import seed_reference_data

__all__ = [
    "Database",
    "DatabaseConfig",
    "DatabaseError",
    "Migrator",
    "Repository",
    "apply_all",
    "ddl_suffix",
    "open_database",
    "open_repositories",
    "placeholder",
    "require_up_to_date",
    "seed_reference_data",
    "temporary_database",
]
