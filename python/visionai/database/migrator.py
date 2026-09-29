"""Schema migrations.

Migrations are numbered ``.sql`` files in this directory and are applied in
order, each exactly once, recorded in ``schema_migrations``. Nothing about the
schema is created by hand: :func:`status` reports what is pending so the
application can refuse to run against an out-of-date database rather than
failing later on a missing column.

The SQL is written portably. :func:`to_mysql` adds the table options MySQL
needs and rewrites the few constructs SQLite and MySQL spell differently, so a
single file defines both databases.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from visionai.database.driver import Database, DatabaseError, ddl_suffix

LOG = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
MIGRATION_TABLE = "schema_migrations"

_FILENAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
_CREATE_TABLE = re.compile(
    r"^CREATE TABLE\s+(?P<name>\w+)\s*\($", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True)
class Migration:
    """One numbered migration file."""

    version: int
    name: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def checksum(self) -> str:
        return hashlib.sha256(self.read().encode("utf-8")).hexdigest()[:16]

    def statements(self) -> list[str]:
        return split_statements(self.read())


def discover(directory: Path | None = None) -> list[Migration]:
    """All migration files, sorted by version."""
    root = directory or MIGRATIONS_DIR
    found: list[Migration] = []
    for path in sorted(root.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if not match:
            LOG.warning("ignoring migration file with an unexpected name: %s", path.name)
            continue
        found.append(Migration(int(match.group(1)), match.group(2), path))
    found.sort(key=lambda migration: migration.version)

    seen: set[int] = set()
    for migration in found:
        if migration.version in seen:
            raise DatabaseError(f"duplicate migration version {migration.version}")
        seen.add(migration.version)
    return found


def split_statements(script: str) -> list[str]:
    """Split a SQL script into statements, dropping comments and blank fragments."""
    cleaned_lines = [
        line for line in script.splitlines() if not line.strip().startswith("--")
    ]
    cleaned = "\n".join(cleaned_lines)
    statements: list[str] = []
    for chunk in cleaned.split(";"):
        statement = chunk.strip()
        if statement:
            statements.append(statement)
    return statements


def to_mysql(statement: str) -> str:
    """Translate one portable statement into MySQL syntax.

    Only the constructs that genuinely differ are rewritten: the
    ``AUTOINCREMENT`` primary key and any ``AUTOINCREMENT`` column.
    """
    result = statement
    result = re.sub(
        r"\bINTEGER PRIMARY KEY AUTOINCREMENT\b",
        "INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bSMALLINT\b",
        "TINYINT",
        result,
        flags=re.IGNORECASE,
    )
    if _CREATE_TABLE.match(result.strip() + "("):
        result = result.rstrip() + ddl_suffix("mysql")
    return result


def to_sqlite(statement: str) -> str:
    """Translate one portable statement into SQLite syntax."""
    result = re.sub(
        r"\bINTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY\b",
        "INTEGER PRIMARY KEY AUTOINCREMENT",
        statement,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\bTINYINT\b", "SMALLINT", result, flags=re.IGNORECASE)


def translate(statement: str, dialect: str) -> str:
    return to_mysql(statement) if dialect == "mysql" else to_sqlite(statement)


@dataclass
class MigrationStatus:
    """What is applied and what is still pending."""

    applied: list[dict]
    pending: list[Migration]
    drifted: list[str]

    @property
    def is_up_to_date(self) -> bool:
        return not self.pending and not self.drifted

    def describe(self) -> str:
        lines = [f"applied: {len(self.applied)}", f"pending: {len(self.pending)}"]
        if self.pending:
            lines.append("pending migrations: " + ", ".join(m.label for m in self.pending))
        if self.drifted:
            lines.append("checksum drift: " + ", ".join(self.drifted))
        return "\n".join(lines)


class Migrator:
    """Applies pending migrations and reports status."""

    def __init__(self, database: Database, directory: Path | None = None) -> None:
        self.database = database
        self.directory = directory or MIGRATIONS_DIR
        self.migrations = discover(self.directory)

    def ensure_table(self) -> None:
        self.database.execute(
            f"CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} ("
            "version INTEGER NOT NULL PRIMARY KEY, "
            "name VARCHAR(64) NOT NULL, "
            "checksum VARCHAR(32) NOT NULL DEFAULT '', "
            "applied_at DATETIME NOT NULL, "
            "duration_ms REAL NOT NULL DEFAULT 0)"
            + ddl_suffix(self.database.dialect)
        )

    def applied_versions(self) -> dict[int, dict]:
        self.ensure_table()
        rows = self.database.query(
            f"SELECT version, name, checksum, applied_at FROM {MIGRATION_TABLE} ORDER BY version"
        )
        return {int(row["version"]): row for row in rows}

    def status(self) -> MigrationStatus:
        applied = self.applied_versions()
        known = {migration.version: migration for migration in self.migrations}
        pending = [m for version, m in sorted(known.items()) if version not in applied]

        drifted: list[str] = []
        for version, row in applied.items():
            migration = known.get(version)
            if migration is None:
                drifted.append(f"{version} (not present in the migrations directory)")
            elif row.get("checksum") and row["checksum"] != migration.checksum():
                drifted.append(f"{migration.label} (file changed after being applied)")

        return MigrationStatus(
            applied=[applied[v] for v in sorted(applied)],
            pending=pending,
            drifted=drifted,
        )

    def migrate(self, target: int | None = None, dry_run: bool = False) -> list[str]:
        """Apply pending migrations up to ``target`` (default: all).

        Returns the labels of the migrations applied. Each migration runs in its
        own transaction so a failure leaves the schema at the last good version.
        """
        self.ensure_table()
        applied = self.applied_versions()
        performed: list[str] = []
        dialect = self.database.dialect

        for migration in self.migrations:
            if migration.version in applied:
                continue
            if target is not None and migration.version > target:
                break
            if dry_run:
                performed.append(migration.label)
                continue

            started = time.perf_counter()
            LOG.info("applying migration %s", migration.label)
            try:
                with self.database.transaction():
                    for statement in migration.statements():
                        self.database.execute(translate(statement, dialect))
                    duration_ms = (time.perf_counter() - started) * 1000.0
                    self.database.execute(
                        f"INSERT INTO {MIGRATION_TABLE} "
                        "(version, name, checksum, applied_at, duration_ms) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            migration.version,
                            migration.name,
                            migration.checksum(),
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            round(duration_ms, 3),
                        ),
                    )
            except DatabaseError as exc:
                raise DatabaseError(
                    f"migration {migration.label} failed: {exc}. The schema is at "
                    f"version {max(applied) if applied else 0}."
                ) from exc
            performed.append(migration.label)

        if performed and not dry_run:
            LOG.info("applied %d migration(s)", len(performed))
        return performed

    def rollback_last(self) -> str | None:
        """Remove the most recently applied migration record.

        MySQL DDL is not transactional, so this only drops the tracking row. It
        exists so an operator can see what happened, not to undo DDL silently.
        """
        applied = self.applied_versions()
        if not applied:
            return None
        version = max(applied)
        self.database.execute(
            f"DELETE FROM {MIGRATION_TABLE} WHERE version = %s", (version,)
        )
        return str(applied[version].get("name"))

    def seed(self) -> dict[str, int]:
        """Insert the default roles, permissions and expression classes."""
        from visionai.database.seed import seed_reference_data

        return seed_reference_data(self.database)

    def describe(self) -> dict:
        status = self.status()
        return {
            "directory": str(self.directory),
            "total": len(self.migrations),
            "applied": len(status.applied),
            "pending": [m.label for m in status.pending],
            "drifted": status.drifted,
            "up_to_date": status.is_up_to_date,
        }


def apply_all(database: Database, directory: Path | None = None, seed: bool = True) -> dict:
    """Create the schema and load reference data. The usual entry point."""
    migrator = Migrator(database, directory)
    applied = migrator.migrate()
    seeded = migrator.seed() if seed else {}
    return {"applied": applied, "seeded": seeded, "status": migrator.describe()}


def require_up_to_date(database: Database, directory: Path | None = None) -> None:
    """Raise if the schema is behind, so the app never runs against a stale one."""
    status = Migrator(database, directory).status()
    if not status.is_up_to_date:
        raise DatabaseError(
            "the database schema is out of date:\n" + status.describe()
        )


def iter_labels(migrations: Iterable[Migration]) -> list[str]:
    return [migration.label for migration in migrations]
