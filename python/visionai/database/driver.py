"""Database connection management and dialect abstraction.

One schema, two dialects. MySQL 8 is the production target; SQLite is the
throwaway store the test-suite uses so the whole data layer is exercised without
a database server. The differences that actually matter are small and are
handled here rather than leaking into the repositories:

* placeholders - PyMySQL uses ``%s``, sqlite3 uses ``?``;
* autoincrement - declared in DDL, never referenced by name in queries, so
  repositories always read back the generated key;
* row access - PyMySQL returns tuples by default, sqlite3 too, so a shared
  ``dict_row`` factory is configured for both.

Everything else is written to be portable: ``DATETIME``/``NUMERIC`` rather than
MySQL-only types, no ``ENUM``, no inline ``ENGINE=`` clauses outside the
dialect-specific suffix appended by :func:`ddl_suffix`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionai.ai.errors import ConfigError

LOG = logging.getLogger(__name__)

DEFAULT_CHARSET = "utf8mb4"


class DatabaseError(RuntimeError):
    """Any database failure, so callers need not import a driver."""


class ConnectionError_(DatabaseError):
    """The database could not be reached or authenticated against."""


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings. The password is never logged or serialised."""

    host: str = "127.0.0.1"
    port: int = 3306
    database: str = "linux_ai_vision"
    user: str = "visionai"
    password: str = ""
    charset: str = DEFAULT_CHARSET
    connect_timeout: int = 5
    sqlite_path: str | None = None
    echo: bool = False

    @property
    def is_sqlite(self) -> bool:
        """True when a local store was requested instead of MySQL.

        ``sqlite_path is None`` means "use MySQL", which is the default. A value
        of ``":memory:"`` is a real, throwaway SQLite database.
        """
        return self.sqlite_path is not None

    def safe(self) -> dict[str, Any]:
        """A description safe to log or show in the UI: no password, ever."""
        if self.is_sqlite:
            return {
                "dialect": "sqlite",
                "host": "local",
                "port": 0,
                "database": self.sqlite_path,
                "user": "n/a",
                "password": "n/a",
            }
        return {
            "dialect": "mysql",
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": "***" if self.password else "(unset)",
        }

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> DatabaseConfig:
        """Read MYSQL_* environment variables, never accepting a default password."""
        env = environ if environ is not None else dict(os.environ)
        sqlite_path = env.get("VISIONAI_SQLITE_PATH", "").strip() or None
        return cls(
            host=env.get("MYSQL_HOST", "127.0.0.1"),
            port=_as_int(env.get("MYSQL_PORT"), 3306),
            database=env.get("MYSQL_DATABASE", "linux_ai_vision"),
            user=env.get("MYSQL_USER", "visionai"),
            password=env.get("MYSQL_PASSWORD", ""),
            charset=env.get("MYSQL_CHARSET", DEFAULT_CHARSET),
            connect_timeout=_as_int(env.get("MYSQL_CONNECT_TIMEOUT"), 5),
            sqlite_path=sqlite_path,
        )

    def validate(self) -> DatabaseConfig:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"MYSQL_PORT {self.port} is not a valid port")
        if not self.database:
            raise ConfigError("MYSQL_DATABASE must not be empty")
        if not self.is_sqlite and not self.user:
            raise ConfigError("MYSQL_USER must not be empty")
        return self


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def ddl_suffix(dialect: str) -> str:
    """Table options appended to MySQL DDL and omitted elsewhere."""
    if dialect == "mysql":
        return " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    return ""


def placeholder(dialect: str) -> str:
    return "%s" if dialect == "mysql" else "?"


class Database:
    """A thin, thread-safe connection wrapper.

    sqlite3 connections are not safe to share across threads, so one connection
    is created per thread and cached. PyMySQL connections are pooled the same way
    for consistency.
    """

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = (config or DatabaseConfig()).validate()
        self.dialect = "sqlite" if self.config.is_sqlite else "mysql"
        self._local = threading.local()
        self._lock = threading.Lock()
        self._closed = False
        self._rowcount = 0

    def connect(self) -> Any:
        if self._closed:
            raise DatabaseError("database has been closed")
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            return existing
        connection = self._open()
        with self._lock:
            existing = getattr(self._local, "connection", None)
            if existing is not None:
                self._close_one(connection)
                return existing
            self._local.connection = connection
        return connection

    def _open(self) -> Any:
        if self.dialect == "sqlite":
            return self._open_sqlite()
        return self._open_mysql()

    def _open_sqlite(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.config.sqlite_path,
            timeout=self.config.connect_timeout,
            check_same_thread=False,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _open_mysql(self) -> Any:
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            raise DatabaseError(
                "PyMySQL is not installed; run `pip install pymysql` or set "
                "VISIONAI_SQLITE_PATH for a local store"
            ) from exc
        try:
            return pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                charset=self.config.charset,
                connect_timeout=self.config.connect_timeout,
                cursorclass=DictCursor,
                autocommit=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError_(
                f"could not connect to {self.config.host}:{self.config.port}/"
                f"{self.config.database} as {self.config.user}: {exc}"
            ) from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run a SELECT and return every row as a dict."""
        statement = self._adapt(sql)
        connection = self.connect()
        try:
            cursor = connection.cursor()
            cursor.execute(statement, tuple(params))
            rows = cursor.fetchall()
            cursor.close()
        except DatabaseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{type(exc).__name__} executing {sql[:120]!r}: {exc}") from exc
        return [self._as_dict(row) for row in rows]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if not row:
            return default
        value = next(iter(row.values()), default)
        return default if value is None else value

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run an INSERT/UPDATE/DELETE and return the affected row count."""
        self._execute_cursor(sql, params)
        return self._rowcount

    def insert(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run an INSERT and return the generated primary key."""
        cursor = self._execute_cursor(sql, params)
        generated = getattr(cursor, "lastrowid", 0) or 0
        if generated:
            return int(generated)
        return int(self.scalar("SELECT LAST_INSERT_ID() AS _id", default=0) or 0)

    def _execute_cursor(self, sql: str, params: Sequence[Any] = ()) -> Any:
        statement = self._adapt(sql)
        connection = self.connect()
        try:
            cursor = connection.cursor()
            cursor.execute(statement, tuple(params))
            self._rowcount = int(cursor.rowcount or 0)
            return cursor
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{type(exc).__name__} executing {sql[:120]!r}: {exc}") from exc

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> int:
        if not rows:
            return 0
        statement = self._adapt(sql)
        connection = self.connect()
        try:
            cursor = connection.cursor()
            cursor.executemany(statement, [tuple(row) for row in rows])
            count = cursor.rowcount
            cursor.close()
            return int(count)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"{type(exc).__name__} bulk-executing: {exc}") from exc

    def script(self, statements: Sequence[str]) -> None:
        """Execute a sequence of DDL statements (used by the migrator)."""
        for statement in statements:
            cleaned = _clean_statement(statement)
            if cleaned:
                self.execute(cleaned)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[Database]:
        """Explicit transaction. SQLite is in autocommit, so it is toggled here."""
        connection = self.connect()
        if self.dialect == "sqlite":
            connection.execute("BEGIN")
            try:
                yield self
            except Exception:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return
        try:
            connection.begin()
            yield self
        except Exception:
            connection.rollback()
            raise
        connection.commit()

    def _adapt(self, sql: str) -> str:
        if self.dialect == "mysql":
            return sql
        return sql.replace("%s", "?")

    @staticmethod
    def _as_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        return dict(row)

    @staticmethod
    def _close_one(connection: Any) -> None:
        with contextlib.suppress(Exception):
            connection.close()

    def close(self) -> None:
        with self._lock:
            connection = getattr(self._local, "connection", None)
            if connection is not None:
                self._close_one(connection)
                self._local.connection = None
            self._closed = True

    def ping(self) -> bool:
        try:
            self.query_one("SELECT 1 AS ok")
        except DatabaseError:
            return False
        return True

    def table_names(self) -> list[str]:
        if self.dialect == "mysql":
            rows = self.query(
                "SELECT TABLE_NAME AS name FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = %s",
                (self.config.database,),
            )
        else:
            rows = self.query("SELECT name FROM sqlite_master WHERE type = 'table'")
        return sorted(str(row["name"]) for row in rows)

    def describe(self) -> dict[str, Any]:
        """Connection status for the database monitor page. Never leaks secrets."""
        info: dict[str, Any] = {
            "connected": False,
            "dialect": self.dialect,
            "target": self.config.safe(),
            "tables": 0,
            "server_version": "",
            "error": "",
        }
        try:
            info["connected"] = self.ping()
            if info["connected"]:
                info["tables"] = len(self.table_names())
                if self.dialect == "mysql":
                    info["server_version"] = str(
                        self.scalar("SELECT VERSION() AS v", default="")
                    )
                else:
                    info["server_version"] = str(
                        self.scalar("SELECT sqlite_version() AS v", default="")
                    )
        except DatabaseError as exc:
            info["error"] = str(exc)
        return info

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _clean_statement(statement: str) -> str:
    """Strip comments and blank fragments from a SQL script fragment."""
    without_comments = "\n".join(
        line for line in statement.splitlines() if not line.strip().startswith("--")
    )
    return without_comments.strip().rstrip(";")


def open_database(config: DatabaseConfig | None = None) -> Database:
    """Open (but do not verify) a database connection."""
    return Database(config)


def temporary_database(directory: Path | None = None) -> Database:
    """An on-disk SQLite database for tests, removed with its directory."""
    if directory is None:
        return Database(DatabaseConfig(sqlite_path=":memory:"))
    directory.mkdir(parents=True, exist_ok=True)
    return Database(DatabaseConfig(sqlite_path=str(directory / "test.db")))


__all__ = [
    "Database",
    "DatabaseConfig",
    "DatabaseError",
    "ConnectionError_",
    "ddl_suffix",
    "open_database",
    "placeholder",
    "temporary_database",
]
