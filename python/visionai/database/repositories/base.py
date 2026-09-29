"""Shared repository behaviour.

Every repository is a thin, explicit set of SQL statements. There is no ORM and
no query builder: the queries are readable, they are always parameterised, and
what runs against the database is what is written here.

The base class provides the small amount of scaffolding each repository repeats:
pagination, row-counting, JSON column encoding and consistent ``not found``
errors.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any

from visionai.database.driver import Database, DatabaseError

LOG = logging.getLogger(__name__)


class NotFoundError(DatabaseError):
    """A requested row does not exist."""


class ValidationError(DatabaseError):
    """Input failed a repository-level rule before touching the database."""


def now() -> str:
    """A timestamp in the format both dialects accept for DATETIME."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def encode_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def decode_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class Repository:
    """Base class holding the database handle and generic helpers."""

    table: str = ""

    def __init__(self, database: Database) -> None:
        self.db = database

    def count(self, where: str = "", params: Sequence[Any] = ()) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {self.table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.db.scalar(sql, params, default=0) or 0)

    def exists(self, where: str, params: Sequence[Any] = ()) -> bool:
        sql = f"SELECT 1 AS ok FROM {self.table} WHERE {where} LIMIT 1"
        return self.db.query_one(sql, params) is not None

    def delete_where(self, where: str, params: Sequence[Any] = ()) -> int:
        return self.db.execute(f"DELETE FROM {self.table} WHERE {where}", params)

    def fetch_all(
        self,
        where: str = "",
        params: Sequence[Any] = (),
        order_by: str = "",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {self.table}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return self.db.query(sql, params)

    def fetch_one(self, where: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        sql = f"SELECT * FROM {self.table} WHERE {where} LIMIT 1"
        return self.db.query_one(sql, params)

    def require(self, row_id: int, label: str = "row") -> dict[str, Any]:
        row = self.db.query_one(f"SELECT * FROM {self.table} WHERE id = %s", (row_id,))
        if row is None:
            raise NotFoundError(f"{label} {row_id} does not exist")
        return row

    def paginate(
        self,
        where: str = "",
        params: Sequence[Any] = (),
        order_by: str = "id DESC",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """A page of rows plus the totals a table view needs."""
        page = max(1, int(page))
        page_size = max(1, min(500, int(page_size)))
        offset = (page - 1) * page_size
        rows = self.fetch_all(where, params, order_by, limit=page_size, offset=offset)
        return {
            "items": rows,
            "page": page,
            "page_size": page_size,
            "total": self.count(where, params),
            "total_pages": max(1, -(-self.count(where, params) // page_size)),
        }

    def bulk_insert(self, sql: str, rows: Iterable[Sequence[Any]]) -> int:
        return self.db.executemany(sql, list(rows))
