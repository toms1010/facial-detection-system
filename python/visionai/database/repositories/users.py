"""Users, roles, profiles and application settings."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from typing import Any

from visionai.database.driver import DatabaseError
from visionai.database.repositories.base import (
    NotFoundError,
    Repository,
    ValidationError,
    decode_json,
    encode_json,
    now,
)
from visionai.security.passwords import (
    Permission,
    hash_password,
    needs_rehash,
    permissions_for_role,
    verify_password,
)

LOG = logging.getLogger(__name__)

ACTIVE = "active"
SUSPENDED = "suspended"
PENDING = "pending"

#: Columns safe to hand to the UI. password_hash is never selected anywhere.
PUBLIC_COLUMNS = (
    "u.id, u.username, u.email, u.role_id, u.status, u.last_login_at, "
    "u.created_at, u.updated_at, r.name AS role"
)
USER_FROM = "users u JOIN roles r ON r.id = u.role_id"


class UserRepository(Repository):
    """Accounts and their profiles. Hashes never leave this class."""

    table = "users"

    def create(
        self,
        username: str,
        email: str,
        password: str,
        role: str = "viewer",
        status: str = ACTIVE,
        profile: Mapping | None = None,
    ) -> int:
        """Create an account. The password is hashed here and discarded."""
        username = str(username).strip()
        email = str(email).strip().lower()
        if not username or "@" not in email:
            raise ValidationError("username and a valid email are required")
        if self.find_by_username(username) is not None:
            raise ValidationError(f"username {username!r} is already taken")
        if self.find_by_email(email) is not None:
            raise ValidationError(f"email {email!r} is already registered")

        role_id = self._role_id(role)
        user_id = self.db.insert(
            "INSERT INTO users (username, email, password_hash, role_id, status, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (username, email, hash_password(password), role_id, status, now(), now()),
        )
        if profile:
            self.save_profile(
                user_id,
                display_name=profile.get("display_name", ""),
                occupation=profile.get("occupation", ""),
                theme=profile.get("theme", "dark"),
                language=profile.get("language", "en"),
                preferences=profile.get("preferences"),
            )
        return user_id

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        """Verify credentials and stamp the login. Returns None on any failure."""
        row = self.db.query_one(
            "SELECT id, username, email, password_hash, role_id, status FROM users "
            "WHERE username = %s",
            (str(username).strip(),),
        )
        if row is None:
            self._equalise(str(username), password)
            return None
        if not verify_password(password, str(row["password_hash"])):
            LOG.info("failed sign-in attempt for %r", row["username"])
            return None
        if str(row["status"]) != ACTIVE:
            LOG.info("sign-in refused for %r: account is %s", row["username"], row["status"])
            return None
        if needs_rehash(str(row["password_hash"])):
            self.set_password(int(row["id"]), password)
            LOG.info("upgraded the stored hash for %r", row["username"])
        self.db.execute(
            "UPDATE users SET last_login_at = %s WHERE id = %s", (now(), row["id"])
        )
        return self.get(int(row["id"]))

    @staticmethod
    def _equalise(username: str, password: str) -> None:
        """Spend comparable time on an unknown user so timing does not leak."""
        with contextlib.suppress(Exception):
            verify_password(password or "", hash_password("timing-equaliser-placeholder"))
        del username

    def get(self, user_id: int) -> dict[str, Any]:
        row = self.db.query_one(
            f"SELECT {PUBLIC_COLUMNS} FROM {USER_FROM} WHERE u.id = %s", (user_id,)
        )
        if row is None:
            raise NotFoundError(f"user {user_id} does not exist")
        return dict(row)

    def find_by_username(self, username: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            f"SELECT {PUBLIC_COLUMNS} FROM {USER_FROM} WHERE u.username = %s",
            (str(username).strip(),),
        )
        return dict(row) if row else None

    def find_by_email(self, email: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            f"SELECT {PUBLIC_COLUMNS} FROM {USER_FROM} WHERE u.email = %s",
            (str(email).strip().lower(),),
        )
        return dict(row) if row else None

    def list_users(
        self,
        role: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """A page of accounts, filtered by role and/or status."""
        clauses: list[str] = []
        params: list[Any] = []
        if role:
            clauses.append("r.name = %s")
            params.append(role)
        if status:
            clauses.append("u.status = %s")
            params.append(status)
        where = " AND ".join(clauses)
        page = max(1, int(page))
        page_size = max(1, min(500, int(page_size)))
        offset = (page - 1) * page_size
        condition = f"WHERE {where}" if where else ""
        total = int(
            self.db.scalar(
                f"SELECT COUNT(*) AS n FROM {USER_FROM} {condition}", params, default=0
            )
            or 0
        )
        rows = self.db.query(
            f"SELECT {PUBLIC_COLUMNS} FROM {USER_FROM} {condition} "
            f"ORDER BY u.username LIMIT {page_size} OFFSET {offset}",
            params,
        )
        return {
            "items": [dict(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, -(-total // page_size)),
        }

    def count_by_role(self) -> dict[str, int]:
        """Accounts per role, for the dashboard counters."""
        rows = self.db.query(
            "SELECT r.name AS role, COUNT(u.id) AS total FROM roles r "
            "LEFT JOIN users u ON u.role_id = r.id GROUP BY r.name ORDER BY r.name"
        )
        return {str(row["role"]): int(row["total"] or 0) for row in rows}

    def set_password(self, user_id: int, password: str) -> None:
        updated = self.db.execute(
            "UPDATE users SET password_hash = %s, updated_at = %s WHERE id = %s",
            (hash_password(password), now(), user_id),
        )
        if not updated:
            raise NotFoundError(f"user {user_id} does not exist")

    def set_status(self, user_id: int, status: str) -> None:
        if status not in (ACTIVE, SUSPENDED, PENDING):
            raise ValidationError(f"unknown account status {status!r}")
        updated = self.db.execute(
            "UPDATE users SET status = %s, updated_at = %s WHERE id = %s",
            (status, now(), user_id),
        )
        if not updated:
            raise NotFoundError(f"user {user_id} does not exist")

    def set_role(self, user_id: int, role: str) -> None:
        updated = self.db.execute(
            "UPDATE users SET role_id = %s, updated_at = %s WHERE id = %s",
            (self._role_id(role), now(), user_id),
        )
        if not updated:
            raise NotFoundError(f"user {user_id} does not exist")

    def delete(self, user_id: int) -> None:
        if not self.db.execute("DELETE FROM users WHERE id = %s", (user_id,)):
            raise NotFoundError(f"user {user_id} does not exist")

    def permissions(self, user_id: int) -> frozenset[Permission]:
        rows = self.db.query(
            "SELECT p.code FROM permissions p "
            "JOIN role_permissions rp ON rp.permission_id = p.id "
            "JOIN users u ON u.role_id = rp.role_id WHERE u.id = %s",
            (user_id,),
        )
        granted = set()
        for row in rows:
            try:
                granted.add(Permission(str(row["code"])))
            except ValueError:
                LOG.warning("unknown permission code in the database: %r", row["code"])
        return frozenset(granted)

    def has_permission(self, user_id: int, permission: Permission | str) -> bool:
        try:
            wanted = permission if isinstance(permission, Permission) else Permission(permission)
        except ValueError:
            return False
        return wanted in self.permissions(user_id)

    def require_permission(self, user_id: int, permission: Permission | str) -> None:
        if not self.has_permission(user_id, permission):
            raise DatabaseError(
                f"user {user_id} is not permitted to perform {permission}"
            )

    def profile(self, user_id: int) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
        if row is None:
            return {
                "user_id": user_id,
                "display_name": "",
                "occupation": "",
                "avatar_path": None,
                "theme": "dark",
                "language": "en",
                "preferences": {},
            }
        row["preferences"] = decode_json(row.get("preferences"), {})
        return dict(row)

    def save_profile(
        self,
        user_id: int,
        display_name: str = "",
        occupation: str = "",
        theme: str = "dark",
        language: str = "en",
        preferences: Mapping | None = None,
        avatar_path: str | None = None,
    ) -> None:
        encoded = encode_json(dict(preferences) if preferences else None)
        existing = self.db.query_one(
            "SELECT 1 AS ok FROM user_profiles WHERE user_id = %s", (user_id,)
        )
        if existing:
            self.db.execute(
                "UPDATE user_profiles SET display_name = %s, occupation = %s, theme = %s, "
                "language = %s, preferences = %s, avatar_path = %s, updated_at = %s "
                "WHERE user_id = %s",
                (display_name, occupation, theme, language, encoded, avatar_path, now(), user_id),
            )
            return
        self.db.execute(
            "INSERT INTO user_profiles (user_id, display_name, occupation, theme, language, "
            "preferences, avatar_path, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user_id, display_name, occupation, theme, language, encoded, avatar_path, now(), now()),
        )

    def _role_id(self, role: str) -> int:
        row = self.db.query_one("SELECT id FROM roles WHERE name = %s", (str(role).strip(),))
        if row is None:
            raise ValidationError(f"role {role!r} is not registered")
        return int(row["id"])


class RoleRepository(Repository):
    """Roles and the permissions granted to them."""

    table = "roles"

    def list_roles(self) -> list[dict[str, Any]]:
        rows = self.fetch_all(order_by="name")
        for row in rows:
            row["permissions"] = sorted(
                str(item["code"])
                for item in self.db.query(
                    "SELECT p.code FROM permissions p "
                    "JOIN role_permissions rp ON rp.permission_id = p.id WHERE rp.role_id = %s",
                    (row["id"],),
                )
            )
        return rows

    def get_role(self, name: str) -> dict[str, Any] | None:
        for role in self.list_roles():
            if role["name"] == str(name):
                return role
        return None

    def effective_permissions(self, name: str) -> frozenset[Permission]:
        """Permissions the code says a role should have."""
        return permissions_for_role(name)

    def grant(self, role: str, permission: Permission) -> None:
        role_row = self.db.query_one("SELECT id FROM roles WHERE name = %s", (role,))
        permission_row = self.db.query_one(
            "SELECT id FROM permissions WHERE code = %s", (permission.value,)
        )
        if role_row is None or permission_row is None:
            raise ValidationError(f"cannot grant {permission.value} to {role!r}")
        self.db.execute(
            "INSERT INTO role_permissions (role_id, permission_id) VALUES (%s, %s)",
            (role_row["id"], permission_row["id"]),
        )

    def revoke(self, role: str, permission: Permission) -> None:
        """Remove a grant. Uses subqueries so it works on MySQL and SQLite alike."""
        self.db.execute(
            "DELETE FROM role_permissions WHERE role_id = "
            "(SELECT id FROM roles WHERE name = %s) AND permission_id = "
            "(SELECT id FROM permissions WHERE code = %s)",
            (role, permission.value),
        )

    def sync_role(self, role: str) -> int:
        """Make the stored grants match the code-defined set for ``role``."""
        role_row = self.db.query_one("SELECT id FROM roles WHERE name = %s", (role,))
        if role_row is None:
            raise ValidationError(f"role {role!r} is not registered")
        self.db.execute("DELETE FROM role_permissions WHERE role_id = %s", (role_row["id"],))
        granted = 0
        for permission in permissions_for_role(role):
            permission_row = self.db.query_one(
                "SELECT id FROM permissions WHERE code = %s", (permission.value,)
            )
            if permission_row is None:
                continue
            self.db.execute(
                "INSERT INTO role_permissions (role_id, permission_id) VALUES (%s, %s)",
                (role_row["id"], permission_row["id"]),
            )
            granted += 1
        return granted


class SettingsRepository(Repository):
    """Application settings, stored as text so both dialects agree."""

    table = "settings"
    key_column = "key_name"

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one(
            "SELECT value FROM settings WHERE key_name = %s", (str(key),)
        )
        if row is None:
            return default
        value = row.get("value")
        return default if value is None else value

    def set(self, key: str, value: Any, scope: str = "app") -> None:
        payload = value if isinstance(value, str) else encode_json(value)
        updated = self.db.execute(
            "UPDATE settings SET value = %s, scope = %s, updated_at = %s WHERE key_name = %s",
            (payload, scope, now(), key),
        )
        if not updated:
            self.db.execute(
                "INSERT INTO settings (key_name, value, scope, updated_at) VALUES (%s, %s, %s, %s)",
                (key, payload, scope, now()),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            str(row["key_name"]): row.get("value")
            for row in self.fetch_all(order_by="key_name")
        }
