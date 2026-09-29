"""Authentication, password hashing and role-based access control."""

from __future__ import annotations

from visionai.security.passwords import (
    DEFAULT_ROLES,
    PasswordError,
    Permission,
    generate_password,
    hash_password,
    known_roles,
    needs_rehash,
    permissions_for_role,
    role_has_permission,
    validate_password_strength,
    verify_password,
)

__all__ = [
    "DEFAULT_ROLES",
    "PasswordError",
    "Permission",
    "generate_password",
    "hash_password",
    "known_roles",
    "needs_rehash",
    "permissions_for_role",
    "role_has_permission",
    "validate_password_strength",
    "verify_password",
]
