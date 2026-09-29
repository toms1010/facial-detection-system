"""Password hashing and role-based access control.

Hashing uses PBKDF2-HMAC-SHA256 from the standard library: no third-party
dependency, no native build step, and a stored format that records the
algorithm, cost and salt so parameters can be raised later without invalidating
existing hashes.

Encoded form::

    pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>

Verification is constant-time. This module never logs a password, a hash, or
anything derived from them.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

LOG = logging.getLogger(__name__)

ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 240_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024

ROLE_ADMINISTRATOR = "administrator"
ROLE_AI_ENGINEER = "ai_engineer"
ROLE_RESEARCHER = "researcher"
ROLE_VIEWER = "viewer"

DEFAULT_ROLES: tuple[str, ...] = (
    ROLE_ADMINISTRATOR,
    ROLE_AI_ENGINEER,
    ROLE_RESEARCHER,
    ROLE_VIEWER,
)


class Permission(str, Enum):
    """Every gated action in the platform."""

    USER_MANAGE = "user:manage"
    ROLE_MANAGE = "role:manage"
    DATABASE_MANAGE = "database:manage"
    SETTINGS_MANAGE = "settings:manage"
    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    DATASET_DELETE = "dataset:delete"
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_ACTIVATE = "model:activate"
    MODEL_DELETE = "model:delete"
    TRAINING_START = "training:start"
    TRAINING_STOP = "training:stop"
    TESTING_RUN = "testing:run"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_WRITE = "experiment:write"
    REPORT_GENERATE = "report:generate"
    HARDING_READ = "hardware:read"
    DETECTION_RUN = "detection:run"
    LOG_READ = "log:read"


ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    ROLE_ADMINISTRATOR: frozenset(Permission),
    ROLE_AI_ENGINEER: frozenset(
        {
            Permission.DATASET_READ,
            Permission.DATASET_WRITE,
            Permission.DATASET_DELETE,
            Permission.MODEL_READ,
            Permission.MODEL_WRITE,
            Permission.MODEL_ACTIVATE,
            Permission.MODEL_DELETE,
            Permission.TRAINING_START,
            Permission.TRAINING_STOP,
            Permission.TESTING_RUN,
            Permission.EXPERIMENT_READ,
            Permission.EXPERIMENT_WRITE,
            Permission.REPORT_GENERATE,
            Permission.HARDING_READ,
            Permission.DETECTION_RUN,
            Permission.SETTINGS_MANAGE,
        }
    ),
    ROLE_RESEARCHER: frozenset(
        {
            Permission.DATASET_READ,
            Permission.MODEL_READ,
            Permission.TESTING_RUN,
            Permission.EXPERIMENT_READ,
            Permission.EXPERIMENT_WRITE,
            Permission.REPORT_GENERATE,
            Permission.HARDING_READ,
            Permission.DETECTION_RUN,
        }
    ),
    ROLE_VIEWER: frozenset(
        {
            Permission.DATASET_READ,
            Permission.MODEL_READ,
            Permission.EXPERIMENT_READ,
            Permission.HARDING_READ,
            Permission.DETECTION_RUN,
        }
    ),
}


class PasswordError(ValueError):
    """The supplied password does not meet the policy."""


@dataclass(frozen=True)
class HashedPassword:
    """A parsed stored hash."""

    algorithm: str
    iterations: int
    salt: bytes
    digest: bytes

    def encode(self) -> str:
        return f"{self.algorithm}${self.iterations}${_b64(self.salt)}${_b64(self.digest)}"

    @classmethod
    def decode(cls, encoded: str) -> HashedPassword:
        parts = str(encoded).split("$")
        if len(parts) != 4 or parts[0] != ALGORITHM:
            raise PasswordError("stored password hash is malformed")
        try:
            return cls(
                algorithm=parts[0],
                iterations=int(parts[1]),
                salt=_unb64(parts[2]),
                digest=_unb64(parts[3]),
            )
        except (ValueError, TypeError) as exc:
            raise PasswordError("stored password hash is malformed") from exc

    def verify(self, candidate: str) -> bool:
        computed = hashlib.pbkdf2_hmac(
            "sha256", candidate.encode("utf-8"), self.salt, self.iterations
        )
        return hmac.compare_digest(computed, self.digest)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text + "=" * (-len(text) % 4))


def validate_password_strength(password: str) -> None:
    """Enforce a minimum policy. Raises :class:`PasswordError` when too weak."""
    if not isinstance(password, str):
        raise PasswordError("password must be a string")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters long"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordError("password is unreasonably long")
    if password.strip() != password:
        raise PasswordError("password must not begin or end with whitespace")
    classes = sum(
        bool(check)
        for check in (
            any(c.islower() for c in password),
            any(c.isupper() for c in password),
            any(c.isdigit() for c in password),
            any(not c.isalnum() for c in password),
        )
    )
    if classes < 3:
        raise PasswordError(
            "password must combine at least three of: lower case, upper case, "
            "digits, symbols"
        )


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Hash a password after checking it against the policy."""
    validate_password_strength(password)
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return HashedPassword(ALGORITHM, iterations, salt, digest).encode()


def verify_password(password: str, encoded: str) -> bool:
    """Check a password against a stored hash. Never raises on a bad hash."""
    try:
        return HashedPassword.decode(encoded).verify(password)
    except PasswordError:
        LOG.warning("password verification failed: malformed stored hash")
        return False


def needs_rehash(encoded: str) -> bool:
    """True when a stored hash used weaker parameters than the current default."""
    try:
        return HashedPassword.decode(encoded).iterations < DEFAULT_ITERATIONS
    except PasswordError:
        return True


def generate_password(length: int = 20) -> str:
    """A strong random password, used when an administrator resets an account."""
    alphabet = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789!@#$%^&*-_=+"
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        try:
            validate_password_strength(candidate)
        except PasswordError:
            continue
        return candidate


def permissions_for_role(role: str) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(str(role).strip().lower(), frozenset())


def role_has_permission(role: str, permission: Permission | str) -> bool:
    try:
        wanted = Permission(permission) if not isinstance(permission, Permission) else permission
    except ValueError:
        return False
    return wanted in permissions_for_role(role)


def known_roles() -> tuple[str, ...]:
    return DEFAULT_ROLES


@lru_cache(maxsize=256)
def all_permissions() -> tuple[Permission, ...]:
    return tuple(Permission)


__all__ = [
    "ALGORITHM",
    "DEFAULT_ROLES",
    "HashedPassword",
    "MIN_PASSWORD_LENGTH",
    "PasswordError",
    "Permission",
    "ROLE_PERMISSIONS",
    "all_permissions",
    "generate_password",
    "hash_password",
    "known_roles",
    "needs_rehash",
    "permissions_for_role",
    "role_has_permission",
    "validate_password_strength",
    "verify_password",
]
