-- 0001_core.sql
-- Identity: roles, permissions, users, profiles and application settings.
--
-- Passwords are stored only as PBKDF2-HMAC-SHA256 hashes; no column here ever
-- holds a plaintext password.

CREATE TABLE roles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          VARCHAR(64)  NOT NULL UNIQUE,
    description   VARCHAR(255) NOT NULL DEFAULT '',
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE permissions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    code          VARCHAR(64)  NOT NULL UNIQUE,
    description   VARCHAR(255) NOT NULL DEFAULT ''
);

CREATE TABLE role_permissions (
    role_id        INTEGER NOT NULL,
    permission_id  INTEGER NOT NULL,
    PRIMARY KEY (role_id, permission_id),
    FOREIGN KEY (role_id)       REFERENCES roles(id)       ON DELETE CASCADE,
    FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
);

CREATE TABLE users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       VARCHAR(64)  NOT NULL UNIQUE,
    email          VARCHAR(255) NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    role_id        INTEGER      NOT NULL,
    status         VARCHAR(32)  NOT NULL DEFAULT 'active',
    last_login_at  DATETIME         NULL,
    created_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE RESTRICT,
    CONSTRAINT ck_users_status CHECK (status IN ('active', 'suspended', 'pending'))
);

CREATE TABLE user_profiles (
    user_id        INTEGER PRIMARY KEY,
    display_name   VARCHAR(128) NOT NULL DEFAULT '',
    occupation     VARCHAR(128) NOT NULL DEFAULT '',
    avatar_path    VARCHAR(512)     NULL,
    preferences    TEXT             NULL,
    theme          VARCHAR(32)  NOT NULL DEFAULT 'dark',
    language       VARCHAR(16)  NOT NULL DEFAULT 'en',
    created_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE settings (
    key_name       VARCHAR(128) PRIMARY KEY,
    value          TEXT             NULL,
    scope          VARCHAR(32)  NOT NULL DEFAULT 'app',
    description    VARCHAR(255) NOT NULL DEFAULT '',
    updated_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_users_role ON users (role_id);
CREATE INDEX ix_users_status ON users (status);
