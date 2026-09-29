-- 0002_taxonomy.sql
-- Expression classes and camera devices.
--
-- emotion_classes is the single source of truth for label sets: dataset labels,
-- model descriptors and the UI all resolve names through this table rather
-- than hard-coding them.

CREATE TABLE emotion_classes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          VARCHAR(64)  NOT NULL UNIQUE,
    display_name  VARCHAR(64)  NOT NULL,
    emoji         VARCHAR(16)  NOT NULL DEFAULT '',
    color_rgb     VARCHAR(16)  NOT NULL DEFAULT '',
    polarity      VARCHAR(32)  NOT NULL DEFAULT 'neutral',
    description   VARCHAR(255) NOT NULL DEFAULT '',
    is_active     SMALLINT     NOT NULL DEFAULT 1,
    sort_order    INTEGER      NOT NULL DEFAULT 0,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_emotion_polarity
        CHECK (polarity IN ('positive', 'neutral', 'neutral_positive', 'negative'))
);

CREATE TABLE camera_devices (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    index_number  INTEGER      NOT NULL,
    path          VARCHAR(255)     NULL,
    name          VARCHAR(255) NOT NULL DEFAULT '',
    vendor        VARCHAR(128) NOT NULL DEFAULT '',
    is_webcam     SMALLINT     NOT NULL DEFAULT 1,
    is_active     SMALLINT     NOT NULL DEFAULT 0,
    last_used_at  DATETIME         NULL,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_camera_device UNIQUE (index_number)
);
