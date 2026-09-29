-- 0003_datasets.sql
-- Dataset management.
--
-- Image bytes are never stored in the database. dataset_images holds the
-- filesystem path plus the label sidecar path; the image itself lives on disk
-- and the database only references it.

CREATE TABLE datasets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              VARCHAR(160) NOT NULL,
    version           INTEGER      NOT NULL DEFAULT 1,
    description       TEXT             NULL,
    root_path         VARCHAR(512) NOT NULL,
    licence           VARCHAR(255) NOT NULL DEFAULT '',
    attribution       VARCHAR(255) NOT NULL DEFAULT '',
    status            VARCHAR(32)  NOT NULL DEFAULT 'draft',
    total_images      INTEGER      NOT NULL DEFAULT 0,
    class_count       INTEGER      NOT NULL DEFAULT 0,
    train_count       INTEGER      NOT NULL DEFAULT 0,
    val_count         INTEGER      NOT NULL DEFAULT 0,
    test_count        INTEGER      NOT NULL DEFAULT 0,
    health_status     VARCHAR(32)  NOT NULL DEFAULT 'unchecked',
    created_by        INTEGER          NULL,
    created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT uq_dataset_version UNIQUE (name, version),
    CONSTRAINT ck_dataset_status
        CHECK (status IN ('draft', 'validating', 'ready', 'archived')),
    CONSTRAINT ck_dataset_health
        CHECK (health_status IN ('unchecked', 'ok', 'warning', 'error'))
);

CREATE TABLE dataset_images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER      NOT NULL,
    relative_path   VARCHAR(512) NOT NULL,
    absolute_path   VARCHAR(512) NOT NULL DEFAULT '',
    file_size_bytes INTEGER          NULL,
    width           INTEGER          NULL,
    height          INTEGER          NULL,
    checksum        VARCHAR(64)      NULL,
    split           VARCHAR(16)  NOT NULL DEFAULT 'train',
    is_valid        SMALLINT     NOT NULL DEFAULT 1,
    is_duplicate    SMALLINT     NOT NULL DEFAULT 0,
    prelabel_model  VARCHAR(160)     NULL,
    annotated_by    VARCHAR(32)  NOT NULL DEFAULT 'ai',
    imported_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    CONSTRAINT uq_dataset_image UNIQUE (dataset_id, relative_path),
    CONSTRAINT ck_dataset_split
        CHECK (split IN ('train', 'val', 'test', 'unassigned'))
);

CREATE TABLE dataset_labels (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id      INTEGER      NOT NULL,
    class_id      INTEGER          NULL,
    class_name    VARCHAR(64)  NOT NULL,
    x_center      REAL         NOT NULL DEFAULT 0.5,
    y_center      REAL         NOT NULL DEFAULT 0.5,
    width_norm    REAL         NOT NULL DEFAULT 1.0,
    height_norm   REAL         NOT NULL DEFAULT 1.0,
    confidence    REAL             NULL,
    source        VARCHAR(32)  NOT NULL DEFAULT 'manual',
    updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES dataset_images(id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES emotion_classes(id) ON DELETE SET NULL,
    CONSTRAINT ck_label_source
        CHECK (source IN ('manual', 'ai', 'import'))
);

CREATE INDEX ix_datasets_status ON datasets (status);
CREATE INDEX ix_dataset_images_dataset ON dataset_images (dataset_id, split);
CREATE INDEX ix_dataset_images_checksum ON dataset_images (dataset_id, checksum);
CREATE INDEX ix_dataset_labels_image ON dataset_labels (image_id);
CREATE INDEX ix_dataset_labels_class ON dataset_labels (class_name);
