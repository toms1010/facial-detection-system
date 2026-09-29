-- 0004_models.sql
-- Model registry, versions, training runs, metrics and evaluation results.
--
-- A model is a logical family ("Face Detector"); a model_version is one
-- immutable, addressable artefact built from a dataset version. Activating a
-- version is always an explicit act: nothing here replaces a live model by
-- default.

CREATE TABLE models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          VARCHAR(160) NOT NULL UNIQUE,
    task_type     VARCHAR(64)  NOT NULL DEFAULT 'expression',
    description   TEXT             NULL,
    owner_id      INTEGER          NULL,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT ck_model_task
        CHECK (task_type IN ('detection', 'expression', 'embedding', 'other'))
);

CREATE TABLE model_versions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id          INTEGER      NOT NULL,
    version           VARCHAR(32)  NOT NULL,
    weights_path      VARCHAR(512) NOT NULL,
    framework         VARCHAR(32)  NOT NULL DEFAULT 'onnx',
    descriptor_path   VARCHAR(512)     NULL,
    class_set         TEXT             NULL,
    input_size        INTEGER      NOT NULL DEFAULT 224,
    dataset_id        INTEGER          NULL,
    status            VARCHAR(32)  NOT NULL DEFAULT 'registered',
    is_active         SMALLINT     NOT NULL DEFAULT 0,
    file_size_bytes   INTEGER          NULL,
    checksum          VARCHAR(64)      NULL,
    licence           VARCHAR(255) NOT NULL DEFAULT '',
    notes             TEXT             NULL,
    created_by        INTEGER          NULL,
    created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id)   REFERENCES models(id)        ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)      ON DELETE SET NULL,
    FOREIGN KEY (created_by) REFERENCES users(id)         ON DELETE SET NULL,
    CONSTRAINT uq_model_version UNIQUE (model_id, version),
    CONSTRAINT ck_model_status
        CHECK (status IN ('registered', 'validated', 'active', 'retired', 'failed'))
);

CREATE TABLE training_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id          INTEGER      NOT NULL,
    dataset_id        INTEGER      NOT NULL,
    dataset_version   INTEGER      NOT NULL DEFAULT 1,
    run_name          VARCHAR(160) NOT NULL,
    status            VARCHAR(32)  NOT NULL DEFAULT 'queued',
    base_version      VARCHAR(32)      NULL,
    config_json       TEXT             NULL,
    device            VARCHAR(64)  NOT NULL DEFAULT 'cpu',
    epochs            INTEGER      NOT NULL DEFAULT 100,
    batch_size        INTEGER      NOT NULL DEFAULT 16,
    image_size        INTEGER      NOT NULL DEFAULT 640,
    learning_rate     REAL         NOT NULL DEFAULT 0.01,
    optimizer         VARCHAR(64)  NOT NULL DEFAULT 'auto',
    workers           INTEGER      NOT NULL DEFAULT 4,
    checkpoint_path   VARCHAR(512)     NULL,
    early_stopping    SMALLINT     NOT NULL DEFAULT 0,
    started_at        DATETIME         NULL,
    finished_at       DATETIME         NULL,
    error_message     TEXT             NULL,
    launched_by       INTEGER          NULL,
    created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id)    REFERENCES models(id)    ON DELETE CASCADE,
    FOREIGN KEY (dataset_id)  REFERENCES datasets(id)  ON DELETE RESTRICT,
    FOREIGN KEY (launched_by) REFERENCES users(id)    ON DELETE SET NULL,
    CONSTRAINT ck_run_status
        CHECK (status IN ('queued', 'running', 'paused', 'completed', 'failed', 'cancelled'))
);

CREATE TABLE training_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER      NOT NULL,
    epoch         INTEGER      NOT NULL,
    train_loss    REAL             NULL,
    val_loss      REAL             NULL,
    precision     REAL             NULL,
    recall        REAL             NULL,
    f1            REAL             NULL,
    map50         REAL             NULL,
    map50_95      REAL             NULL,
    accuracy      REAL             NULL,
    learning_rate REAL             NULL,
    gpu_percent   REAL             NULL,
    gpu_memory_mb REAL             NULL,
    recorded_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    CONSTRAINT uq_run_epoch UNIQUE (run_id, epoch)
);

CREATE TABLE test_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id  INTEGER      NOT NULL,
    run_name          VARCHAR(160) NOT NULL,
    source_kind       VARCHAR(32)  NOT NULL DEFAULT 'dataset',
    source_path       VARCHAR(512) NOT NULL DEFAULT '',
    image_count       INTEGER      NOT NULL DEFAULT 0,
    status            VARCHAR(32)  NOT NULL DEFAULT 'queued',
    accuracy          REAL             NULL,
    precision         REAL             NULL,
    recall            REAL             NULL,
    f1                REAL             NULL,
    map50             REAL             NULL,
    map50_95          REAL             NULL,
    average_latency_ms REAL            NULL,
    device            VARCHAR(64)  NOT NULL DEFAULT 'cpu',
    error_message     TEXT             NULL,
    started_at        DATETIME         NULL,
    finished_at       DATETIME         NULL,
    created_by        INTEGER          NULL,
    created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by)       REFERENCES users(id)         ON DELETE SET NULL,
    CONSTRAINT ck_test_source
        CHECK (source_kind IN ('image', 'folder', 'video', 'webcam', 'dataset')),
    CONSTRAINT ck_test_status
        CHECK (status IN ('queued', 'running', 'completed', 'failed'))
);

CREATE TABLE test_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id     INTEGER      NOT NULL,
    image_path      VARCHAR(512) NOT NULL DEFAULT '',
    correct         SMALLINT     NOT NULL DEFAULT 0,
    actual_class    VARCHAR(64)      NULL,
    predicted_class VARCHAR(64)      NULL,
    confidence      REAL             NULL,
    latency_ms      REAL             NULL,
    face_count      INTEGER      NOT NULL DEFAULT 0,
    FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE
);

CREATE TABLE confusion_matrix (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id     INTEGER      NOT NULL,
    actual_class    VARCHAR(64)  NOT NULL,
    predicted_class VARCHAR(64)  NOT NULL,
    count           INTEGER      NOT NULL DEFAULT 0,
    FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE,
    CONSTRAINT uq_confusion_cell UNIQUE (test_run_id, actual_class, predicted_class)
);

CREATE INDEX ix_models_task ON models (task_type);
CREATE INDEX ix_model_versions_model ON model_versions (model_id, status);
CREATE INDEX ix_model_versions_active ON model_versions (model_id, is_active);
CREATE INDEX ix_training_runs_status ON training_runs (status, created_at);
CREATE INDEX ix_training_metrics_run ON training_metrics (run_id, epoch);
CREATE INDEX ix_test_runs_version ON test_runs (model_version_id, created_at);
