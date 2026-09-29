-- 0005_experiments.sql
-- Experiment tracking: which dataset version, which model version, which
-- hardware, and what the numbers came out as.

CREATE TABLE experiments (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               VARCHAR(160) NOT NULL,
    description        TEXT             NULL,
    dataset_id         INTEGER          NULL,
    dataset_version    INTEGER      NOT NULL DEFAULT 1,
    model_version_id   INTEGER          NULL,
    training_run_id    INTEGER          NULL,
    test_run_id        INTEGER          NULL,
    config_json        TEXT             NULL,
    hardware_json      TEXT             NULL,
    status             VARCHAR(32)  NOT NULL DEFAULT 'draft',
    notes              TEXT             NULL,
    created_by         INTEGER          NULL,
    created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id)       REFERENCES datasets(id)       ON DELETE SET NULL,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE SET NULL,
    FOREIGN KEY (training_run_id)  REFERENCES training_runs(id)  ON DELETE SET NULL,
    FOREIGN KEY (test_run_id)      REFERENCES test_runs(id)      ON DELETE SET NULL,
    FOREIGN KEY (created_by)       REFERENCES users(id)          ON DELETE SET NULL,
    CONSTRAINT ck_experiment_status
        CHECK (status IN ('draft', 'running', 'complete', 'abandoned'))
);

CREATE TABLE experiment_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER      NOT NULL,
    metric_name     VARCHAR(64)  NOT NULL,
    metric_value    REAL             NULL,
    unit            VARCHAR(32)  NOT NULL DEFAULT '',
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
    CONSTRAINT uq_experiment_metric UNIQUE (experiment_id, metric_name)
);

CREATE INDEX ix_experiments_status ON experiments (status, created_at);
CREATE INDEX ix_experiments_created_by ON experiments (created_by);
