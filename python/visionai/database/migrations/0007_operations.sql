-- 0007_operations.sql
-- Operational logging: application events, hardware samples and system metrics.
--
-- application_logs is a structured event log. Message bodies are scrubbed of
-- machine identifiers before they reach this table; see
-- visionai.hardware.privacy.scrub_text.

CREATE TABLE application_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    level         VARCHAR(16)  NOT NULL DEFAULT 'INFO',
    category      VARCHAR(32)  NOT NULL DEFAULT 'app',
    message       TEXT         NOT NULL,
    module        VARCHAR(160) NOT NULL DEFAULT '',
    line_number   INTEGER          NULL,
    context_json  TEXT             NULL,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_log_level
        CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'))
);

CREATE TABLE system_metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    cpu_percent         REAL             NULL,
    cpu_temperature     REAL             NULL,
    ram_percent         REAL             NULL,
    ram_used_bytes      INTEGER          NULL,
    ram_total_bytes     INTEGER          NULL,
    gpu_percent         REAL             NULL,
    gpu_temperature     REAL             NULL,
    gpu_memory_used     INTEGER          NULL,
    gpu_memory_total    INTEGER          NULL,
    disk_percent        REAL             NULL,
    net_rx_bytes_per_second REAL         NULL,
    net_tx_bytes_per_second REAL         NULL,
    inference_latency_ms REAL            NULL,
    inference_fps       REAL             NULL,
    face_count          INTEGER          NOT NULL DEFAULT 0,
    backend             VARCHAR(32)  NOT NULL DEFAULT ''
);

CREATE TABLE hardware_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    category        VARCHAR(64)  NOT NULL,
    severity        VARCHAR(16)  NOT NULL DEFAULT 'info',
    message         TEXT         NOT NULL,
    sensor_name     VARCHAR(160) NOT NULL DEFAULT '',
    sensor_value    REAL             NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_app_logs_created ON application_logs (created_at);
CREATE INDEX ix_app_logs_category ON application_logs (category, level);
CREATE INDEX ix_system_metrics_time ON system_metrics (recorded_at);
CREATE INDEX ix_hardware_logs_category ON hardware_logs (category, created_at);
