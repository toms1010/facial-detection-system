-- 0006_sessions.sql
-- Detection sessions and per-face results.
--
-- Privacy note: this records *numbers*, not people. No face image, embedding or
-- biometric template is ever written here. Storing any of those would require a
-- separate consent flow and a different retention policy.

CREATE TABLE detection_sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name      VARCHAR(160) NOT NULL DEFAULT '',
    camera_id         INTEGER          NULL,
    camera_label      VARCHAR(255) NOT NULL DEFAULT '',
    source_kind       VARCHAR(32)  NOT NULL DEFAULT 'webcam',
    detector_name     VARCHAR(160) NOT NULL DEFAULT '',
    classifier_name   VARCHAR(160) NOT NULL DEFAULT '',
    model_version_id  INTEGER          NULL,
    is_trained_model  SMALLINT     NOT NULL DEFAULT 0,
    started_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at       DATETIME         NULL,
    frame_count       INTEGER      NOT NULL DEFAULT 0,
    max_face_count    INTEGER      NOT NULL DEFAULT 0,
    average_fps       REAL             NULL,
    average_latency_ms REAL           NULL,
    user_id           INTEGER          NULL,
    FOREIGN KEY (camera_id)        REFERENCES camera_devices(id)  ON DELETE SET NULL,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id)          REFERENCES users(id)           ON DELETE SET NULL,
    CONSTRAINT ck_session_source
        CHECK (source_kind IN ('webcam', 'video', 'synthetic', 'image'))
);

CREATE TABLE face_detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          INTEGER      NOT NULL,
    frame_number        INTEGER      NOT NULL,
    tracking_id         INTEGER      NOT NULL,
    x_center            REAL         NOT NULL,
    y_center            REAL         NOT NULL,
    width               REAL         NOT NULL,
    height              REAL         NOT NULL,
    detection_score     REAL         NOT NULL DEFAULT 0,
    first_seen_frame    INTEGER      NOT NULL DEFAULT 0,
    last_seen_frame     INTEGER      NOT NULL DEFAULT 0,
    hit_count           INTEGER      NOT NULL DEFAULT 1,
    recorded_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES detection_sessions(id) ON DELETE CASCADE,
    CONSTRAINT uq_session_track_frame UNIQUE (session_id, frame_number, tracking_id)
);

CREATE TABLE expression_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id        INTEGER      NOT NULL,
    expression          VARCHAR(64)  NOT NULL,
    confidence          REAL         NOT NULL DEFAULT 0,
    is_confident        SMALLINT     NOT NULL DEFAULT 0,
    is_heuristic        SMALLINT     NOT NULL DEFAULT 0,
    is_trained_model    SMALLINT     NOT NULL DEFAULT 0,
    alternative         VARCHAR(64)      NULL,
    recorded_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (detection_id) REFERENCES face_detections(id) ON DELETE CASCADE
);

CREATE INDEX ix_sessions_started ON detection_sessions (started_at);
CREATE INDEX ix_face_detections_session ON face_detections (session_id, frame_number);
CREATE INDEX ix_expression_results_detection ON expression_results (detection_id);
CREATE INDEX ix_expression_results_expression ON expression_results (expression);
