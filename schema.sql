CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS price_snapshots (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker          TEXT NOT NULL,
    asset_type      TEXT NOT NULL,        -- 'equity' or 'crypto'
    price           NUMERIC(18,6) NOT NULL,
    volume          NUMERIC(24,2),
    prev_close      NUMERIC(18,6),
    day_high        NUMERIC(18,6),
    day_low         NUMERIC(18,6),
    raw_payload     JSONB
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
    ON price_snapshots (ticker, captured_at DESC);

CREATE TABLE IF NOT EXISTS price_movements (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker          TEXT NOT NULL,
    asset_type      TEXT NOT NULL,
    price_before    NUMERIC(18,6) NOT NULL,
    price_after     NUMERIC(18,6) NOT NULL,
    move_pct        NUMERIC(10,6) NOT NULL,
    z_score         NUMERIC(8,4),
    iso_score       NUMERIC(8,6),
    flagged         BOOLEAN NOT NULL DEFAULT TRUE,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot_id     UUID REFERENCES price_snapshots(id),
    model_version   TEXT NOT NULL,
    is_anomaly      BOOLEAN NOT NULL,
    confidence      NUMERIC(5,4) NOT NULL,
    z_score         NUMERIC(8,4),
    iso_score       NUMERIC(8,6),
    input_features  JSONB
);

CREATE INDEX IF NOT EXISTS idx_predictions_time
    ON predictions (predicted_at DESC);

CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    version_tag     TEXT NOT NULL UNIQUE,
    mlflow_run_id   TEXT,
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    f1_score        NUMERIC(5,4),
    notes           TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT FALSE
);