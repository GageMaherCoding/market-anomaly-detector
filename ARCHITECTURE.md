# Architecture — Market Anomaly Detector

## 1. What it is

A real-time data pipeline that ingests live market prices for 25 equities and
crypto assets, scores every price movement for statistical anomalies with a
hybrid ML model, and serves the results through a REST API and live dashboards.
The whole system runs continuously from a single `docker compose up`.

The point of the project is not to predict the market — it is to demonstrate an
end-to-end **production ML system**: streaming ingestion, feature engineering, a
served + monitored model, experiment tracking, data-quality testing, drift
detection, observability, and CI.

## 2. High-level diagram

```
                  ┌───────────────┐   prices.raw    ┌──────────────┐
   yfinance ────▶ │ price_producer │ ──────────────▶│    Kafka     │
   (25 tickers)   └───────────────┘                 └──────┬───────┘
                                                           │
                                                    ┌──────▼────────┐
                                                    │ price_consumer │
                                                    └──────┬────────┘
                                                           │ INSERT
                                          ┌────────────────▼─────────────────┐
                           reads 48h hist │           PostgreSQL              │
                        ┌─────────────────┤  price_snapshots                  │
                        │                 │  predictions                      │
                ┌───────▼────────┐ writes │  price_movements                  │
                │ detection_loop  │───────▶  model_versions                   │
                │ (HybridDetector)│        └───┬───────────┬──────────────┬───┘
                └────────────────┘            │           │              │
                                         dbt  │     FastAPI│       Grafana│
                                     ┌────────▼───┐ ┌──────▼─────┐ ┌──────▼──────┐
                                     │ staging →  │ │ /health    │ │ price-feed  │
                                     │ intermediate│ │ /anomalies │ │ anomaly-    │
                                     │ → mart     │ │ /predictions│ │  monitor    │
                                     └────────────┘ └────────────┘ └─────────────┘

   train.py ──────────▶ MLflow  (experiment tracking + model registry)
   drift_detector.py ─▶ PSI drift checks  ──▶ email alert on critical drift
```

## 3. Components

| Component | File | Role |
|---|---|---|
| Producer | `price_producer.py` | Polls yfinance every 60s for 25 tickers, publishes `PriceEvent`s to the Kafka topic `prices.raw` (keyed by ticker, gzip, `acks=all`). |
| Consumer | `price_consumer.py` | Consumes `prices.raw` (manual offset commit) and persists each event into `price_snapshots`. |
| Detector loop | `detection_loop.py` | Every 60s, pulls 48h of history per ticker, builds features, scores with a per-ticker `HybridDetector`, writes every prediction to `predictions` and flagged ones to `price_movements`. |
| Detector model | `detector.py` | The `HybridDetector` — z-score + Isolation Forest (see §6). |
| Features | `features.py` | Rolling-window feature engineering (z-score, deltas, ranges) shared by the loop, trainer, and drift checker. |
| Trainer | `train.py` | Batch-trains an Isolation Forest on all collected features and logs params/metrics/model to MLflow, registering `market-anomaly-iso-forest`. |
| Drift checker | `drift_detector.py` | Computes Population Stability Index (PSI) per feature between a baseline and recent window; alerts on critical drift. Runs hourly as the `drift` service. |
| API | `api.py` | FastAPI service exposing `/health`, `/anomalies`, `/predictions`. |
| dbt project | `market_anomaly/` | Curated analytics layer: `staging → intermediate → mart` with 29 data-quality tests and lineage docs. |
| Dashboards | `grafana/` | Datasource + two dashboards, provisioned as code. |

## 4. Data flow

1. **Ingest** — `price_producer` fetches `last_price`, volume, prev close, day
   high/low per ticker and emits a JSON event to Kafka.
2. **Persist** — `price_consumer` writes each event as a row in
   `price_snapshots` (the immutable raw record).
3. **Score** — `detection_loop` reads recent history, engineers features, and
   asks the `HybridDetector` whether the latest point is anomalous. Every score
   is logged to `predictions`; anomalies also land in `price_movements`.
4. **Curate** — dbt transforms raw snapshots into a tested analytical mart.
5. **Serve & observe** — the API exposes anomalies/predictions; Grafana
   visualizes the live feed and anomaly history straight from Postgres.
6. **Operate** — `train.py` versions models in MLflow; `drift_detector`
   watches the input distribution and alerts when it shifts.

## 5. Data model (`schema.sql`)

- **`price_snapshots`** — immutable raw captures: `ticker`, `asset_type`,
  `price`, `volume`, `prev_close`, `day_high`, `day_low`, `captured_at`,
  `raw_payload` (JSONB). Indexed on `(ticker, captured_at DESC)`.
- **`predictions`** — one row per scored snapshot: `is_anomaly`, `confidence`,
  `z_score`, `iso_score`, `model_version`, `input_features` (JSONB), FK to the
  snapshot. Indexed on `predicted_at DESC`.
- **`price_movements`** — flagged anomalies only: `price_before/after`,
  `move_pct`, scores, `flagged`.
- **`model_versions`** — registry bookkeeping: `version_tag`, `mlflow_run_id`,
  metrics, `is_active`.

## 6. The detection model

A **two-signal hybrid**, chosen so the system catches both kinds of anomaly:

- **Z-score (statistical)** — flags when the latest price is ≥ `2.5` standard
  deviations from its 20-period rolling mean. Catches sharp, obvious spikes
  immediately, even with little history.
- **Isolation Forest (unsupervised ML)** — trained on the 7-dimensional feature
  vector (deltas, rolling stats, range, intra-hour movement count). Catches
  *multivariate* anomalies a single z-score misses. It warms up after 30 samples
  per ticker and refits every 50.

An observation is anomalous if **either** signal fires. Confidence blends the
two (`0.6 × z-confidence + 0.4 × iso-confidence`) so a point flagged by both
ranks above one flagged by either alone. Each ticker gets its own detector
instance, so AAPL's volatility profile never contaminates DOGE's.

## 7. MLOps

- **Experiment tracking & registry** — `train.py` logs every run (params,
  `flag_rate`, score statistics, the serialized model) to **MLflow**, and
  registers versioned models under `market-anomaly-iso-forest`. MLflow uses a
  SQLite metadata store and proxied artifact serving, so its state is fully
  isolated from the application database.
- **Drift detection** — `drift_detector.py` computes **PSI** per feature between
  a baseline window and the recent window. PSI ≥ 0.1 warns, ≥ 0.2 is critical
  and triggers an email alert (when SMTP is configured) recommending a retrain.

## 8. Analytics layer (dbt)

`market_anomaly/` models the raw tables into a clean, tested mart:

```
sources (public.*) → stg_price_snapshots → int_price_features → mart_anomaly_signals
        (view)                (view)              (table)
```

- **staging** — type-casts and normalizes raw snapshots, derives `asset_class`.
- **intermediate** — rolling features (z-score, deltas, moving averages) in SQL,
  mirroring the Python feature logic.
- **mart** — joins features with prediction history into the analytical table.

29 dbt tests (`not_null`, `unique`, `accepted_values`) enforce data quality, and
`dbt docs` generates a browsable lineage graph.

## 9. Infrastructure & CI

- **Docker Compose** orchestrates 10 services (Postgres, Zookeeper, Kafka,
  producer, consumer, detector, drift, API, MLflow, Grafana). All carry
  `restart: unless-stopped` so the stack is genuinely always-on.
- **Grafana provisioning** — the datasource and both dashboards are committed as
  YAML/JSON and load automatically; nothing is configured by hand.
- **CI** (`.github/workflows/ci.yml`) runs on every push/PR:
  - *Lint & unit tests* — `ruff` + `pytest` (DB-free core tests).
  - *dbt build* — spins up Postgres, applies the schema, seeds sample data, and
    runs the dbt models **and** their tests.

## 10. Design decisions & trade-offs

- **Kafka for a single-node pipeline** is overkill for 25 tickers, but it makes
  the ingestion/scoring boundary explicit and the design horizontally scalable —
  the realistic shape of a production system.
- **Hybrid over a single model** trades a little complexity for robustness: the
  z-score gives an interpretable, instant signal; the forest covers the
  multivariate cases.
- **Dashboards read raw tables, not the dbt mart.** Raw tables update every 60s,
  so the dashboards are truly live. The mart is the *curated* layer and refreshes
  on a `dbt run`; pointing BI at it is a deliberate next step paired with
  scheduled refreshes.
- **MLflow on SQLite, not Postgres.** The official image lacks `psycopg2`; SQLite
  also keeps MLflow's internal schema out of the application database, which is
  cleaner separation of concerns.

## 11. Known limitations (honest)

- **Not a trading signal.** It detects that a move is unusual, not its direction.
  yfinance is delayed and polled at 60s — there is no tradeable edge here.
- **Drift baseline is short.** Production would use a ~30-day baseline; it is
  shortened here so the check is meaningful with a couple of weeks of data.
- **In-memory detector state.** Each detector's Isolation Forest buffer resets on
  restart and re-warms from DB history; fine for this scale, not for HA.
- **Single broker / single node.** No replication or partitioning tuning.

## 12. How this would scale at FAANG scale

- **Ingestion** — replace the polling producer with a push/websocket feed;
  partition Kafka topics by ticker; run many consumer instances in the group.
- **Feature store** — move feature computation to a shared online/offline store
  (e.g. Feast) so training and serving share definitions and avoid skew.
- **Serving** — split scoring from the loop into a stateless, autoscaled service
  reading features from the store; promote models from the MLflow registry
  behind a champion/challenger setup.
- **Storage** — partition/shard the snapshot table by time, or move to a
  columnar/time-series store; the dbt mart becomes incremental.
- **Ops** — Kubernetes with health/readiness probes, the drift job on a real
  scheduler (Airflow/cron), and alerts wired to PagerDuty rather than email.

## 13. Future work

- Schedule `dbt run` (CI cron) and repoint Grafana panels at the mart.
- Add backtesting/labeling to measure precision/recall instead of just flag rate.
- Deploy the API publicly (Render/Cloud Run) for a shareable live URL.
