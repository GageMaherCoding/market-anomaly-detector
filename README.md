# Market Anomaly Detector

Real-time anomaly detection on live market data. The system streams prices for
25 equities and crypto assets, scores every movement with a hybrid
statistical and ML model, and serves the results through a REST API and live
Grafana dashboards. It all runs continuously from one `docker compose up`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design: data flow, the detection
model, MLOps, and trade-offs.

## Stack

Python · Kafka · PostgreSQL · scikit-learn (Isolation Forest) · FastAPI ·
dbt · MLflow · Grafana · Docker Compose

## Results

Offline evaluation on a labeled benchmark, run with `python evaluate.py`. It
builds seeded price series and injects two kinds of anomaly: **point shocks**
(sudden price jumps) and **contextual anomalies** (volume/liquidity surges that
leave the price level untouched). Every point is scored through the real detector:

| configuration | precision | recall | F1 |
|---|---|---|---|
| z-score only | 0.42 | 0.55 | 0.48 |
| hybrid (z-score + Isolation Forest) | **0.83** | **0.90** | **0.86** |

| recall by anomaly kind | point shock | contextual |
|---|---|---|
| z-score only | 1.00 | 0.02 |
| hybrid | 1.00 | 0.78 |

The z-score alone catches every point shock but is blind to contextual anomalies,
and it fires on ordinary drift often enough to be imprecise. The hybrid wins on
both fronts. Requiring the Isolation Forest to confirm a z-score trip discards the
z-score's standalone false positives (precision 0.42 → 0.83), and a volume-gated
solo signal lets the forest catch the contextual anomalies the z-score cannot see
(contextual recall 0.02 → 0.78). Metrics log to MLflow, and CI gates the hybrid's
F1 against the baseline so the result cannot silently regress.

## Quickstart

Requires Docker Desktop.

```bash
docker compose up -d
```

First run pulls images and takes a few minutes. Then everything is live:

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8080/health | also `/anomalies`, `/predictions` |
| Grafana | http://localhost:3000 | login `admin` / `admin` |
| MLflow | http://localhost:5000 | experiments + model registry |
| Postgres | `localhost:5432` | db `anomalydb`, user/pass `postgres`/`password` |

Pre-provisioned Grafana dashboards:
- **Live Price Feed** at http://localhost:3000/d/price-feed
- **Anomaly Monitor** at http://localhost:3000/d/anomaly-monitor

Check data is flowing:

```bash
docker compose logs producer --tail 10      # live prices being fetched
docker compose logs detector --tail 10      # per-ticker scoring
curl http://localhost:8080/health
```

## Local development (without Docker for the Python parts)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Python tools read connection settings from `.env` (see `.env.example`) and
default to `localhost`, so they work against the Dockerized Postgres/Kafka/MLflow.

```bash
python train.py            # train + register a model (promoted to @champion)
python evaluate.py         # score on a labeled benchmark, log metrics to MLflow
python drift_detector.py   # one-off drift check
```

On a fresh `docker compose up` no champion exists yet, so the detector serves an
adaptive per-ticker Isolation Forest as a fallback. Once enough snapshots have
accumulated, run `python train.py` to register and promote a champion; the
detection loop picks it up live (it re-checks the registry every
`CHAMPION_RELOAD_CYCLES` cycles, default 60) without a restart.

## dbt (analytics layer)

```bash
cd market_anomaly
dbt build                  # run models + data-quality tests
dbt docs generate
dbt docs serve --port 18080   # lineage graph (8080 is taken by the API)
```

Connection profile lives at `~/.dbt/profiles.yml` (host `127.0.0.1`, db
`anomalydb`, schema `dbt_dev`).

## Tests & lint

```bash
ruff check .
pytest
```

CI (`.github/workflows/ci.yml`) runs both on every push, plus a full `dbt build`
against a throwaway Postgres.

## Project structure

```
.
├── price_producer.py / price_consumer.py   # ingestion (Kafka)
├── detection_loop.py / detector.py          # scoring + hybrid model
├── features.py                              # shared feature engineering
├── train.py / evaluate.py / drift_detector.py  # MLOps: train, evaluate, drift
├── api.py                                   # FastAPI service
├── config.py / schema.sql                   # config + DB schema
├── market_anomaly/                          # dbt project (staging→int→mart)
├── grafana/                                 # provisioned datasource + dashboards
├── ci/seed.sql                              # sample data for CI
├── tests/                                   # pytest unit tests
├── docker-compose.yml / Dockerfile
└── ARCHITECTURE.md
```

## Troubleshooting

- **Containers show `(Paused)`** after a Docker Desktop restart: run
  `docker compose unpause`.
- **Kafka "dependency failed to start / unhealthy"** on a bulk restart: it just
  needs a few more seconds, so re-run `docker compose up -d`.
- **`train.py` UnicodeEncodeError on Windows**: already handled (the script
  forces UTF-8 stdout); if running other scripts, set `PYTHONUTF8=1`.

## Deployment

The API is container-ready for a free host (Render / Cloud Run). Point its
`DATABASE_URL` at a managed Postgres and deploy the image built from
`Dockerfile`. This gives a shareable public `/anomalies` URL. (Not yet wired;
see ARCHITECTURE.md §13.)

## Disclaimer

This flags unusual price movements, not their direction. It is **not** trading
advice or a signal.
