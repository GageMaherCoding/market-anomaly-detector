# Market Anomaly Detector

Real-time anomaly detection on live market data. The system streams prices for
25 equities and crypto assets, scores every movement with a hybrid
statistical + ML model, and serves the results through a REST API and live
Grafana dashboards — all continuously, from one `docker compose up`.

> Built as a portfolio project to demonstrate an end-to-end **production ML
> system**: streaming ingestion, feature engineering, a served + monitored
> model, experiment tracking, data-quality testing, drift detection, and CI.
> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Stack

Python · Kafka · PostgreSQL · scikit-learn (Isolation Forest) · FastAPI ·
dbt · MLflow · Grafana · Docker Compose · GitHub Actions

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
- **Live Price Feed** — http://localhost:3000/d/price-feed
- **Anomaly Monitor** — http://localhost:3000/d/anomaly-monitor

Check data is flowing:

```bash
docker compose logs producer --tail 10      # live prices being fetched
docker compose logs detector --tail 10      # per-ticker scoring
curl http://localhost:8080/health
```

## Services

| Service | What it does |
|---|---|
| `producer` | Fetches prices from yfinance → Kafka |
| `consumer` | Kafka → `price_snapshots` in Postgres |
| `detector` | Scores each ticker every 60s → `predictions` / `price_movements` |
| `drift` | Hourly PSI drift check over the feature distribution |
| `api` | FastAPI read API |
| `mlflow` | Tracking server + model registry |
| `grafana` | Dashboards (provisioned from `grafana/`) |
| `postgres` / `kafka` / `zookeeper` | Infrastructure |

## Local development (without Docker for the Python parts)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Python tools read connection settings from `.env` (see `.env.example`) and
default to `localhost`, so they work against the Dockerized Postgres/Kafka/MLflow.

```bash
python train.py            # train + log a model to MLflow
python drift_detector.py   # one-off drift check
```

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
├── train.py / drift_detector.py             # MLOps: training, drift
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

- **Containers show `(Paused)`** after a Docker Desktop restart →
  `docker compose unpause`.
- **Kafka "dependency failed to start / unhealthy"** on a bulk restart → it just
  needs a few more seconds; re-run `docker compose up -d`.
- **`train.py` UnicodeEncodeError on Windows** → already handled (the script
  forces UTF-8 stdout); if running other scripts, set `PYTHONUTF8=1`.

## Deployment

The API is container-ready for a free host (Render / Cloud Run). Point its
`DATABASE_URL` at a managed Postgres and deploy the image built from
`Dockerfile`. This gives a shareable public `/anomalies` URL. (Not yet wired —
see ARCHITECTURE.md §13.)

## Disclaimer

This detects *unusual* price movements, not their direction. It is a systems /
ML engineering demonstration, **not** trading advice or a trading signal.
