import logging
import os
import sys
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

from config import cfg
from features import FEATURE_COLS, build_features

MODEL_NAME = "market-anomaly-iso-forest"

# Windows consoles default to cp1252, which can't encode the emoji MLflow
# prints in its "View run" message. Force UTF-8 so runs don't crash on exit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=cfg.log_level)
log = logging.getLogger(__name__)

# Log to the MLflow tracking server (Docker). Override with MLFLOW_TRACKING_URI.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def load_training_data() -> pd.DataFrame:
    engine = create_engine(cfg.db.url)
    df = pd.read_sql(
        "SELECT * FROM price_snapshots ORDER BY ticker, captured_at",
        engine,
    )
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    return df


def build_all_features(raw: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, group in raw.groupby("ticker"):
        if len(group) < 5:
            continue
        feat = build_features(group.copy())
        feat["ticker"] = ticker
        frames.append(feat)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train(contamination: float = 0.05, n_estimators: int = 100):
    mlflow.set_experiment("market-anomaly-detection")

    with mlflow.start_run() as run:
        raw = load_training_data()
        features_df = build_all_features(raw)

        if features_df.empty:
            log.error("No feature data — run producer first")
            return

        X = features_df[FEATURE_COLS].dropna()
        log.info(f"Training on {len(X)} rows")

        mlflow.log_params({
            "contamination": contamination,
            "n_estimators": n_estimators,
            "n_training_rows": len(X),
            "tickers": cfg.tickers,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        })

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)

        scores = model.score_samples(X)
        preds = model.predict(X)
        flag_rate = (preds == -1).sum() / len(preds)

        mlflow.log_metrics({
            "flag_rate": round(float(flag_rate), 4),
            "score_mean": round(float(scores.mean()), 4),
            "score_std": round(float(scores.std()), 4),
        })

        mlflow.sklearn.log_model(
            model,
            name="isolation_forest",
            registered_model_name=MODEL_NAME,
        )

        # Promote this version to "champion" so the detection service serves it.
        # In production this promotion would be gated on evaluate.py metrics.
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}' and run_id='{run.info.run_id}'"
        )
        if versions:
            version = versions[0].version
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            log.info(f"Promoted {MODEL_NAME} v{version} to @champion")

        log.info(f"Run complete: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    train()