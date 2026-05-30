"""Offline evaluation of the hybrid detector on a labeled benchmark.

Most anomaly-detection demos never answer "how do you know it works?". This
script does: it builds a labeled benchmark (seeded random-walk price series with
injected price shocks at known positions), trains the Isolation Forest the same
way the production trainer does, then scores every point through the *real*
``HybridDetector`` and reports precision / recall / F1 against the known labels.

It reports two configurations so the precision/recall trade-off is explicit:
  * z-score only  — the statistical signal alone (high precision on large moves)
  * hybrid        — z-score OR Isolation Forest (broader coverage, more FPs)

Metrics are logged to MLflow (if reachable) so each model carries an evaluation
record alongside its training run.

Run: python evaluate.py
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import cfg
from detector import HybridDetector
from features import FEATURE_COLS, build_features

# UTF-8 for Windows consoles (MLflow prints emoji on run exit).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=cfg.log_level)
log = logging.getLogger(__name__)

RNG = np.random.default_rng(42)
BENCHMARK_TICKERS = ["AAPL", "MSFT", "NVDA", "BTC-USD", "ETH-USD"]


def _synthetic_series(ticker: str, n: int = 400, start: float = 100.0,
                      vol: float = 0.002) -> pd.DataFrame:
    """A seeded random-walk price series shaped like a raw-snapshot frame."""
    asset_type = "crypto" if ticker.endswith("-USD") else "equity"
    rets = RNG.normal(0, vol, n)
    prices = start * np.exp(np.cumsum(rets))
    base = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame({
        "ticker": ticker,
        "asset_type": asset_type,
        "price": prices,
        "volume": 1_000_000.0,
        "prev_close": prices,
        "day_high": prices * 1.01,
        "day_low": prices * 0.99,
        "captured_at": [base + pd.Timedelta(minutes=i) for i in range(n)],
    })


def _inject_anomalies(df: pd.DataFrame, n_anoms: int = 12,
                      shock: float = 0.06) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject sudden price shocks at random positions; return (df, labels)."""
    df = df.copy()
    labels = np.zeros(len(df), dtype=int)
    idx = RNG.choice(np.arange(40, len(df)), size=n_anoms, replace=False)
    price_col = df.columns.get_loc("price")
    for i in idx:
        direction = 1 if RNG.random() > 0.5 else -1
        df.iloc[i, price_col] *= (1 + direction * shock)
        labels[i] = 1
    return df, labels


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def run_evaluation() -> dict:
    """Build the benchmark, train the model, score it, return metrics."""
    # Train the Isolation Forest on clean series (no injected anomalies).
    train_feats = [build_features(_synthetic_series(f"{t}_train"))
                   for t in BENCHMARK_TICKERS]
    X_train = pd.concat(train_feats, ignore_index=True)[FEATURE_COLS].dropna()
    model = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    model.fit(X_train)

    detector = HybridDetector(iso_model=model)

    y_true: list[int] = []
    y_hybrid: list[int] = []
    y_zonly: list[int] = []
    for t in BENCHMARK_TICKERS:
        raw, labels = _inject_anomalies(_synthetic_series(f"{t}_test"))
        feat = build_features(raw)
        row_labels = labels[feat.index.to_numpy()]   # rows that survived feature build
        for pos in range(len(feat)):
            result = detector.score(feat.iloc[[pos]])
            y_true.append(int(row_labels[pos]))
            y_hybrid.append(int(result.is_anomaly))
            y_zonly.append(int(abs(result.z_score) >= detector.z_threshold))

    return {
        "n_samples": len(y_true),
        "n_anomalies": int(sum(y_true)),
        "zscore_only": _metrics(y_true, y_zonly),
        "hybrid": _metrics(y_true, y_hybrid),
    }


def main():
    m = run_evaluation()
    z, h = m["zscore_only"], m["hybrid"]
    print(
        f"\n  Benchmark: {m['n_samples']} samples, {m['n_anomalies']} injected anomalies\n"
        f"  {'config':<14}{'precision':>10}{'recall':>9}{'f1':>8}   confusion (TP/FP/FN/TN)\n"
        f"  {'z-score only':<14}{z['precision']:>10.3f}{z['recall']:>9.3f}{z['f1']:>8.3f}"
        f"   {z['tp']}/{z['fp']}/{z['fn']}/{z['tn']}\n"
        f"  {'hybrid':<14}{h['precision']:>10.3f}{h['recall']:>9.3f}{h['f1']:>8.3f}"
        f"   {h['tp']}/{h['fp']}/{h['fn']}/{h['tn']}\n"
    )

    # Log to MLflow if a tracking server is reachable; never fail the run on it.
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("market-anomaly-detection")
        flat = {"eval_n_samples": m["n_samples"], "eval_n_anomalies": m["n_anomalies"]}
        for cfg_name, d in (("zscore", z), ("hybrid", h)):
            for k, v in d.items():
                flat[f"eval_{cfg_name}_{k}"] = v
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(flat)
        log.info("Logged evaluation metrics to MLflow")
    except Exception as e:
        log.warning(f"Skipped MLflow logging: {e}")

    return m


if __name__ == "__main__":
    main()
