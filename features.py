from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from config import cfg

ENGINE = create_engine(cfg.db.url)


def get_ticker_history(ticker: str, window_hours: int = 48) -> pd.DataFrame:
    query = text("""
        SELECT ticker, asset_type, price, volume,
               prev_close, day_high, day_low, captured_at
        FROM price_snapshots
        WHERE ticker = :ticker
          AND captured_at >= NOW() - INTERVAL '48 hours'
        ORDER BY captured_at ASC
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    return df


def build_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy().sort_values("captured_at")

    df["price_delta"]     = df["price"].diff()
    df["price_delta_pct"] = df["price"].pct_change() * 100
    df["price_delta_abs"] = df["price_delta"].abs()

    df["rolling_mean"] = df["price"].rolling(window, min_periods=3).mean()
    df["rolling_std"]  = df["price"].rolling(window, min_periods=3).std()

    # Baseline excludes the current point so an anomalous price doesn't dampen
    # its own z-score (avoids look-in leakage).
    baseline_mean = df["rolling_mean"].shift(1)
    baseline_std  = df["rolling_std"].shift(1)
    df["z_score"] = (df["price"] - baseline_mean) / baseline_std.replace(0, 1e-9)

    df["high_low_range"] = df["day_high"] - df["day_low"]

    # Set index for time-based rolling, then reset
    df = df.set_index("captured_at")
    df["move_count_1h"] = (
        df["price_delta_abs"]
        .rolling("1h")
        .apply(lambda x: (x > 0).sum(), raw=True)
    )
    df = df.reset_index()

    df = df.dropna(subset=["z_score"])
    return df


FEATURE_COLS = [
    "price_delta_abs",
    "price_delta_pct",
    "rolling_mean",
    "rolling_std",
    "z_score",
    "high_low_range",
    "move_count_1h",
]