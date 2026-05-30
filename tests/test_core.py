"""Unit tests for the detection core.

These are deliberately database-free so they run fast in CI without spinning up
Postgres. They cover feature engineering, the hybrid detector's decision logic,
the PSI drift metric, and ticker classification.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from detector import HybridDetector
from drift_detector import compute_psi
from features import FEATURE_COLS, build_features
from price_producer import classify


def _synthetic(prices: list[float]) -> pd.DataFrame:
    """Build a raw-snapshot-shaped DataFrame from a list of prices."""
    n = len(prices)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        "ticker": ["TEST"] * n,
        "asset_type": ["equity"] * n,
        "price": prices,
        "volume": [1_000_000.0] * n,
        "prev_close": prices,
        "day_high": [p * 1.01 for p in prices],
        "day_low": [p * 0.99 for p in prices],
        "captured_at": [base + timedelta(minutes=i) for i in range(n)],
    })


def test_build_features_produces_all_feature_cols():
    df = build_features(_synthetic([100.0 + i * 0.1 for i in range(30)]))
    for col in FEATURE_COLS:
        assert col in df.columns
    assert not df.empty


def test_classify_distinguishes_crypto_from_equity():
    assert classify("AAPL") == "equity"
    assert classify("SPY") == "equity"
    assert classify("BTC-USD") == "crypto"
    assert classify("ETH-USD") == "crypto"


def test_detector_flags_a_zscore_spike():
    # 29 flat points then a sharp jump -> large z-score on the last row.
    prices = [100.0] * 29 + [130.0]
    feats = build_features(_synthetic(prices))
    result = HybridDetector(z_threshold=2.5).score(feats)
    assert result.is_anomaly is True
    assert abs(result.z_score) >= 2.5
    assert result.confidence > 0


def test_detector_treats_flat_series_as_normal():
    feats = build_features(_synthetic([100.0] * 30))
    result = HybridDetector(z_threshold=2.5).score(feats)
    assert result.is_anomaly is False
    assert result.z_score == 0.0


def test_psi_is_near_zero_for_identical_distributions():
    x = np.random.default_rng(1).normal(0, 1, 1000)
    assert compute_psi(x, x) < 0.01


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(2)
    baseline = rng.normal(0, 1, 1000)
    shifted = rng.normal(3, 1, 1000)
    assert compute_psi(baseline, shifted) > 0.2
