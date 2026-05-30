from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import cfg
from features import FEATURE_COLS

log = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    ticker: str
    asset_type: str
    is_anomaly: bool
    confidence: float
    z_score: float
    iso_score: float
    price: float
    price_delta_pct: float
    reason: str


class HybridDetector:
    def __init__(
        self,
        z_threshold: float = None,
        iso_contamination: float = 0.05,
        min_samples_for_iso: int = 30,
    ):
        self.z_threshold = z_threshold or cfg.z_score_threshold
        self.iso_contamination = iso_contamination
        self.min_samples_for_iso = min_samples_for_iso
        self._iso_model: IsolationForest | None = None
        self._fit_data: list[list[float]] = []

    def _fit_isolation_forest(self, X: np.ndarray):
        self._iso_model = IsolationForest(
            contamination=self.iso_contamination,
            n_estimators=100,
            random_state=42,
        )
        self._iso_model.fit(X)

    def score(self, df: pd.DataFrame) -> AnomalyResult:
        if df.empty:
            raise ValueError("Empty DataFrame")

        latest = df.iloc[-1]
        z = float(latest.get("z_score", 0))
        z_flag = abs(z) >= self.z_threshold

        iso_score = 0.0
        iso_flag = False
        feature_row = [latest.get(c, 0.0) for c in FEATURE_COLS]
        self._fit_data.append(feature_row)

        if len(self._fit_data) >= self.min_samples_for_iso:
            X = np.array(self._fit_data)
            if len(self._fit_data) % 50 == 0 or self._iso_model is None:
                self._fit_isolation_forest(X)
            if self._iso_model is not None:
                iso_score = float(self._iso_model.score_samples([feature_row])[0])
                iso_flag = iso_score < -0.1

        is_anomaly = z_flag or iso_flag
        z_conf = min(abs(z) / (self.z_threshold * 2), 1.0)
        iso_conf = max(0, min((-iso_score) / 0.3, 1.0)) if iso_score != 0 else 0.0
        confidence = round((z_conf * 0.6 + iso_conf * 0.4) if is_anomaly else 0.0, 4)

        reasons = []
        if z_flag:
            reasons.append(f"z-score={z:.2f}")
        if iso_flag:
            reasons.append(f"iso-score={iso_score:.4f}")
        reason = " | ".join(reasons) if reasons else "no anomaly"

        return AnomalyResult(
            ticker=str(latest.get("ticker", "")),
            asset_type=str(latest.get("asset_type", "")),
            is_anomaly=is_anomaly,
            confidence=confidence,
            z_score=z,
            iso_score=iso_score,
            price=float(latest.get("price", 0)),
            price_delta_pct=float(latest.get("price_delta_pct", 0)),
            reason=reason,
        )