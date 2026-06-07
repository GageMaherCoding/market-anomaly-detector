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
        iso_contamination: float = 0.03,
        min_samples_for_iso: int = 30,
        huge_move_mult: float = 1.6,
        vol_z_gate: float = 3.0,
        iso_model: IsolationForest | None = None,
    ):
        self.z_threshold = z_threshold or cfg.z_score_threshold
        self.iso_contamination = iso_contamination
        self.min_samples_for_iso = min_samples_for_iso
        # A price move this many z-thresholds large is flagged outright, with no
        # need for the Isolation Forest to confirm it.
        self.huge_move_mult = huge_move_mult
        # The Isolation Forest may raise an alarm on its own only when it also
        # sees a volume/liquidity dislocation this large (|volume_z|). That scopes
        # its solo authority to the contextual anomalies the price z-score cannot
        # see, instead of letting it flag every price wobble.
        self.vol_z_gate = vol_z_gate
        # A model passed in (e.g. loaded from the MLflow registry) is served
        # directly. Otherwise the detector adaptively self-fits as a cold-start
        # fallback until a registered model becomes available.
        self._iso_model: IsolationForest | None = iso_model
        self._external_model: bool = iso_model is not None
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

        if self._external_model:
            # Serve the registered model directly — no online refitting.
            # Pass a named frame so feature names match how it was trained.
            X_row = pd.DataFrame([feature_row], columns=FEATURE_COLS)
            iso_score = float(self._iso_model.score_samples(X_row)[0])
            iso_flag = bool(self._iso_model.predict(X_row)[0] == -1)
        else:
            # Cold-start fallback: adaptively fit a per-ticker model from the
            # rolling buffer until a registered model is available.
            self._fit_data.append(feature_row)
            if len(self._fit_data) >= self.min_samples_for_iso:
                X = np.array(self._fit_data)
                if len(self._fit_data) % 50 == 0 or self._iso_model is None:
                    self._fit_isolation_forest(X)
                if self._iso_model is not None:
                    iso_score = float(self._iso_model.score_samples([feature_row])[0])
                    iso_flag = bool(self._iso_model.predict([feature_row])[0] == -1)

        vol_z = float(latest.get("volume_z", 0.0))
        iso_active = self._iso_model is not None

        z_conf = min(abs(z) / (self.z_threshold * 2), 1.0)
        # score_samples sits roughly in [-0.7, -0.5]; map "how far below ~-0.5"
        # into a 0..1 severity used only for the reported confidence.
        iso_sev = max(0.0, min((-iso_score - 0.5) / 0.2, 1.0)) if iso_score != 0 else 0.0

        # Fusion — the two detectors play distinct, complementary roles:
        #   * huge_move      : an outsized price move is an anomaly outright.
        #   * confirmed_price: an ordinary z-score trip is trusted only when the
        #                      Isolation Forest agrees, which discards the
        #                      z-score's standalone false positives (the precision
        #                      win). With no IF yet (cold start) z stands alone.
        #   * contextual     : the IF escalates on its own only alongside a real
        #                      volume/liquidity dislocation (|volume_z| >= gate) —
        #                      the anomalies the price z-score cannot see.
        huge_move = abs(z) >= self.z_threshold * self.huge_move_mult
        confirmed_price = z_flag and (iso_flag or not iso_active)
        contextual = iso_flag and abs(vol_z) >= self.vol_z_gate
        is_anomaly = huge_move or confirmed_price or contextual
        confidence = round((z_conf * 0.6 + iso_sev * 0.4) if is_anomaly else 0.0, 4)

        reasons = []
        if huge_move or confirmed_price:
            reasons.append(f"z-score={z:.2f}")
        if contextual:
            reasons.append(f"iso={iso_score:.3f} vol_z={vol_z:.2f}")
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