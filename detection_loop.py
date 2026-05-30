# detection_loop.py
# ─────────────────────────────────────────────────────────────
# Reads recent snapshots from PostgreSQL, scores each ticker
# with the hybrid detector, and writes anomalies to
# price_movements and all predictions to predictions.
#
# Run: python detection_loop.py
# ─────────────────────────────────────────────────────────────
import json
import logging
import time

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import cfg
from detector import AnomalyResult, HybridDetector
from features import build_features, get_ticker_history

logging.basicConfig(level=cfg.log_level)
log = logging.getLogger(__name__)

ENGINE  = create_engine(cfg.db.url, pool_size=cfg.db.pool_size)
Session = sessionmaker(bind=ENGINE)

INSERT_PREDICTION = text("""
    INSERT INTO predictions
        (snapshot_id, model_version, is_anomaly, confidence,
         z_score, iso_score, input_features)
    VALUES
        (:snapshot_id, :model_version, :is_anomaly, :confidence,
         :z_score, :iso_score, :input_features)
""")

INSERT_MOVEMENT = text("""
    INSERT INTO price_movements
        (ticker, asset_type, price_before, price_after,
         move_pct, z_score, iso_score, flagged)
    VALUES
        (:ticker, :asset_type, :price_before, :price_after,
         :move_pct, :z_score, :iso_score, :flagged)
""")

GET_LATEST_SNAPSHOT = text("""
    SELECT id FROM price_snapshots
    WHERE ticker = :ticker
    ORDER BY captured_at DESC
    LIMIT 1
""")


def run():
    # One detector instance per ticker — maintains rolling buffer
    detectors: dict[str, HybridDetector] = {
        ticker: HybridDetector() for ticker in cfg.tickers
    }

    log.info(f"Detection loop started for {len(cfg.tickers)} tickers")

    while True:
        for ticker in cfg.tickers:
            session = Session()
            try:
                # Pull recent history for this ticker
                df = get_ticker_history(ticker, window_hours=48)

                if len(df) < 5:
                    log.debug(f"{ticker}: not enough history yet ({len(df)} rows)")
                    continue

                # Build features
                feat_df = build_features(df)
                if feat_df.empty:
                    continue

                # Score with hybrid detector
                result: AnomalyResult = detectors[ticker].score(feat_df)

                # Get latest snapshot ID for foreign key
                snapshot_row = session.execute(
                    GET_LATEST_SNAPSHOT, {"ticker": ticker}
                ).fetchone()
                snapshot_id = snapshot_row[0] if snapshot_row else None

                # Log every prediction
                session.execute(INSERT_PREDICTION, {
                    "snapshot_id":    snapshot_id,
                    "model_version":  "v1.0.0",
                    "is_anomaly":     result.is_anomaly,
                    "confidence":     result.confidence,
                    "z_score":        result.z_score,
                    "iso_score":      result.iso_score,
                    "input_features": json.dumps({
                        "price":           result.price,
                        "price_delta_pct": result.price_delta_pct,
                        "reason":          result.reason,
                    }),
                })

                # If anomaly — log to price_movements and print alert
                if result.is_anomaly:
                    prev_price = float(feat_df.iloc[-2]["price"]) if len(feat_df) >= 2 else result.price
                    session.execute(INSERT_MOVEMENT, {
                        "ticker":      ticker,
                        "asset_type":  result.asset_type,
                        "price_before": prev_price,
                        "price_after":  result.price,
                        "move_pct":     result.price_delta_pct,
                        "z_score":      result.z_score,
                        "iso_score":    result.iso_score,
                        "flagged":      True,
                    })
                    log.warning(
                        f"🚨 ANOMALY | {ticker} | "
                        f"price=${result.price:.2f} | "
                        f"move={result.price_delta_pct:+.2f}% | "
                        f"confidence={result.confidence:.2f} | "
                        f"{result.reason}"
                    )
                else:
                    log.info(f"✓ {ticker} | price=${result.price:.2f} | z={result.z_score:.2f} | normal")

                session.commit()

            except Exception as e:
                session.rollback()
                log.error(f"Error processing {ticker}: {e}")
            finally:
                session.close()

        log.info(f"Cycle complete — sleeping {cfg.poll_interval_seconds}s")
        time.sleep(cfg.poll_interval_seconds)


if __name__ == "__main__":
    run()