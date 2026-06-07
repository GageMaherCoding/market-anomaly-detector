import logging
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from config import cfg
from features import FEATURE_COLS, build_features

logging.basicConfig(level=cfg.log_level)
log = logging.getLogger(__name__)

PSI_WARNING   = 0.1
PSI_CRITICAL  = 0.2
N_BINS        = 10


def compute_psi(expected: np.ndarray, actual: np.ndarray) -> float:
    breakpoints = np.unique(np.nanpercentile(expected, np.linspace(0, 100, N_BINS + 1)))
    exp_pct = np.clip(np.histogram(expected, bins=breakpoints)[0] / len(expected), 1e-6, None)
    act_pct = np.clip(np.histogram(actual,   bins=breakpoints)[0] / len(actual),   1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def send_alert(subject: str, body: str):
    user  = os.getenv("SMTP_USER", "")
    passw = os.getenv("SMTP_PASS", "")
    to    = os.getenv("ALERT_EMAIL", "")
    if not all([user, passw, to]):
        log.warning("SMTP not configured — skipping alert")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = user
    msg["To"]      = to
    with smtplib.SMTP_SSL(os.getenv("SMTP_HOST", "smtp.gmail.com"), 465) as s:
        s.login(user, passw)
        s.sendmail(user, [to], msg.as_string())
    log.info(f"Alert sent: {subject}")


def run_drift_check():
    engine  = create_engine(cfg.db.url)
    # Reference = the established baseline (snapshots older than REF_AGE days).
    # Current   = the recent window we test against that baseline.
    # Production would use a ~30-day baseline; kept short here so the check is
    # meaningful with only a couple of weeks of collected data. Override via env.
    ref_age_days = int(os.getenv("DRIFT_REF_AGE_DAYS", "2"))
    curr_hours   = int(os.getenv("DRIFT_CURR_HOURS", "24"))
    cutoff_ref  = datetime.now(timezone.utc) - timedelta(days=ref_age_days)
    cutoff_curr = datetime.now(timezone.utc) - timedelta(hours=curr_hours)

    ref = pd.read_sql(
        text("SELECT * FROM price_snapshots WHERE captured_at < :cutoff"),
        engine, params={"cutoff": cutoff_ref},
    )
    curr = pd.read_sql(
        text("SELECT * FROM price_snapshots WHERE captured_at >= :cutoff"),
        engine, params={"cutoff": cutoff_curr},
    )

    if ref.empty or curr.empty:
        log.warning("Not enough data for drift check")
        return

    ref["captured_at"]  = pd.to_datetime(ref["captured_at"],  utc=True)
    curr["captured_at"] = pd.to_datetime(curr["captured_at"], utc=True)

    ref_feat  = build_features(ref)
    curr_feat = build_features(curr)

    results = {}
    alerts  = []

    for col in FEATURE_COLS:
        if col not in ref_feat or col not in curr_feat:
            continue
        r = ref_feat[col].dropna().values
        c = curr_feat[col].dropna().values
        if len(r) < 20 or len(c) < 10:
            continue
        psi = compute_psi(r, c)
        results[col] = round(psi, 4)
        if psi >= PSI_CRITICAL:
            alerts.append(f"CRITICAL: {col} PSI={psi:.3f}")
        elif psi >= PSI_WARNING:
            alerts.append(f"WARNING: {col} PSI={psi:.3f}")

    log.info(f"Drift results: {results}")

    if any(v >= PSI_CRITICAL for v in results.values()):
        send_alert(
            subject="[CRITICAL] Market anomaly model — data drift detected",
            body="\n".join([
                "Significant drift detected. Retraining recommended.",
                "", *[f"  {k}: {v}" for k, v in results.items()],
                "", *alerts, "", "Run: python train.py",
            ]),
        )
    return results


if __name__ == "__main__":
    run_drift_check()