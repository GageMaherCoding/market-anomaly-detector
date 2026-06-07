import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from config import cfg

log = logging.getLogger(__name__)
app = FastAPI(title="Market Anomaly API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

ENGINE = create_engine(cfg.db.url, pool_size=cfg.db.pool_size)


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    timestamp: str

class AnomalyResponse(BaseModel):
    id: str
    detected_at: str
    ticker: str
    asset_type: str
    price_before: float
    price_after: float
    move_pct: float
    z_score: Optional[float]
    iso_score: Optional[float]

class PredictionResponse(BaseModel):
    id: str
    predicted_at: str
    ticker: Optional[str]
    is_anomaly: bool
    confidence: float
    z_score: Optional[float]
    model_version: str


@app.get("/health", response_model=HealthResponse)
def health():
    db_ok = False
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db_connected=db_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/anomalies", response_model=list[AnomalyResponse])
def get_anomalies(
    hours: int = Query(default=24, le=168),
    ticker: Optional[str] = Query(default=None),
    asset_type: Optional[str] = Query(default=None),
    min_move_pct: float = Query(default=0.5),
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    where_extra = ""
    params: dict = {"cutoff": cutoff, "min_move": min_move_pct}
    if ticker:
        where_extra += " AND ticker = :ticker"
        params["ticker"] = ticker
    if asset_type:
        where_extra += " AND asset_type = :asset_type"
        params["asset_type"] = asset_type

    query = f"""
        SELECT id::text, detected_at, ticker, asset_type,
               price_before, price_after, move_pct, z_score, iso_score
        FROM price_movements
        WHERE flagged = TRUE
          AND detected_at >= :cutoff
          AND ABS(move_pct) >= :min_move
          {where_extra}
        ORDER BY detected_at DESC
        LIMIT 200
    """
    with ENGINE.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return [AnomalyResponse(**dict(r._mapping)) for r in rows]


@app.get("/predictions", response_model=list[PredictionResponse])
def get_predictions(
    hours: int = Query(default=6, le=48),
    anomalies_only: bool = Query(default=False),
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = f"""
        SELECT p.id::text, p.predicted_at, s.ticker,
               p.is_anomaly, p.confidence, p.z_score, p.model_version
        FROM predictions p
        LEFT JOIN price_snapshots s ON s.id = p.snapshot_id
        WHERE p.predicted_at >= :cutoff
        {"AND p.is_anomaly = TRUE" if anomalies_only else ""}
        ORDER BY p.predicted_at DESC
        LIMIT 500
    """
    with ENGINE.connect() as conn:
        rows = conn.execute(text(query), {"cutoff": cutoff}).fetchall()
    return [PredictionResponse(**dict(r._mapping)) for r in rows]