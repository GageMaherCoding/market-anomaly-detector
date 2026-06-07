import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import yfinance as yf
from kafka import KafkaProducer
from kafka.errors import KafkaError

from config import cfg

logging.basicConfig(level=cfg.log_level)
log = logging.getLogger(__name__)


@dataclass
class PriceEvent:
    captured_at: str
    ticker: str
    asset_type: str
    price: float
    volume: float
    prev_close: float
    day_high: float
    day_low: float


def classify(ticker: str) -> str:
    return "crypto" if ticker.endswith("-USD") else "equity"


def fetch_price(ticker: str) -> PriceEvent | None:
    try:
        data = yf.Ticker(ticker).fast_info
        price = getattr(data, "last_price", None)
        if price is None:
            return None
        return PriceEvent(
            captured_at=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            asset_type=classify(ticker),
            price=float(price),
            # Real-time traded volume. Falls back to the 10-day average only when
            # the live value is missing (e.g. market closed). The earlier code
            # stored three_month_average_volume, a near-constant figure, which
            # made every volume-based feature inert.
            volume=float(
                getattr(data, "last_volume", None)
                or getattr(data, "ten_day_average_volume", 0)
                or 0
            ),
            prev_close=float(getattr(data, "previous_close", 0) or 0),
            day_high=float(getattr(data, "day_high", 0) or 0),
            day_low=float(getattr(data, "day_low", 0) or 0),
        )
    except Exception as e:
        log.warning(f"Failed to fetch {ticker}: {e}")
        return None


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,
        acks="all",
        compression_type="gzip",
    )


def run():
    producer = make_producer()
    log.info(f"Producer started. Polling {len(cfg.tickers)} tickers every {cfg.poll_interval_seconds}s")

    while True:
        for ticker in cfg.tickers:
            event = fetch_price(ticker)
            if event is None:
                continue
            try:
                producer.send(
                    cfg.kafka.topic_raw,
                    key=ticker.encode(),
                    value=asdict(event),
                )
                log.info(f"{ticker}: ${event.price}")
            except KafkaError as e:
                log.error(f"Kafka error for {ticker}: {e}")

        producer.flush()
        time.sleep(cfg.poll_interval_seconds)


if __name__ == "__main__":
    run()