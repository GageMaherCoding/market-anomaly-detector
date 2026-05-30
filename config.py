import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    topic_raw: str = "prices.raw"
    topic_anomalies: str = "prices.anomalies"
    group_id: str = "price-consumer-group"

@dataclass
class DBConfig:
    url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/anomalydb"
    )
    pool_size: int = 5

@dataclass
class AppConfig:
    kafka: KafkaConfig = None
    db: DBConfig = None
    poll_interval_seconds: int = 60
    rolling_window: int = 20
    z_score_threshold: float = 2.5
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    tickers: list = None

    def __post_init__(self):
        self.kafka = KafkaConfig()
        self.db = DBConfig()
        self.tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "META",
            "AMZN", "TSLA", "AMD",  "NFLX",  "SPY",
            "QQQ",  "JPM",  "BAC",  "GS",    "COIN",
            "PLTR", "SNOW", "UBER", "LYFT",  "ABNB",
            "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "BNB-USD",
        ]

cfg = AppConfig()