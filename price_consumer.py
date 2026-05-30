import json
import logging

from kafka import KafkaConsumer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import cfg

logging.basicConfig(level=cfg.log_level)
log = logging.getLogger(__name__)

INSERT_SQL = text("""
    INSERT INTO price_snapshots
        (ticker, asset_type, price, volume, prev_close,
         day_high, day_low, captured_at, raw_payload)
    VALUES
        (:ticker, :asset_type, :price, :volume, :prev_close,
         :day_high, :day_low, :captured_at, :raw_payload)
""")


def run():
    engine = create_engine(cfg.db.url, pool_size=cfg.db.pool_size)
    Session = sessionmaker(bind=engine)

    consumer = KafkaConsumer(
        cfg.kafka.topic_raw,
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=cfg.kafka.group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_records=100,
    )

    log.info(f"Consumer started on topic={cfg.kafka.topic_raw}")

    for message in consumer:
        event = message.value
        session = Session()
        try:
            session.execute(INSERT_SQL, {
                "ticker":      event["ticker"],
                "asset_type":  event["asset_type"],
                "price":       event["price"],
                "volume":      event["volume"],
                "prev_close":  event["prev_close"],
                "day_high":    event["day_high"],
                "day_low":     event["day_low"],
                "captured_at": event["captured_at"],
                "raw_payload": json.dumps(event),
            })
            session.commit()
            consumer.commit()
        except Exception as e:
            session.rollback()
            log.error(f"Failed to write event: {e}")
        finally:
            session.close()


if __name__ == "__main__":
    run()