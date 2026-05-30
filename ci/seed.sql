-- Minimal sample data so the dbt models build and their tests pass in CI,
-- where no live producer is running. One equity + one crypto ticker, a handful
-- of timestamped rows each (enough for the rolling-window features).
INSERT INTO price_snapshots
    (ticker, asset_type, price, volume, prev_close, day_high, day_low, captured_at)
VALUES
    ('AAPL',   'equity', 150.00, 1000000, 149.00, 151.00, 148.50, NOW() - INTERVAL '60 minutes'),
    ('AAPL',   'equity', 150.50, 1000000, 149.00, 151.00, 148.50, NOW() - INTERVAL '50 minutes'),
    ('AAPL',   'equity', 151.00, 1000000, 149.00, 151.50, 148.50, NOW() - INTERVAL '40 minutes'),
    ('AAPL',   'equity', 150.80, 1000000, 149.00, 151.50, 148.50, NOW() - INTERVAL '30 minutes'),
    ('AAPL',   'equity', 152.00, 1000000, 149.00, 152.50, 148.50, NOW() - INTERVAL '20 minutes'),
    ('AAPL',   'equity', 151.50, 1000000, 149.00, 152.50, 148.50, NOW() - INTERVAL '10 minutes'),
    ('BTC-USD','crypto', 65000.0,    500, 64000.0, 66000.0, 63000.0, NOW() - INTERVAL '60 minutes'),
    ('BTC-USD','crypto', 65500.0,    500, 64000.0, 66000.0, 63000.0, NOW() - INTERVAL '50 minutes'),
    ('BTC-USD','crypto', 66000.0,    500, 64000.0, 66500.0, 63000.0, NOW() - INTERVAL '40 minutes'),
    ('BTC-USD','crypto', 65800.0,    500, 64000.0, 66500.0, 63000.0, NOW() - INTERVAL '30 minutes'),
    ('BTC-USD','crypto', 67000.0,    500, 64000.0, 67500.0, 63000.0, NOW() - INTERVAL '20 minutes'),
    ('BTC-USD','crypto', 66500.0,    500, 64000.0, 67500.0, 63000.0, NOW() - INTERVAL '10 minutes');
